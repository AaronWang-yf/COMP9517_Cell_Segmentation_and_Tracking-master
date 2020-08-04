import os
import numpy as np
import cv2
from scipy import ndimage


def rescale(image):
    return image / 255


def median(image):
    image_ = image / 255 + (.5 - np.median(image / 255))
    return np.maximum(np.minimum(image_, 1.), .0)


def standard_normalization(image):
    image_ = image / 255 + (.5 - (np.max(image / 255) - np.min(image / 255)))
    return np.maximum(np.minimum(image_, 1.), .0)


def clahe(image):
    cl = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    cl1 = cl.apply(image)
    return cl1 / 255


def centralized_clahe(image):
    cl = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
    image_ = cl.apply(image)
    return median(image_)


def hist_equalization(image):
    return cv2.equalizeHist(image) / 255


def get_normal_fce(normalization):
    if normalization == 'HE':
        return hist_equalization  # cv2.equalizeHist
    if normalization == 'CLAHE':
        return clahe
    if normalization == 'STD':
        return standard_normalization
    if normalization == 'RESCALE':
        return rescale
    if normalization == 'MEDIAN':
        return median
    if normalization == 'CENTCLAHE':
        return centralized_clahe
    else:
        Exception('normalization function was not picked')
    return None


def get_pixel_weight_function(pixel_weight):
    if pixel_weight == 'unet':
        return get_pixel_weight_unet  # exposure.equalize_hist
    if pixel_weight == 'distance':
        return get_pixel_weights_distance
    if pixel_weight == 'NO_WEIGHTS':
        return get_plain_weights
    else:
        Exception('normalization function was not picked')
    return None


def get_plain_weights(gt):
    '''
    returns image of ones

    '''

    weights = np.ones(gt.shape)

    return weights

def get_pixel_weights_distance(gt, markers, gap_size):
    '''
    best weights, defined from real GT
    1 + a * sum ( max (b - dist(pixel, marker), 0 )^2 )
    a = 0.075
    b = 40

    '''

    a = 0.075
    b = 20

    weight = np.zeros(gt.shape, np.float)

    objects = np.delete(np.unique(gt), [0])

    for i in objects:
        # find object
        obj = cv2.inRange(gt, int(i), int(i))

        obj_inv = ((obj == 0) * 255).astype(np.uint8)

        dmap = ndimage.distance_transform_edt(obj_inv)
        dmap = np.maximum(b - dmap, 0)
        weight += dmap

    all_objects = cv2.inRange(markers, 0, 0) == 0

    # remove pixels inside markers
    weight = np.maximum(weight - all_objects * 255 * 255, 0)

    weight = a * weight * 32

    return weight


def get_pixel_weight_unet(markers):
    '''
    best weights, defined from real GT
    1 + a * sum ( max (b - dsit(pixel, marker), 0 )^2 )
    a = 0.025
    b = 40

    '''

    a = 10
    b = 20

    _, gt = cv2.connectedComponents(markers)

    max_value = max(gt.shape)
    d1 = np.ones(gt.shape, np.uint32) * max_value
    d2 = np.ones(gt.shape, np.uint32) * max_value

    objects = np.delete(np.unique(gt), [0])

    for i in objects:
        # find object
        obj = cv2.inRange(gt, int(i), int(i))

        obj_inv = ((obj == 0) * 255).astype(np.uint8)

        dn = ndimage.distance_transform_edt(obj_inv)

        d_tmp = np.minimum(dn, d1)
        d_mask = (d_tmp == d1) * dn + (d_tmp != d1) * d1
        d2 = np.minimum(d_mask, d2)
        d1 = d_tmp

    weight = a * np.exp(-(d1 + d2) ** 2 / (2 * b ** 2)) * 32

    return weight


def get_image_size(path):
    names = os.listdir(path)
    name = names[0]
    o = cv2.imread(os.path.join(path, name), cv2.IMREAD_GRAYSCALE)
    return o.shape[0:2]


def remove_uneven_illumination(img, blur_kernel_size=501):
    """
    remove UE illumination by LPF
    :param img:
    :param blur_kernel_size:
    :return:
    """

    img_f = img.astype(np.float32)
    img_mean = np.mean(img_f)

    img_blur = cv2.GaussianBlur(img_f, (blur_kernel_size, blur_kernel_size), 0)
    result = np.maximum(np.minimum((img_f - img_blur) + img_mean, 255), 0).astype(np.int32)

    return result


def add_padding(mask, pad=1):
    m, n = mask.shape
    new_mask = np.zeros((m+(2*pad), n+(2*pad)), dtype=np.uint)
    new_mask[pad:-pad, pad:-pad] = mask
    return new_mask


def remove_padding(mask, pad=1):
    return mask[pad:-pad, pad:-pad]


def preprocess_gt(gt,
                  shrink=0,
                  markers=False):

    if shrink != 0 and markers:
        gt = gt.astype(np.uint16)
        gt = add_padding(gt)

        indexes = np.unique(gt)
        indexes.sort()
        indexes = indexes[1:]
        assert 0 not in indexes
        new_gt = np.zeros(gt.shape, dtype=np.uint)
        if len(indexes) > 1:
            for i in indexes:
                binary = np.squeeze((gt == i))
                new_gt += get_only_largest_component(binary) * i
        gt = shrink_markers(new_gt, shrink)

        gt = remove_padding(gt)

    gt = (gt > 0).astype(np.uint8)
    gt = cv2.medianBlur(gt.squeeze(), 5)

    return gt


def get_only_largest_component(bin_image):
    idx, cc = cv2.connectedComponents(bin_image.astype(np.uint8))

    if idx == 1:
        return bin_image
    max_size = 0
    max_index = 0
    for index in range(1, idx):
        component = cc == index

        if np.sum(component) > max_size:
            max_size = np.sum(component)
            max_index = index

    assert max_index != 0
    return cc == max_index


def shrink_markers(gt, m_size):
    assert 0 < m_size <= 100, m_size

    labels = np.unique(gt)
    result = np.zeros(gt.shape)

    for label in labels:

        # do not process background
        if label == 0:
            continue

        # pick marker
        mask = (gt == label).astype(np.uint8)
        mask = np.squeeze(mask)
        dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        # shrink marker
        max_shrink = np.floor(np.max(dt))
        thr = max_shrink * (100 - m_size) // 99
        marker = dt > thr
        assert np.sum(dt > thr) > 0

        # preserve only the largest component
        idx, _ = cv2.connectedComponents(marker.astype(np.uint8))
        if idx > 2:
            marker = get_only_largest_component(marker)
            # idx, _ = cv2.connectedComponents(marker.astype(np.uint8))
            # assert idx <= 2, 'there is more than one component'

        result += marker
    return result


def smooth_components(mask, marker_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (marker_size, marker_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def shrink_markers_inv(markers, m_size):
    assert 0 < m_size <= 100, m_size

    labels = np.unique(markers)
    result = np.zeros(markers.shape)

    for label in labels:

        # do not process background
        if label == 0:
            continue

        # pick marker
        marker = (markers == label).astype(np.uint8)
        dt = cv2.distanceTransform(marker, cv2.DIST_L2, 3)

        # shrink marker
        max_shrink = np.floor(np.max(dt))
        radius = int(np.floor((max_shrink / m_size) * 100 - max_shrink)) + 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        mask = cv2.dilate(marker, kernel)

        # TODO: remove overlaping
        free = (result == 0)
        assert not np.max(free) > 1, np.max(free)
        mask = mask * free

        result += mask * label
    return result


""" not used """


def load_images(path,
                cut=False,
                new_mi=0,
                new_ni=0,
                normalization='HE',
                uneven_illumination=False):
    names = os.listdir(path)
    names.sort()

    mi, ni = get_image_size(path)

    dm = (mi % 16) // 2
    mi16 = mi - mi % 16
    dn = (ni % 16) // 2
    ni16 = ni - ni % 16

    total = len(names)

    image = np.empty((total, mi, ni, 1), dtype=np.float32)

    normal_fce = get_normal_fce(normalization)

    for i, name in enumerate(names):

        o = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)

        if o is None:
            print('image {} was not loaded'.format(name))
            return None
        if type(o[0, 0]) == np.uint16:
            o = (o / 255).astype(np.uint8)
        if uneven_illumination:
            o = remove_uneven_illumination(o)

        o = np.minimum(o, 255).astype(np.uint8)

        image_ = normal_fce(o)
        image_ = image_.reshape((1, mi, ni, 1)) - .5
        image[i, :, :, :] = image_
    if cut:
        image = image[:, dm:mi16 + dm, dn:ni16 + dn, :]
    if new_ni > 0 and new_ni > 0:
        image2 = np.zeros((total, new_mi, new_ni, 1), dtype=np.float32)
        image2[:, :mi, :ni, :] = image
        image = image2
    print('loaded images from directory {} to shape {}'.format(path, image.shape))
    return image
