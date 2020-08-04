import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import watershed


def merge_seg_fce_marker_fce(marker_function=None,
                             segmentation_function=None,
                             foreground=None,
                             merge_method='watershed'):
    assert merge_method in ['watershed', 'topdist', 'eucldist'], merge_method

    if merge_method == 'watershed':
        labels = watershed_merge(segmentation_function,
                                 marker_function,
                                 foreground)
    elif merge_method == 'topdist':
        labels = topdist_merge(marker_function,
                               foreground)
    elif merge_method == 'eucldist':
        labels = eucldist_merge(marker_function,
                                foreground)
    else:
        labels = None
    return labels


def postprocess_markers(img,
                        threshold=240,
                        c=12,
                        dic=False,
                        h=4):
    """
    erosion_size == c
    step == h
    threshold == tm
    """

    # original matlab code:
    # res = opening(img, size); % size filtering
    # res = hconvex(res, h) == h; % local contrast filtering
    # res = res & (img >= t); % absolute intensity filtering

    if dic:
        # old version of postprocessing, DIC dataset
        m = img.astype(np.uint8)
        _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

        # filling gaps
        hol = binary_fill_holes(new_m * 255).astype(np.uint8)

        # morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c))
        glob_f = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c))
        markers = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        new_m = (hconvex(markers, h) == h).astype(np.uint8)
        glob_f = ((markers > threshold).astype(np.uint8) * new_m)

    # label connected components
    idx, markers = cv2.connectedComponents(glob_f)

    # print(threshold, c, circular, h)
    return idx, markers


# postprocess markers
# PhC and SIM+ compatibility
def postprocess_markers_09(img,
                           threshold=240,
                           c=12,
                           dic=False,
                           h=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    markers = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    new_m = ((hconvex(markers, h) > 0) & (img > threshold)).astype(np.uint8)
    idx, res = cv2.connectedComponents(new_m)
    return idx, res


def hmax(img, h=50):
    h_img = img.astype(np.uint16) + h
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rec0 = img

    # reconstruction
    for i in range(255):
        rec1 = np.minimum(cv2.dilate(rec0, kernel), h_img)
        if np.sum(rec0 - rec1) == 0:
            break
        rec0 = rec1

    # retype to uint8
    hmax_result = np.maximum(np.minimum((rec1 - h), 255), 0).astype(np.uint8)
    return hmax_result


def hconvex(img, h=5):
    return img - hmax(img, h)


# postprocess foreground
def postprocess_foreground(b, threshold=230):
    # tresholding
    bt = cv2.inRange(b, int(threshold), 255)

    return bt


def watershed_merge(segmentation_fce, marker_fce, foreground):
    result = watershed(segmentation_fce, marker_fce, mask=foreground)
    return result


def topdist_merge(marker_fc, cell_mask):
    dt = cv2.distanceTransform((marker_fc == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dtws = watershed(dt, marker_fc, mask=cell_mask)
    return dtws


def eucldist_merge(marker_fc, cell_mask):
    mi, ni = marker_fc.shape[:2]

    free = (marker_fc == 0)

    labels = np.unique(marker_fc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    result = marker_fc
    progress = True

    while progress:
        loss = np.sum(free)

        new_result = np.zeros((mi, ni))

        for index in labels:
            mask = result == index
            new_mask = cv2.dilate(mask.astype(np.uint8), kernel)
            new_mask = new_mask * (free == 1)
            new_result += (mask + new_mask) * index
            free = free & (new_result == 0)

        progress = loss != np.sum(free)

        result = new_result
    return (result * cell_mask // 255).astype(np.uint8)


# OLD FUNCTIONS

# postprocess cell mask
def postprocess_cell_mask(b, threshold=230):
    # tresholding
    bt = cv2.inRange(b, int(threshold), 255)

    return bt


# postprocess markers
def postprocess_markers_old(img,
                        threshold=240,
                        c=12,
                        circular=True,
                        h=4):
    """
    erosion_size == c
    step == h
    threshold == tm
    """

    # distance transform | only for circular objects
    if circular:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c))
        # markers = cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        markers = cv2.dilate(img.astype(np.uint8), kernel)
        new_m = ((hconvex(markers, h) > 0) & (img > threshold)).astype(np.uint8)
    else:
        # old version of postprocessing
        # threshold
        m = img.astype(np.uint8)
        _, new_m = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)

        # filling gaps
        hol = binary_fill_holes(new_m * 255).astype(np.uint8)

        # morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, c))
        new_m = cv2.morphologyEx(hol, cv2.MORPH_OPEN, kernel)

    # label connected components
    idx, res = cv2.connectedComponents(new_m)

    return idx, res
