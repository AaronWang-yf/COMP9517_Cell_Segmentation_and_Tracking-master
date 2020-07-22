import os
import numpy as np
from .preprocessing import get_normal_fce, remove_uneven_illumination
import cv2
import glob
import requests
import zipfile


class Normalizer:
    def __init__(self, normalization: str, uneven_illumination: bool = False):
        self.normalfce = get_normal_fce(normalization)
        self.uneven_illumination = uneven_illumination
        self.normalization = normalization

    def make(self, image):
        image = np.squeeze(image)
        if self.uneven_illumination:
            image = remove_uneven_illumination(image)
        return self.normalfce(image)

    def get_normalfce(self):
        return self.normalfce


def get_image_shape(img_path):
    # get one file
    if os.path.isdir(img_path):
        names = os.listdir(img_path)
        names = [name for name in names if name[-4:] in ['.png', '.tif']]
        img_path = os.path.join(img_path, names[0])

    if os.path.isfile(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        assert img is not None, f'ER: cannot read image file: {img_path}'
        if len(img.shape) == 2:
            return img.shape[0], img.shape[1], 1
        return img.shape

    return None


def get_divisible_shape(original_size, divisor):
    mi, ni, depth = original_size
    new_mi = mi - mi % divisor
    if mi % divisor != 0:
        new_mi += divisor
    new_ni = ni - ni % divisor
    if ni % divisor != 0:
        new_ni += divisor
    return new_mi, new_ni, depth


def get_formatted_shape(img_path, divisor=16):
    """
    returns new and original image dimension
    new variables are declared to be divisible by 'divisor'

    :param img_path: path to an image directory, or image file path
    :param divisor: declared divisor of new dimensions
    :return: (new_n, new_m, n_channels), (original_m, original_n, n_channels)
    """

    original_size = get_image_shape(img_path)
    new_size = get_divisible_shape(original_size, divisor)

    return new_size, original_size


def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            files = os.listdir(flist)
            files = [os.path.join(flist, f) for f in files if f[-4:] in ['.jpg', '.png', '.tif']]
            files.sort()
            return files

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]

    return []


def clean_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        return
    files = glob.glob(os.path.join(dir_path, '*'))
    for f in files:
        os.remove(f)


def download_pretrained_model(model_name):
    print(f'downloading pretrained model: {model_name} ...')
    url = f'https://www.fi.muni.cz/~xlux/deepwater/{model_name}.zip'

    # check if the url exists
    request = requests.get(url)
    if request.status_code != 200:
        print(f'Pretrained model {model_name} is not available.')
        return False

    zip_file = f'checkpoints/{model_name}.zip'

    myfile = requests.get(url)
    open(zip_file, 'wb').write(myfile.content)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(f'checkpoints')

    os.remove(zip_file)

    return True


def overlay_labels(o, labels):
    labels = np.squeeze(labels)
    o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
    labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)

    fg_mask = (labels != 0).astype(np.uint8)
    bg_mask = (labels == 0).astype(np.uint8)

    labels_rgb[:, :, 0] = labels_rgb[:, :, 0] * fg_mask
    labels_rgb[:, :, 0] = labels_rgb[:, :, 0] + bg_mask * 180
    labels_rgb[:, :, 1] = labels_rgb[:, :, 1] + bg_mask * 180
    labels_rgb[:, :, 2] = labels_rgb[:, :, 2] + bg_mask * 180

    return cv2.addWeighted(o_rgb.astype(np.uint8), 0.7, labels_rgb, 0.3, 0)


def create_tracking(path, output_path, threshold=0.15):

    # check if path exists
    if not os.path.isdir(path):
        print('input path is not a valid path')
        return

    names = os.listdir(path)
    names = [name for name in names if '.tif' in name and 'mask' in name]
    names.sort()

    img = cv2.imread(os.path.join(path, names[0]), cv2.IMREAD_ANYDEPTH)
    mi, ni = img.shape
    print('Relabelling the segmentation masks.')
    records = {}


    old = np.zeros((mi, ni))
    index = 1
    n_images = len(names)

    for i, name in enumerate(names):
        result = np.zeros((mi, ni), np.uint16)

        img = cv2.imread(os.path.join(path, name), cv2.IMREAD_ANYDEPTH)

        labels = np.unique(img)[1:]

        parent_cells = []

        for label in labels:
            mask = (img == label) * 1

            mask_size = np.sum(mask)
            overlap = mask * old
            candidates = np.unique(overlap)[1:]

            max_score = 0
            max_candidate = 0

            for candidate in candidates:
                score = np.sum(overlap == candidate * 1) / mask_size
                if score > max_score:
                    max_score = score
                    max_candidate = candidate

            if max_score < threshold:
                # no parent cell detected, create new track

                records[index] = [i, i, 0]
                result = result + mask * index
                index += 1
            else:

                if max_candidate not in parent_cells:
                    # prolonging track
                    records[max_candidate][1] = i
                    result = result + mask * max_candidate

                else:
                    # split operations
                    # if have not been done yet, modify original record
                    if records[max_candidate][1] == i:
                        records[max_candidate][1] = i - 1
                        # find mask with max_candidate label in the result and rewrite it to index
                        m_mask = (result == max_candidate) * 1
                        result = result - m_mask * max_candidate + m_mask * index

                        records[index] = [i, i, max_candidate.astype(np.uint16)]
                        index += 1

                    # create new record with parent cell max_candidate
                    records[index] = [i, i, max_candidate.astype(np.uint16)]
                    result = result + mask * index
                    index += 1

                # update of used parent cells
                parent_cells.append(max_candidate)
        # store result
        cv2.imwrite(os.path.join(output_path, name), result.astype(np.uint16))
        old = result

    # store tracking
    print('Generating the tracking file.')
    with open(os.path.join(output_path, 'res_track.txt'), "w") as file:
        for key in records.keys():
            file.write('{} {} {} {}\n'.format(key, records[key][0], records[key][1], records[key][2]))


def remove_edge_cells(label_img, border=20):
    if (border is None) or (border == 0):
        return label_img
    edge_indexes = get_edge_indexes(label_img, border=border)
    return remove_indexed_cells(label_img, edge_indexes)


def get_edge_indexes(label_img, border=20):
    mask = np.ones(label_img.shape)
    mi, ni = mask.shape
    mask[border:mi - border, border:ni - border] = 0
    border_cells = mask * label_img
    indexes = (np.unique(border_cells))

    result = []

    # get only cells with center inside the mask
    for index in indexes:
        cell_size = sum(sum(label_img == index))
        gap_size = sum(sum(border_cells == index))
        if cell_size * 0.5 < gap_size:
            result.append(index)

    return result


def remove_indexed_cells(label_img, indexes):
    mask = np.ones(label_img.shape)
    for i in indexes:
        mask -= (label_img == i)
    return label_img * mask


def safe_quantization(img, dtype=np.uint8):
    if dtype == np.uint8:
        img = np.maximum(img, 0)
        img = np.minimum(img, 255)

    return img.astype(dtype)


def find_sequences(path):
    dirs = os.listdir(path)
    dirs = [d for d in dirs if len(d) == 2 and d.isdigit()]
    return dirs
