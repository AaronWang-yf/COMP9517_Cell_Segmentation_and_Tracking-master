import os
import numpy as np
import keras
import cv2
from .utils import Normalizer, load_flist, safe_quantization, find_sequences
from .config import Config
from .preprocessing import get_pixel_weight_function, preprocess_gt, smooth_components
from datetime import datetime


from .eidg import random_transform


class SampleGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self,
                 img_flist,
                 gt_flist,
                 config,
                 augmentation=True,
                 markers=False):

        """Initialization"""
        self.batch_size = config.BATCH_SIZE
        self.markers = markers
        self.indexes = np.arange(len(img_flist))
        self.seed = config.SEED
        self.norm = Normalizer(config.NORMALIZATION, config.UNEVEN_ILLUMINATION)
        self.dim = config.DIM
        self.dim_original = config.DIM_ORIGINAL
        self.shuffle = config.MODE == 1
        self.shrink = config.SHRINK
        self.marker_diameter = config.MARKER_DIAMETER
        self.weights = config.WEIGHTS
        self.gt_depth = 2  # - (config.PIXEL_WEIGHTS == 'NO_WEIGHTS')*1
        self.weight_function = get_pixel_weight_function(config.PIXEL_WEIGHTS)
        self.img_flist = img_flist
        self.gt_flist = gt_flist
        self.debug = config.DEBUG
        self.augmentation = augmentation
        self.rotation_range = 180
        self.zoom_range = 0.3
        self.fill_mode = 'reflect'
        self.flip = True
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        n_batches = int(np.ceil(len(self.img_flist) / self.batch_size))
        return n_batches

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        m, n, _ = self.dim
        mo, no, _ = self.dim_original

        seed = np.random.randint(0, np.iinfo(np.int32).max)

        x = np.zeros((len(indexes), m, n), dtype=np.float32)
        y = np.zeros((len(indexes), m, n), dtype=np.float32)
        w = np.ones((len(indexes), m, n), dtype=np.float32)

        for i, index in enumerate(indexes):
            # TODO: load images also in other formats (color, 16 bits)
            img = self.get_image(index)
            img = safe_quantization(img, dtype=np.uint8)
            img = (self.norm.make(img) - .5)
            x[i, :mo, :no, ...] = img

            gt = self.get_gt(index)
            gt = preprocess_gt(gt, shrink=self.shrink, markers=self.markers)
            gt = safe_quantization(gt, dtype=np.uint8)
            y[i, :mo, :no, ...] = gt

        if self.augmentation:
            x = random_transform(x,
                                 rotation_range=self.rotation_range,
                                 zoom_range=self.zoom_range,
                                 fill_mode=self.fill_mode,
                                 horizontal_flip=self.flip,
                                 sync_seed=seed)
            y = random_transform(y,
                                 rotation_range=self.rotation_range,
                                 zoom_range=self.zoom_range,
                                 fill_mode=self.fill_mode,
                                 horizontal_flip=self.flip,
                                 sync_seed=seed)
            y = y > 0
        x = np.expand_dims(x, axis=3)
        y = np.stack((y, w), axis=3)

        # Find list of IDs
        if self.debug > 2:
            """ store x and y to check quality """
            now = datetime.now().strftime("%H%M%S")
            np.save(f'x_{index}_{now}', x)
            np.save(f'y_{index}_{now}', y)

        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            # np.random.seed(self.seed)
            np.random.shuffle(self.indexes, )

    def get_image(self, index):
        img = cv2.imread(self.img_flist[index], cv2.IMREAD_GRAYSCALE)
        return img

    def get_gt(self, index):
        gt = cv2.imread(self.gt_flist[index], cv2.IMREAD_ANYDEPTH)
        return gt

    def get_all(self):
        batches_x = []
        batches_y =[]
        for i in range(len(self)):
            x, y = self[i]
            batches_x.append(x)
            batches_y.append(y)
        x = np.concatenate(batches_x, axis=0)
        y = np.concatenate(batches_y, axis=0)
        return x, y


class TestSampleGenerator(keras.utils.Sequence):
    """Generates testing data for Keras"""
    def __init__(self,
                 img_flist,
                 config):

        """Initialization"""
        self.batch_size = config.BATCH_SIZE
        self.img_flist = img_flist
        self.indexes = np.arange(len(self.img_flist))
        self.seed = config.SEED
        self.norm = Normalizer(config.NORMALIZATION, config.UNEVEN_ILLUMINATION)
        self.dim = config.DIM
        self.dim_original = config.DIM_ORIGINAL
        self.shuffle = config.MODE == 1
        self.shrink = config.SHRINK
        self.debug = config.DEBUG

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        length = np.ceil(len(self.img_flist) / self.batch_size)
        if self.debug:
            print(f'dataset length: {length}')
        return length.astype(np.int)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes = [i for i in indexes if i < len(self.img_flist)]

        x = np.zeros((len(indexes), *self.dim), dtype=np.float32)
        m, n, d = self.dim_original

        for i, index in enumerate(indexes):
            # TODO: load images also in other formats (color, 16 bits)
            img = self._read_image(index)
            img = self.norm.make(img)
            x[i, :m, :n, :d] = (img - .5).reshape(self.dim_original)
        if self.debug:
            print(x.shape, indexes)

        return x

    def get_all(self):
        batches = []
        for i in range(len(self)):
            batches.append(self[i])
        x = np.concatenate(batches, axis=0)
        return x

    def _read_image(self, index):
        img = cv2.imread(self.img_flist[index], cv2.IMREAD_GRAYSCALE)
        return img

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            # np.random.seed(self.seed)
            np.random.shuffle(self.indexes, )

    def get_image(self, index, crop=False):
        img = cv2.imread(self.img_flist[index], cv2.IMREAD_GRAYSCALE)
        if crop:
            img = self._crop_image(img)
        return img

    def _crop_image(self, image):
        mi, ni, _ = self.dim_original
        return image[:mi, :ni, ...]


class Dataset:
    """
    class to provide data in a proper format
    maintain data generator given by the config file
    dataset follows the structure of CTC datasets
    """
    def __init__(self,
                 config: Config = None,
                 markers: bool = False):
        self.config = config
        self.markers = markers
        self.mode = config.MODE
        self.name = config.DATASET_NAME
        self.out_paths = config.OUT_PATH
        self.seq = config.SEQUENCE
        self.seed = config.SEED
        self.batch_size = config.BATCH_SIZE
        self.pixel_weights = config.PIXEL_WEIGHTS
        self.dim = config.DIM
        self.dim_original = config.DIM_ORIGINAL
        self.marker_masks = config.MARKER_MASKS
        self.cell_masks = config.CELL_MASKS
        self.reference = config.REFERENCE
        self.digits = config.DIGITS
        self.data_path = config.DATA_PATH
        self.validation_sequence = config.VALIDATION_SEQUENCE
        self.generator = None

        #  get flists
        self.img_sequences = self._get_img_sequences()
        self.flist_img = None
        self.flist_gt = None
        self.flist_img_val = None
        self.flist_gt_val = None

        self.update_flists()

    def __len__(self):
        return len(self.flist_img)

    def get_validation_samples_generator(self):
        return SampleGenerator(img_flist=self.flist_img_val,
                               gt_flist=self.flist_gt_val,
                               config=self.config,
                               augmentation=False,
                               markers=self.markers)

    def get_training_samples_generator(self):
        return SampleGenerator(img_flist=self.flist_img,
                               gt_flist=self.flist_gt,
                               config=self.config,
                               augmentation=True,
                               markers=self.markers)

    def get_img_generator(self):
        return TestSampleGenerator(img_flist=self.flist_img,
                                   config=self.config)

    def get_batch(self, index):
        if self.generator in None:
            self.generator = self.get_img_generator()
        return self.generator[index]

    def _get_flists(self,
                    sequences=None,
                    shuffle=True):

        gt_flist = []
        img_flist = []

        if sequences is None:
            sequences = self.img_sequences

        for seq in sequences:

            img_path = os.path.join(self.config.DATA_PATH, self.name, seq)
            assert os.path.isdir(img_path), f'the following path do not exist: {img_path}'

            img_files = load_flist(img_path)

            if self.mode == 1:
                if self.markers:
                    gt_path = os.path.join(self.config.DATA_PATH,
                                           self.name,
                                           f'{seq}_{self.reference}',
                                           self.marker_masks)
                    # gt_path = self.marker_masks
                else:
                    gt_path = os.path.join(self.config.DATA_PATH,
                                           self.name,
                                           f'{seq}_{self.reference}',
                                           self.cell_masks)
                    # gt_path = self.cell_masks
                assert os.path.isdir(gt_path), f'the following path do not exist: {gt_path}'

                gts = load_flist(gt_path)
                index = [os.path.split(gt)[-1][-(self.digits + 4):-4] for gt in gts]
                img_files.sort()
                for i in index:
                    assert str.isdigit(i)
                    img_flist.append(img_files[int(i)])

                for g in gts:
                    gt_flist.append(g)
            else:
                img_flist = img_files

        assert len(img_flist) == len(gt_flist) or len(gt_flist) == 0, ''
        assert len(img_flist) > 0, 'no samples were found'
        return img_flist, gt_flist

    def _get_img_sequences(self):
        if self.mode == 1:
            # list all the possible sequence directories
            # exclude validation_seq

            dirs = find_sequences(os.path.join(self.data_path, self.name))
            if self.validation_sequence is not None:
                dirs.remove(self.validation_sequence)

            assert len(dirs) > 0, f'no training sequences were found in {self.data_path}'
            return dirs

        else:
            return [self.seq]

    def update_flists(self):
        self.flist_img, self.flist_gt = \
            self._get_flists()

        if self.mode == 1:
            if self.validation_sequence is None:
                total = len(self.flist_img)
                assert total >= 2, 'dataset with less than two samples' \
                                   'cannot be split into a training and' \
                                   'a validation part'
                split_index = np.ceil(total / 10).astype(np.int)

                # shuffle
                np.random.seed(self.seed)
                np.random.shuffle(self.flist_img, )
                np.random.seed(self.seed)
                np.random.shuffle(self.flist_gt, )

                self.flist_img_val = self.flist_img[:split_index]
                self.flist_gt_val = self.flist_gt[:split_index]
                self.flist_img = self.flist_img[split_index:]
                self.flist_gt = self.flist_gt[split_index:]

                # sort
                self.flist_img_val.sort()
                self.flist_gt_val.sort()
                self.flist_img.sort()
                self.flist_gt.sort()

            else:
                self.flist_img_val, self.flist_gt_val = \
                    self._get_flists(sequences=self.validation_sequence)

    def get_image(self, index):
        if self.generator is None:
            self.generator = self.get_img_generator()
        return self.generator.get_image(index)

    def save_all(self):
        if self.generator is None:
            self.generator = self.get_img_generator()
        x = self.get_all()
        np.save('reference_images.npy', x)

    def get_all(self):
        generator = self.get_training_samples_generator()
        return generator.get_all()
