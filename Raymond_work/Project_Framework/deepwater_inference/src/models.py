from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Lambda
from keras.optimizers import Adam, SGD
from .dataset import Dataset
# from tensorflow import set_random_seed
import tensorflow as tf
import os
from .loss_functions import cross_entropy_balanced, w_cren_2ch_bala
import sys
from tqdm import tqdm
import numpy as np

from .callbacks import ReduceLRCallback
from keras.callbacks import EarlyStopping


class UNetModel(Model):
    """
    instance of model, that predicts cell markers
    """
    def __init__(self, config, markers=False):
        self.config = config

        self.dim = config.DIM
        self.lr = config.LR
        self.feature_maps = config.FEATURE_MAPS
        self.downsampling_levels = config.DOWNSAMPLING_LEVELS
        self.output_layers = config.N_OUTPUTS
        self.debug = config.debug
        self.markers = markers
        self.loss_function = w_cren_2ch_bala
        self.steps_per_epoch = config.STEPS_PER_EPOCH
        self.epochs = config.EPOCHS

        tf.random.set_seed(2)
        # set_random_seed(2)
        input_layer, output_layer = self._create_unet()
        super(UNetModel, self).__init__(input_layer, output_layer)
        self.compile(optimizer=Adam(lr=self.lr), loss=self.loss_function)

    def _create_level(self, input_layer, level, n_maps):
        if level == 0:
            return input_layer
        else:
            c1 = Conv2D(n_maps, (3, 3), activation='relu', padding='same')(input_layer)
            pool = MaxPooling2D((2, 2), padding='same')(c1)
            c2 = Conv2D(n_maps*2, (3, 3), activation='relu', padding='same')(pool)

            down = self._create_level(c2, level - 1, n_maps * 2)

            c3 = Conv2D(n_maps*2, (3, 3), activation='relu', padding='same')(down)
            upsampling = UpSampling2D((2, 2), interpolation='bilinear')(c3)
            concatenate = Concatenate(axis=3)([upsampling, c1])
            c4 = Conv2D(n_maps, (3, 3), activation='relu', padding='same')(concatenate)

            return c4

    def _create_unet(self):
        input_layer = Input(shape=self.dim)
        n_maps = self.feature_maps
        level = self.downsampling_levels

        c1 = Conv2D(n_maps, (3, 3), activation='relu', padding='same')(input_layer)
        down = self._create_level(c1, level, n_maps)
        c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(down)

        # number of output layers
        outputs = []
        for _ in range(self.output_layers):
            out = Conv2D(2, (1, 1), activation='softmax', padding='same')(c2)
            outputs.append(out)

        # add more tasks together, if necessary
        if len(outputs) == 1:
            output_layer = outputs[0]
        else:
            output_layer = Concatenate(axis=3)(outputs)
        return input_layer, output_layer

    def load(self, model_path):
        print(model_path)
        if os.path.isfile(model_path):
            self.load_weights(model_path)

    def train_model(self, dataset: Dataset):
        val_generator = dataset.get_validation_samples_generator()
        train_generator = dataset.get_training_samples_generator()

        callbacks = []
        reducelr = ReduceLRCallback(
            factor=0.3,
            cooldown=5,
            patience=5,
            min_delta=1e-4,
            min_lr=0,
            verbose=0,)
        callbacks.append(reducelr)

        earlystop = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=True,
            mode="auto",
            baseline=None,
            )
        callbacks.append(earlystop)

        self.fit_generator(train_generator,
                           steps_per_epoch=self.steps_per_epoch,
                           epochs=self.epochs,
                           callbacks=callbacks,
                           validation_data=val_generator,
                           use_multiprocessing=True,
                           workers=5)

    def predict_dataset(self, dataset: Dataset, batch_index=None):
        generator = dataset.get_img_generator()

        if batch_index is None:

            prediction = self.predict_generator(generator,
                                                verbose=True,
                                                use_multiprocessing=True,
                                                workers=10)

        else:
            x = generator[batch_index]
            prediction = self.predict(x)

        m, n, _ = self.config.DIM_ORIGINAL
        prediction = prediction[:, :m, :n, :]
        return prediction
