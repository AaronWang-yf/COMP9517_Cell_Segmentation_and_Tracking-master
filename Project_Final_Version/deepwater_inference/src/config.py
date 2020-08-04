import os
import yaml
from src.utils import find_sequences


class Config(dict):
    def __init__(self, config_path: str):

        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    # get atributes from config
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        return DEFAULT_CONFIG.get(name, None)

    def load_args(self, args):

        # DEFAULT PARAMETERS
        # cannot be set by config
        self._dict['CONFIG_PATH'] = args.checkpoint_path
        self._dict['DATA_PATH'] = args.data_path

        # optional args
        # MODE
        if args.mode is not None:
            self._dict['MODE'] = args.mode
        # DATASET_NAME
        if args.name is not None:
            self._dict['DATASET_NAME'] = args.name
        # MODEL NAME
        self._dict['MODEL_NAME'] = args.model_name
        if args.model_name is None:
            self._dict['MODEL_NAME'] = args.name
        # SEQUENCE
        if args.sequence is not None:
            self._dict['SEQUENCE'] = args.sequence

        if self._dict['MODE'] > 1:
            if self.SEQUENCE is None:
                sequences = find_sequences(os.path.join(self.DATA_PATH, self.DATASET_NAME))
                print(f'ERROR: select one of the available sequences: {sequences}')
                print(f'use argument --sequence "sequence_number"')
                exit()
            self._dict['TEST_PATH'] = os.path.join(self.DATA_PATH, self.DATASET_NAME, self.SEQUENCE)
            self._dict['OUT_PATH'] = os.path.join(self.DATA_PATH, self.DATASET_NAME, self.SEQUENCE + '_RES')
            self._dict['VIZ_PATH'] = os.path.join(self.DATA_PATH, self.DATASET_NAME, self.SEQUENCE + '_VIZ')

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    # mode 1 TRAIN
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'MODEL': 1,                     # 1: marker model, 2: foreground model
    'MARKERS': 3,                   # 1: eroded markers, 2: weak markers
    'SEED': 10,                     # random seed
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console
    'MARKERS_ANNOTATION': 'weak',   # weak: not touching markers, full: markers from full annotations
    'VERSION': 1.1,                 # DW version

    # models
    'TRAIN_MARKERS': True,
    'TRAIN_FOREGROUND': True,

    # gt postprocessing
    'SHRINK': 0,
    'WEIGHTS': False,

    'LR': 0.0001,                           # learning rate
    'BETA1': 0.0,                           # adam optimizer beta1
    'BETA2': 0.9,                           # adam optimizer beta2
    'BATCH_SIZE': 16,                        # input batch size for training
    'INPUT_SIZE': 0,                        # input image size for training 0 for original size
    'STEPS_PER_EPOCH': 10,                  # batches in one epoch
    'EPOCHS': 5,                            # maximal number of training epochs

    'LOSS_MARKERS': 'mse',                  # training loss - markers
    'LOSS_FOREGROUND': 'mse',               # training loss - foreground
    'MODEL_NAME': 'DIC-C2DH-HeLa',          # model name for an evaluation
    'CONFIGURATION_PATH': 'checkpoints',    # path to the folder with configuration files
    'DATA_PATH': 'datasets',                # path to images
    'CELL_MASKS': 'SEG',             # folder with semantic segmentations
    'MARKER_MASKS': 'TRA',                 # 'SEG': full markers, 'TRA': detection markers
    'MERGE_METHOD': 'watershed',
    'N_OUTPUTS': 1,                          # number of output layers
    'MODEL_MARKER_PATH': 'model_markers',
    'MODEL_FOREGROUND_PATH': 'model_foreground',
    'NEW_MODEL': False,

    # model architecture
    'FEATURE_MAPS': 32,                 # feature maps in the first level
    'DOWNSAMPLING_LEVELS': 4,           # number of U-NET down-sampling steps
    'PIXEL_WEIGHTS': 'NO_WEIGHTS',      # NO_WEIGHTS: no weights; UNET: unet weights; DIST: based on distance

    # segmentation postprocessing
    'FRAME_BORDER': 0,
    'DISPLAY_RESULTS': True,            # save colorful results in XX_VIZ folder

    # debug
    'TESTMODE': False,                  # store dataset images for given training setting
    # tracking
    'TRACKING': True,                # consistent mask labeling through the sequence
}
