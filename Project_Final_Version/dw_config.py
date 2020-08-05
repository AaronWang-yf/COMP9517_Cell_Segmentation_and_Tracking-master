# WARNING: THIS FILE IS NOT SUPPOSED TO BE MODIFIED!!!
# In this file, absolute path is required.
from deepwater_inference.src.config import Config 
import os
from shutil import copyfile

class DwConfig:
    def __init__(self,params):
        self.model_name = params.dataset
        self.name = params.dataset
        self.sequence = str(list(params.images_idx.keys())[0]) # For Sequence x, you should input 0x
        self.checkpoint_path = "./deepwater_inference/checkpoints"
        self.data_path = "./datasets"
        self.dw_path = "."
        self.mode = 2 # mode 2 for segmentation
        self.validation_sequence = None 
        self.one_network = None 
        self.new_model = False 
        self.annotations = None 
        self.test_settings = False 

def load_config(params,mode=None):
    # data_path, checkpoint_path
    dw_config = DwConfig(params)
    if not os.path.exists(dw_config.data_path):
        os.makedirs(dw_config.data_path)
    if not os.path.exists(dw_config.data_path):
        os.makedirs(dw_config.data_path)

    # config path definition
    example_config_path = "./deepwater_inference/config_example.yml"
    data_config_path = os.path.join(dw_config.data_path, dw_config.name, "config.yml")
    checkpoint_config_path = os.path.join(dw_config.checkpoint_path, dw_config.name, "config.yml")

    # read config
    if dw_config.model_name is not None:
        # redefine checkpoint config path
        checkpoint_config_path = os.path.join(dw_config.checkpoint_path, dw_config.model_name, "config.yml")
        assert os.path.isfile(checkpoint_config_path), f"{checkpoint_config_path} do not exist"

        # load config file
        config_path = checkpoint_config_path

    else:
        if os.path.isfile(data_config_path):
            pass
        elif os.path.isfile(checkpoint_config_path):
            copyfile(checkpoint_config_path, data_config_path)
        else:
            copyfile(example_config_path, data_config_path)
        config_path = data_config_path
    
    # if os.path.isfile(data_config_path):
    #     pass
    # elif os.path.isfile(checkpoint_config_path):
    #     copyfile(checkpoint_config_path, data_config_path)
    # else:
    #     copyfile(example_config_path, data_config_path)
    # config_path = data_config_path


    print(f"the configuration file was loaded from {config_path}")
    config = Config(config_path)
    config.load_args(dw_config)

    # model_name = dw_config.model_name if dw_config.model_name is not None else dw_config.name
    config.MODEL_NAME = dw_config.model_name if dw_config.model_name is not None else dw_config.name

    config.MODEL_MARKER_PATH = dw_config.checkpoint_path + "/" + dw_config.model_name

    # create a configuration path if doesn't exist
    if not os.path.exists(dw_config.checkpoint_path):
        os.makedirs(dw_config.checkpoint_path)
    if not os.path.exists(os.path.join(dw_config.checkpoint_path, config.MODEL_NAME)):
        os.makedirs(os.path.join(dw_config.checkpoint_path, config.MODEL_NAME))

    # for each mode we add different info to config
    # train mode
    if config.MODE == 1:

        config.IMG_PATH = os.path.join(config.DATA_PATH, config.DATASET_NAME)
        # TODO: include different sources of gt images / markers

        config.VAL_SEQUENCE = dw_config.validation_sequence
        # TODO: parse another specific training arguments from commandline
        config.NEW_MODEL = dw_config.new_model

        if dw_config.one_network is not None:
            if dw_config.one_network == 'markers':
                config.TRAIN_FOREGROUND = False
            if dw_config.one_network == 'foreground':
                config.TRAIN_MARKERS = False

        if dw_config.annotations is not None:
            config.MARKER_ANNOTATIONS = dw_config.annotations

        if config.MARKER_ANNOTATIONS == 'weak':
            if not str(config.MARKER_DIAMETER).isdigit() or config.MARKER_DIAMETER == 0:
                print('set MARKER_DIAMETER in config file.')
                exit()
            config.SHRINK = 0

        if config.MARKER_ANNOTATIONS == 'full':
            if not str(config.CELL_DIAMETER).isdigit() or config.CELL_DIAMETER == 0:
                print('set CELL_DIAMETER in config file.')
                exit()
            config.SHRINK = 70
            config.MARKER_DIAMETER = np.ceil((1 - (config.SHRINK / 100)) * config.CELL_DIAMETER).astype(np.uint8)

        config.TESTMODE = dw_config.test_settings

    # test mode
    elif config.MODE == 2:
        config.IMG_PATH = os.path.join(config.DATA_PATH, config.DATASET_NAME, config.SEQUENCE)
        config.TEST_PATH = config.IMG_PATH  # merge variables

    # eval mode
    elif mode == 3:
        pass

    return config



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
    'CONFIGURATION_PATH': './deepwater_inference/checkpoints',    # path to the folder with configuration files
    'DATA_PATH': 'datasets',                # path to images
    'CELL_MASKS': 'SEG',             # folder with semantic segmentations
    'MARKER_MASKS': 'TRA',                 # 'SEG': full markers, 'TRA': detection markers
    'MERGE_METHOD': 'watershed',
    'N_OUTPUTS': 1,                          # number of output layers
    #'MODEL_MARKER_PATH': './deepwater_inference/checkpoints',
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