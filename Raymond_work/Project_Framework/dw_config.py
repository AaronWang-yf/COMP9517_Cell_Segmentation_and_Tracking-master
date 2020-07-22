from deepwater_inference.src.config import Config 

class DwConfig:
    def __init__(self,params):
        self.model_name = params.dataset
        self.name = params.dataset
        self.sequence = "01" # For Sequence x, you should input 0x
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

    print(f"the configueration file was loaded from {config_path}")
    config = Config(config_path)
    config.load_args(dw_config)

    # model_name = dw_config.model_name if dw_config.model_name is not None else dw_config.name
    config.MODEL_NAME = dw_config.model_name if dw_config.model_name is not None else dw_config.name

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