import argparse
import os
from shutil import copyfile
from src.config import Config
from src.deepwater_object import DeepWater
import numpy as np

# turn off tf verbosity
if True:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def deepwater(mode=2):
    config = load_config(mode)

    # load model
    model = DeepWater(config)

    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart segmentation...\n')
        model.test()

    # eval mode
    elif config.MODE == 3:
        print('\nstart eval...\n')
        model.eval()

    else:
        print('\nERR: incorrect mode\n')


def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        help='pre-trained model name')
    parser.add_argument('--name',
                        type=str,
                        default=None,
                        help='name of the dataset')
    parser.add_argument('--sequence',
                        type=str,
                        help='sequence number (01, 02, ...)')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='checkpoints',
                        help='path to stored models (default: ./checkpoints)')
    parser.add_argument('--data_path',
                        type=str,
                        default='datasets',
                        help='path to directory with datasets (default: ./datasets)')
    parser.add_argument('--dw_path',
                        type=str,
                        default='.',
                        help='path to directory with deepwater')
    parser.add_argument('--mode',
                        type=int,
                        default=mode,
                        choices=[1, 2, 3],
                        help='mode of the program run')
    parser.add_argument('--validation_sequence',
                        type=str,
                        default=None,
                        help='sequence to validate the training. /'
                             'If it is not specified, model is validated on randomly picked 10% of training samples')
    parser.add_argument('--one_network',
                        type=str,
                        choices=['markers', 'foreground'],
                        default=None,
                        help='choose only one model to train')
    parser.add_argument('--new_model',
                        type=str2bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='choose to train new model from scratch')
    parser.add_argument('--annotations',
                        type=str,
                        choices=['full', 'weak'],
                        default=None,
                        help='markers from a full or a weak annotations')
    parser.add_argument('--test_settings',
                        type=str2bool,
                        nargs='?',
                        default=False,
                        const=True,
                        help='true to test an image pre-processing')

    args = parser.parse_args()

    # data_path, checkpoint_path
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # config path definition
    example_config_path = "config_example.yml"
    data_config_path = os.path.join(args.data_path, args.name, "config.yml")
    checkpoint_config_path = os.path.join(args.checkpoint_path, args.name, "config.yml")

    # download_pretrained_model - all are included
    # if not os.path.isdir(dataset_path):
    #     if not download_pretrained_model(config.MODEL_NAME):
    #         os.makedirs(dataset_path)

    # read config
    if args.model_name is not None:
        # redefine checkpoint config path
        checkpoint_config_path = os.path.join(args.checkpoint_path, args.model_name, "config.yml")
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
    config.load_args(args)

    # model_name = args.model_name if args.model_name is not None else args.name
    config.MODEL_NAME = args.model_name if args.model_name is not None else args.name

    # create a configuration path if doesn't exist
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.exists(os.path.join(args.checkpoint_path, config.MODEL_NAME)):
        os.makedirs(os.path.join(args.checkpoint_path, config.MODEL_NAME))

    # for each mode we add different info to config
    # train mode
    if config.MODE == 1:

        config.IMG_PATH = os.path.join(config.DATA_PATH, config.DATASET_NAME)
        # TODO: include different sources of gt images / markers

        config.VAL_SEQUENCE = args.validation_sequence
        # TODO: parse another specific training arguments from commandline
        config.NEW_MODEL = args.new_model

        if args.one_network is not None:
            if args.one_network == 'markers':
                config.TRAIN_FOREGROUND = False
            if args.one_network == 'foreground':
                config.TRAIN_MARKERS = False

        if args.annotations is not None:
            config.MARKER_ANNOTATIONS = args.annotations

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

        config.TESTMODE = args.test_settings

    # test mode
    elif config.MODE == 2:
        config.IMG_PATH = os.path.join(config.DATA_PATH, config.DATASET_NAME, config.SEQUENCE)
        config.TEST_PATH = config.IMG_PATH  # merge variables

    # eval mode
    elif mode == 3:
        pass

    return config


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    deepwater()
