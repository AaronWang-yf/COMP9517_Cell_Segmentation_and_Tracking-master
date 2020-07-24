from preprocessor import Preprocessor
from detector import Detector
from matcher import Matcher
from drawer import Drawer
from cell import Cell
import glob
import cv2
import matplotlib.pyplot as plt
import time
from param import Params 
import os

def main():
    params = Params() 
    if not os.path.isdir(params.dataset_root):
        raise Exception("Unable to load images from "+params.dataset_root+": not a directory")

    if not os.path.exists(params.output_dir):
        os.mkdir(params.output_dir) 
    
    if not os.path.isdir(params.output_dir):
        raise Exception("Unable to save results to "+params.output_dir+": not a directory")

    if (params.dataset == "DIC-C2DH-HeLa"):
        path = params.dataset_root + "/"+ str(list(params.images_idx.keys())[0])
    else:
        path = params.dataset_root
    # seq = []
    images = glob.glob(path + '*.tif')

    preprocessor = Preprocessor(images,params)

    print(len(preprocessor.get_masks()))
    image,mask = preprocessor.next()
    plt.imshow(image) 
    plt.imshow(mask)

main()