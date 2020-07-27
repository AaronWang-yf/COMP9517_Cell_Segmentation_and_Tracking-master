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
import numpy as np

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
    elif (params.dataset=="PhC-C2DL-PSC" and params.nn_method=="DeepWater"):
        path = params.dataset_root + "/"+ str(list(params.images_idx.keys())[0])
    else:
        path = params.dataset_root
    # seq = []
    images = glob.glob(path + '/*.tif')
    #sort the order of images
    images = [(int(x[-7:-4]),x) for x in images]
    images.sort(key=lambda x:x[0])
    images = [x[1] for x in images]

    preprocessor = Preprocessor(images,params)

    print("Now the length of masks is: ",len(preprocessor.get_masks()))
    masks = preprocessor.get_masks() 
    m0 = masks[0] 
    m1 = masks[1]
    image,mask = preprocessor.next()
    cv2.imwrite("test_image.tif",image) 
    cv2.imwrite("test_mask.tif",mask.astype(np.uint16)) 
    cv2.imwrite("m0.tif",m0.astype(np.uint16)) 
    cv2.imwrite("m1.tif",m1.astype(np.uint16))


main()