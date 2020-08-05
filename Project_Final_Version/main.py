"""
UNSW 20T2 COMP9517 Computer Vision Project
Group Name: Serve Auntie A Cup of Cappuccino
Group Members: Raymond Lu, Con Tieu-Vinh, Yunfan Wang, Xiaocong Chen, Shuang Liang

Acknowledgement to:
1.Tomas Sixta, J-Net, https://github.com/tsixta/jnet
2.Filip Lux, deepwater, https://gitlab.fi.muni.cz/xlux/deepwater
"""

"""
Package Specification:
1. Tensorflow 2.1.0
2. PyTorch 0.4.1.post2
"""

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

# Before running this program, please alter the settings in the param.py
def main():

    params = Params()

    if not os.path.isdir(params.dataset_root):
        raise Exception("Unable to load images from "+params.dataset_root+": not a directory")

    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir) 
    
    if not os.path.isdir(params.output_dir):
        raise Exception("Unable to save results to "+params.output_dir+": not a directory")

    if (params.dataset == "DIC-C2DH-HeLa"):
        path = params.dataset_root + "/"+ str(list(params.images_idx.keys())[0])
    elif (params.dataset=="PhC-C2DL-PSC" and params.nn_method=="DeepWater"):
        path = params.dataset_root + "/"+ str(list(params.images_idx.keys())[0])
    else:
        path = params.dataset_root

    print("The current data path is: ",path)
    images = glob.glob(path + '/*.tif')
    #sort the order of images
    images = [(int(x[-7:-4]),x) for x in images]
    images.sort(key=lambda x:x[0])
    images = [x[1] for x in images]
        
    preprocessor = Preprocessor(images,params)
    detector = Detector(preprocessor,params)
    matcher = Matcher(detector)
    drawer = Drawer(matcher,preprocessor)
    masks = preprocessor.get_masks()

    print('Generating all frames and cell states...')
    # Based on the contours for all images, tracking trajectory and mitosis image by image
    drawer.load()
    print('Successfully loaded all images')

    gen_path = path + "/gen"
    print("Now the gen path is ",gen_path)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
        print("gen path created successfully")

    counter = 1
    for g in drawer.get_gen_images():
        cv2.imwrite(f'{gen_path}/{counter}.tif', g)
        cv2.imwrite(f'{gen_path}/mask_{counter}.tif', masks[counter - 1])
        counter += 1
    print('Saved all images')

    # Now standby for user to issue commands for retrieval
    # input: num_1  num_2
    # 1st num represents the image number
    # 2nd num represents the cell number in the given image
    # 2nd num is not compulsory, with 1 number it will just show the image
    # press ENTER without any input will end the program
    # all inputs are assumed right

    analysis_path = path + "/analysis"
    if not os.path.exists(analysis_path):
      os.makedirs(analysis_path)
      print("analysis path created successfully")


    while True:
        frame = None 
        cell_id = None
        string = input('Input a frame and cell ID (optional) separated by a space...\n')
        if string:
            string = string.split(' ')
            frame = int(string[0])
            if len(string) > 1:
                try:
                    cell_id = int(string[1])
                    display_image = drawer.serve(frame, cell_id)
                except ValueError:
                    print(f'Not an integer')
                    display_image = drawer.serve(frame)
            else:
                display_image = drawer.serve(frame)

            file_name = analysis_path+"/frame_"+str(frame)+"_cellID_"+str(cell_id)+".tif"
            cv2.imwrite(file_name,display_image)

        else:
            break

if __name__ == '__main__':
    main()