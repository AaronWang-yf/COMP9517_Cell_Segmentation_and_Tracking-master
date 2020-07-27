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

# path = 'data/Fluo-N2DL-HeLa/Sequence 1/'



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
    print(path)
    images = glob.glob(path + '/*.tif')
    #sort the order of images
    images = [(int(x[-7:-4]),x) for x in images]
    images.sort(key=lambda x:x[0])
    images = [x[1] for x in images]
        
    preprocessor = Preprocessor(images,params)
    detector = Detector(preprocessor)
    matcher = Matcher(detector)
    drawer = Drawer(matcher,preprocessor)
    masks = preprocessor.get_masks()

    print('Generating all frames and cell states...')
    # based on the contours for all images, tracking trajectory and mitosis image by image
    drawer.load()
    print('Successfully loaded all images')

    counter = 1
    for g in drawer.get_gen_images():
        cv2.imwrite(f'{path}/gen/{counter}.tif', g)
        cv2.imwrite(f'{path}/gen/mask_{counter}.tif', masks[counter - 1])
        counter += 1
    print('Saved all images')

    # Now standby for user to issue commands for retrieval
    # input: num_1  num_2
    # 1st num represents the image number
    # 2nd num represents the cell number in the given image
    # 2nd num is not compulsory, with 1 number it will just show the image
    # press ENTER without any input will end the program
    # all inputs are assumed right
    while True:
        string = input('Input a frame and cell ID (optional) separated by a space...\n')
        if string:
            string = string.split(' ')
            frame = int(string[0])
            if len(string) > 1:
                try:
                    id = int(string[1])
                    display_image = drawer.serve(frame, id)
                except ValueError:
                    print(f'Not an integer')
                    display_image = drawer.serve(frame)
            else:
                display_image = drawer.serve(frame)
            plt.imshow(display_image)
            plt.axis('off')
            plt.show()
            # cv2.imshow('image', display_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        else:
            break

if __name__ == '__main__':
    main()