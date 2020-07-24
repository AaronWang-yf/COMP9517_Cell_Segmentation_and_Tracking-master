# COMP 9517 project
# python 3.7
# cv2 3.4
# pytorch

# To run this program, we assume that all files have existed and the parameters from the command line is correct,
# where the first parameter is the sequence number of the dataset
# and the second represents the sequence number in the given dataset


import glob
import os
import sys

import cv2

from detector import Detector
from drawer import Drawer
from matcher import Matcher
from preprocessor import Preprocessor


def main():

    # get the path based on the input parameters
    datasets = ['DIC-C2DH-HeLa', 'Fluo-N2DL-HeLa', 'PhC-C2DL-PSC']
    sequences = ['Sequence 1', 'Sequence 2', 'Sequence 3', 'Sequence 4', 'Sequence 5']
    dataset_num = int(sys.argv[1]) - 1
    seq_num = int(sys.argv[2]) - 1
    # path = './data/Fluo-N2DL-HeLa/Sequence 1/'
    path = './data/' + datasets[dataset_num] + '/' + sequences[seq_num] + '/'
    os.chdir(path)
    if not os.path.exists("gen"):
        os.mkdir("gen")

    # read all images in the file
    seq = []
    images = glob.glob('*.tif')
    for i in images:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        seq.append(image)

    # initialize classes for each dataset
    # also finish segmentation for all images
    if dataset_num == 0:
        preprocessor = Preprocessor(seq)
        detector = Detector(preprocessor)
        matcher = Matcher(detector)
        drawer = Drawer(matcher, preprocessor)
        masks = preprocessor.get_masks()

    elif dataset_num == 1:
        preprocessor = Preprocessor(seq)
        detector = Detector(preprocessor)
        matcher = Matcher(detector)
        drawer = Drawer(matcher, preprocessor)
        masks = preprocessor.get_masks()
    else:
        preprocessor = Preprocessor(seq)
        detector = Detector(preprocessor)
        matcher = Matcher(detector)
        drawer = Drawer(matcher, preprocessor)
        masks = preprocessor.get_masks()
    print('Generating all frames and cell states...')
    # based on the contours for all images, tracking trajectory and mitosis image by image
    drawer.load()
    print('Successfully loaded all images')

    # Save all generated images and their masks to disk
    counter = 1
    for g in drawer.get_gen_images():
        cv2.imwrite(f'./gen/{counter}.tif', g)
        cv2.imwrite(f'./gen/mask_{counter}.tif', masks[counter - 1])
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
            # plt.imshow(display_image)
            # plt.axis('off')
            # plt.show()
            cv2.imshow('image', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            break


if __name__ == '__main__':
    main()
