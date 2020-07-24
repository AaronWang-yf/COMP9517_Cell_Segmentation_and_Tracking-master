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
        os.makedirs(params.output_dir) 
    
    if not os.path.isdir(params.output_dir):
        raise Exception("Unable to save results to "+params.output_dir+": not a directory")

    if (params.dataset == "DIC-C2DH-HeLa"):
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
    detector = Detector(preprocessor)
    matcher = Matcher(detector)
    drawer = Drawer(matcher)
    
    masks = preprocessor.get_masks()
    
    counter = 1
    while True:
        inp = input('Serving next frame... type a Cell ID to inspect details')
        drawer.next()
        try:
            inp = int(inp)
            display_image = drawer.serve(inp)
        except:
            print(f'Not an integer')
            display_image = drawer.serve()
            
        plt.imsave(path + f'gen/{counter}.jpg', display_image)
        plt.imsave(path + f'gen/{counter}_mask.jpg', masks[counter])
        counter += 1


if __name__ == '__main__':
    main()