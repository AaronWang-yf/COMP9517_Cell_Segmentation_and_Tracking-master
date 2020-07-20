import glob

import cv2
import matplotlib.pyplot as plt

from detector import Detector
from drawer import Drawer
from matcher import Matcher
from preprocessor import Preprocessor

path = 'data/Fluo-N2DL-HeLa/'


def main():
    seq = []
    images = glob.glob(path + '*.tif')
    for i in images:
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        seq.append(image)

    preprocessor = Preprocessor(seq)
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

        cv2.imwrite(path + f'gen/{counter}.tif', display_image)
        cv2.imwrite(path + f'gen/{counter}_mask.tif', masks[counter])
        counter += 1


if __name__ == '__main__':
    main()
