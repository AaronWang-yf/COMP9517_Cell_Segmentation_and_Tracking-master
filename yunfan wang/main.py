import os
import project as po
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "./data/Fluo-N2DL-HeLa/Sequence 5"

file_lst = []
os.chdir(path)
for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if name.endswith("tif"):
            file_lst.append(os.path.join(root, name))
if not os.path.exists("mask"):
    os.mkdir("mask")
if not os.path.exists("tracking"):
    os.mkdir("tracking")

# to store all cells info for all images
cells_set = []

### task 1 and 2

for img_path in files:
    img_path = ".\\" + img_path
    cells = po.cells()   # init struct cells
    # get contours and masks
    detector = po.detector(img_path)
    thresh = detector.preprocess()
    cv2.imwrite("./mask/" + img_path, thresh)
    # tacking trajectory and division
    matcher = po.matcher(img_path,cells)
    cells.contours = detector.segmentation(thresh)
    cells.cents = matcher.get_cent()
    cells_set.append(cells)
    matcher.draw_contours()
    result = matcher.get_trajectory(cells_set)
    cv2.imwrite("./tracking/" + img_path, result)


### task 3
# input a image number then show the image
# input a cell number then show the info of that image


