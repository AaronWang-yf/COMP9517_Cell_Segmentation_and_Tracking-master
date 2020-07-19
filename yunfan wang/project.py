import cv2
import matplotlib.pyplot as plt
import numpy as np


# input: image
# output: contour array
# the class contains preprocess method and segmentation method
class detector():
    def __init__(self, img_path):
        self.img_path = img_path
        self.contours = None

    # process get the black-white image as output
    def preprocess(self):
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        # plt.imshow(img, cmap='gray')
        # plt.axis('off')
        # plt.show()

        # process
        kernel = 11
        img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        th = np.where(hist == np.max(hist))
        ret, thresh = cv2.threshold(img, th[0] + 1, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, np.ones((9, 9)))
        thresh = cv2.dilate(thresh, np.ones((9, 9)))
        # plt.imshow(thresh, cmap='gray')
        # plt.axis('off')
        # plt.show()

        return thresh

    # segment the threshed image
    def segmentation(self, thresh):
        # add segmentation method e.g. watershed
        thresh, self.contours, hirearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return self.contours

    def show_image(self, img, cmap='gray'):
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()

# the class is used to save info of cells in one image
# it can be considered as a structure
class cells:
    def __init__(self):
        self.contours = [] # contours of each cell in one image
        self.cents = []   # matched centroids
        # add more parameters needed


def __distance__(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# temporarily drawer wil be included in class mather
# cell will use class cell to store and update info
# input: img_path, structure cells
# output: update cells, results etc.
class matcher():
    def __init__(self,img_path,cells):
        self.img_path = img_path
        self.cells = cells
        self.origin = cv2.imread(img_path, 1)

    def get_cent(self):
        for i, j in zip(self.cells.contours, range(len(self.cells.contours))):
            M = cv2.moments(i)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.cells.cents.append((cX, cY))
        return self.cells.cents

    def draw_contours(self):
        self.origin = cv2.drawContours(self.origin, self.cells.contours, -1, (0, 255, 0), 1)
        for ele in self.cells.cents:
            self.origin = cv2.circle(self.origin, ele, 1, (0, 255, 0), 2) # this can be updated to draw numbers
        return self.origin

    # the function may need be updated with split detecting
    def get_trajectory(self,cells_set):
        img_path = self.img_path
        cents_set  = []
        for ele in cells_set:
            cents_set.append(ele.cents)
        if len(cents_set) > 1:
            index = len(cents_set)
            while index > 1:
                second = cents_set[index - 1]
                first = cents_set[index - 2]
                index = index - 1
                for p1 in second:
                    nearest = None
                    dis = 100
                    for p2 in first:
                        if __distance__(p1, p2) < dis:
                            nearest = p2
                            dis = __distance__(p1, p2)
                    if dis < 5:
                        self.origin = cv2.line(self.origin, p1, nearest, (0, 255, 0), 1, 4)
            # plt.imshow(draw1)
            # plt.axis('off')
            # plt.show()
        # cv2.imwrite("./tracking/" + img_path, self.origin)
        return self.origin

