import cv2
import matplotlib.pyplot as plt
import numpy as np


# input: image
# output: contour array  ( change to contour,cent list)
# the class contains preprocess method and segmentation method
class detector():
    def __init__(self, img_path):
        self.img_path = img_path
        self.cells = []

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

    def __get_cents_areas__(self, contours):
        cents = []
        areas = []
        for i, j in zip(contours, range(len(contours))):
            M = cv2.moments(i)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cents.append((cX, cY))
            areas.append(cv2.contourArea(i))
        return cents, areas

    # segment the threshed image
    #### instead of output the contours, it make a little bit change based on contour
    def segmentation(self, thresh):
        # add segmentation method e.g. watershed
        thresh, contours, hirearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cents, areas = self.__get_cents_areas__(contours)
        for a, b, c in zip(contours, cents, areas):
            cc = cell()
            cc.contour = a
            cc.cent = b
            cc.area = c
            self.cells.append(cc)
        return self.cells

    def show_image(self, img, cmap='gray'):
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.show()


class cell:
    def __init__(self):
        self.contour = []
        self.cent = ()
        self.area = 0


# the class is used to save info of cells in one image
# it can be considered as a structure
class cells:
    def __init__(self):
        self.cell_set = []
        # self.path = []
        # add more parameters needed


def __distance__(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# temporarily drawer wil be included in class mather
# cell will use class cell to store and update info
# input: img_path, structure cells
# output: update cells, results etc.
class matcher():
    def __init__(self, img_path, cells):
        self.img_path = img_path
        self.cells = cells
        self.origin = cv2.imread(img_path, 1)

    def draw_contours(self):
        contours = []
        for i in self.cells.cell_set:
            contours.append(i.contour)
        self.origin = cv2.drawContours(self.origin, contours, -1, (0, 255, 0), 1)
        i = 1
        for ele in self.cells.cell_set:
            cent = ele.cent
            # self.origin = cv2.circle(self.origin, cent, 1, (0, 255, 0), 2)  # this can be updated to draw numbers
            self.origin = cv2.putText(self.origin, str(i), cent, 1, 1, (0, 255, 0), 1)
            i  = i+1
        return self.origin

    # the function may need be updated with split detecting
    def if_same(self, cell_a, cell_b):
        pass

    # ele is a cell
    # first is a set of cells
    def get_distance(self, ele, first):
        distance_list = []
        point0 = ele.cent
        for a in first.cell_set:
            point1 = a.cent
            distance_list.append(__distance__(point0, point1))
        return distance_list

    def get_trajectory(self, cells_set_seq): # cells_set_seq: a sequence of cells in an individual image

        if len(cells_set_seq) > 1:
            index = len(cells_set_seq)
            while index > 1:
                second = cells_set_seq[index - 1]  # current image
                first = cells_set[index - 2]  # previous image
                index = index - 1
                # ele  = class cell
                for ele in second.cell_set:
                    dis_list = self.get_distance(ele, first)
                    dis_list = np.array(dis_list)
                    min_dis = sorted(dis_list)[0]
                    # print(min_dis)
                    min_idx = int(np.where(dis_list == min_dis)[0])
                    nearest = first.cell_set[min_idx]
                    # print(min)
                    if min_dis < 5:
                        self.origin = cv2.line(self.origin, ele.cent, nearest.cent, (0, 255, 0), 1, 4)

        return self.origin

    def update_cells(self):
        return self.cells
