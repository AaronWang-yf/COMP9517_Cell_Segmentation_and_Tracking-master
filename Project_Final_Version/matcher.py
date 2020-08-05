# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import numpy as np

from cell import Cell

# mathcer responses for calculating, updating and analysing all information for cells
# it will return all cell information in a image, e.g. contour, cent, area path etc.

DIST_THRESHOLD = 50
### detect split
MIN_SPLIT_RATIO = 0.2
MAX_SPLIT_RATIO = 0.8
MIN_SIZE_THRESHOLD = 0.5
MAX_SIZE_THRESHOLD = 1.5
MAX_DIS_RATIO = 2


class Matcher:

    def __init__(self, detector):
        self.detector = detector
        self.existing_cells = {}
        self.id_counter = 0
        self.flag = 0

    # Register a new cell
    # add intensity as a new param
    def register(self, contour, cent, area, intensity):
        self.id_counter += 1
        new_cell = Cell(self.id_counter, contour, cent, area, intensity)
        self.existing_cells[self.id_counter] = new_cell

    # Delete a cell
    def delete(self, id):
        if id in self.existing_cells:
            del self.existing_cells[id]
        else:
            print('Cell ID not found for deletion')

     # Gets a set of new contours, generates centroids and matches new cells to existing cells

    def next(self, cells_history):
        # Get next set of contours from Detector
        image, contours = self.detector.next()

        # Calculate centroids and areas
        cents, areas = self.__get_cents_areas__(contours)
        intensities = self.__get_intensity__(image, contours)

        # If there are no existing cells, add all new cells       
        if not cells_history:
            for contour, cent, area, intensity in zip(contours, cents, areas, intensities):
                self.register(contour, cent, area, intensity)
            return image, self.existing_cells
        else:
            pre_cells = cells_history[-1]
            self.existing_cells = {}
            self.id_counter = 0
            for contour, cent, area, intensity in zip(contours, cents, areas, intensities):
                self.register(contour, cent, area, intensity)

            ## Perform matching and update matched cells, or add new cell if min_dist < DIST_THRESHOLD
            # old is key for pre_cells
            # key is the id of an existing cell
            # sim represents similarity, which equals to area * intensity. This is used for measuring the
            # similarity between cells
            self.flag += 1
            for old in pre_cells:
                old_cent = pre_cells[old].get_centroid()
                old_contour = pre_cells[old].get_contour()
                old_area = pre_cells[old].get_area()
                old_intensity = pre_cells[old].intensity
                old_sim = old_area * old_intensity
                ### compare each old cell with two new cells and update new cells based on the result
                distances = [(self.__distance__(self.existing_cells[key].get_centroid(), old_cent), key) for key in
                             self.existing_cells]
                min_dist, min_key = sorted(distances, key=lambda x: x[0], reverse=False)[0]
                sec_dist, sec_key = sorted(distances, key=lambda x: x[0], reverse=False)[1]

                min_area = self.existing_cells[min_key].get_area()
                min_intensity = self.existing_cells[min_key].intensity
                min_cent = self.existing_cells[min_key].get_centroid()
                min_contour = self.existing_cells[min_key].get_contour()
                min_sim = min_area * min_intensity

                sec_area = self.existing_cells[sec_key].get_area()
                sec_intensity = self.existing_cells[sec_key].intensity
                sec_sim = sec_area * sec_intensity

                # If an existing cell meets the mitosis criteria
                if MIN_SPLIT_RATIO <= min_area / old_area <= MAX_SPLIT_RATIO and \
                        MIN_SPLIT_RATIO <= sec_area / old_area <= MAX_SPLIT_RATIO and \
                        sec_dist < DIST_THRESHOLD:
                    pre_cells[old].split_p = True
                    self.existing_cells[min_key].split_c = True
                    self.existing_cells[sec_key].split_c = True

                if min_dist < DIST_THRESHOLD and MIN_SIZE_THRESHOLD <= min_area / old_area <= MAX_SIZE_THRESHOLD and \
                        pre_cells[old].split_p == False:
                    self.existing_cells[min_key] = pre_cells[old]
                    self.existing_cells[min_key].id = min_key
                    self.existing_cells[min_key].split_c = False
                    self.existing_cells[min_key].update(min_contour, min_cent, min_area, min_intensity)
        return image, self.existing_cells

    def __distance__(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

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

    def __get_intensity__(self, img, contours):
        intensities = []
        for contour in contours:
            i = 0
            length = len(contour)
            area = 0
            illumination = 0
            while i < length:
                if i == length - 1:
                    break
                else:
                    x1 = contour[i][0][0]
                    x2 = contour[i + 1][0][0]
                    area = area + 1
                    if x1 != x2:
                        i += 1
                    else:
                        y1 = contour[i][0][1]
                        y2 = contour[i + 1][0][1]
                        for y in range(y1, y2):
                            illumination += img[y, x1]
                        i += 2
            intensities.append(illumination / area)
        return intensities
