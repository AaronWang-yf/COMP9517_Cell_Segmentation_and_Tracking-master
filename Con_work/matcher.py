import cv2
import numpy as np
import copy
from cell import Cell

DIST_THRESHOLD = 20
### detect split
MIN_SPLIT_RATIO = 0.3
MAX_SPLIT_RATIO = 0.7
MIN_SIZE_THRESHOLD = 0.5
MAX_SIZE_THRESHOLD = 1.5
MAX_DIS_RATIO = 2

class Matcher:
    def __init__(self, detector):
        self.detector = detector
        self.existing_cells = {}
        self.id_counter = 1

    # Register a new cell
    ### add splitting certification
    def register(self, contour, cent, area, split=False):
        new_cell = Cell(self.id_counter, contour, cent, area, split=split)
        self.id_counter += 1
        self.existing_cells[self.id_counter] = new_cell

    # Delete a cell
    def delete(self, id):
        if id in self.existing_cells:
            self.existing_cells.pop(id)
        else:
            print('Cell ID not found for deletion')

            # Gets a set of new contours, generates centroids and matches new cells to existing cells

    def next(self):
        # Get next set of contours from Detector
        image, contours = self.detector.next()

        # Calculate centroids and areas
        cents, areas = self.__get_cents_areas__(contours)

        # If no existing cells, add all new cells       
        if self.existing_cells == {}:
            for contour, cent, area in zip(contours, cents, areas):
                self.register(contour, cent, area)
        else:
            ## Perform matching and update matched cells, or add new cell if min_dist < DIST_THRESHOLD
            existing = copy.deepcopy(self.existing_cells)
            for contour, cent, area in zip(contours, cents, areas):
                if len(existing) == 0: # No more existing cells left to assign; add it as a new cell
                    self.register(contour, cent, area)
                    continue
                # Otherwise calculate distance to all existing cells that haven't been updated
                distances = [(self.__distance__(existing[key].get_centroid(), cent), key) for key in
                             existing]
                min_dist, key = sorted(distances, key=lambda x: x[0], reverse=False)[0]
                if min_dist < DIST_THRESHOLD:
                    self.existing_cells[key].update(contour, cent, area)
                    existing.pop(key)
                else:
                    self.register(contour, cent, area)
        
            # Delete all cells which were not updated
            for key in existing:
                self.existing_cells.pop(key)
                    
            # temp_cells = copy.deepcopy(self.existing_cells)
            # if cents:
            #     for key in temp_cells:
            #         existing_cent = temp_cells[key].get_centroid()
            #         existing_area = temp_cells[key].get_area()
            #         distances = [(self.__distance__(existing_cent, cent)) for cent in cents]
            #         distances = np.array(distances)
            #         min_dis = sorted(distances)[0]
            #         if len(distances) > 1:
            #             min_idx = int(np.where(distances == min_dis)[0])
            #         else:
            #             min_idx = 0
            #         if len(distances) > 1:
            #             sec_dis = sorted(distances)[1]
            #             sec_idx = int(np.where(distances == sec_dis)[0])
            #             if MIN_SPLIT_RATIO <= areas[min_idx] / existing_area <= MAX_SPLIT_RATIO and \
            #                     MIN_SPLIT_RATIO <= areas[sec_idx] / existing_area <= MAX_SPLIT_RATIO and \
            #                     sec_dis < DIST_THRESHOLD and sec_dis/min_dis <= MAX_DIS_RATIO:
            #                 # self.existing_cells[key].split = True
            #                 self.delete(self.existing_cells[key].get_id())
            #                 self.register(contours[min_idx], cents[min_idx], areas[min_idx], True)
            #                 self.register(contours[sec_idx], cents[sec_idx], areas[sec_idx], True)
            #                 contours.pop(min_idx), contours.pop(sec_idx)
            #                 cents.pop(min_idx), cents.pop(sec_idx)
            #                 areas.pop(min_idx), areas.pop(sec_idx)
            #         if min_dis < DIST_THRESHOLD and MIN_SIZE_THRESHOLD <= areas[
            #             min_idx] / existing_area <= MAX_SIZE_THRESHOLD:
            #             self.existing_cells[key].update(contours[min_idx], cents[min_idx], areas[min_idx])
            #             contours.pop(min_idx)
            #             cents.pop(min_idx)
            #             areas.pop(min_idx)

            #         else:
            #             self.delete(self.existing_cells[key].get_id())

            # if cents:
            #     for i in range(len(cents)):
            #         self.register(contours[i], cents[i], areas[i])
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
