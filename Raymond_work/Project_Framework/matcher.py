import cv2
import numpy as np
from cell import Cell

DIST_THRESHOLD = 20

class Matcher:
    def __init__(self, detector):
        self.detector = detector
        self.existing_cells = {}
        self.id_counter = 1
    
    # Register a new cell
    def register(self, contour, cent, area):
        new_cell = Cell(self.id_counter, contour, cent, area)
        self.id_counter += 1
        self.existing_cells[self.id_counter] = new_cell
        
    # Delete a cell
    def delete(self, id):
        if id in self.existing_cells:
            del self.existing_cells[id]
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
            # Perform matching and update matched cells, or add new cell if min_dist < DIST_THRESHOLD
            for contour, cent, area in zip(contours, cents, areas):     
                distances = [(self.__distance__(self.existing_cells[key].get_centroid(), cent), key) for key in self.existing_cells]
                min_dist, key = sorted(distances, key = lambda x : x[0], reverse = False)[0]
                if min_dist < DIST_THRESHOLD:
                    self.existing_cells[key].update(contour, cent, area)
                else:
                    self.register(contour, cent, area)
        
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