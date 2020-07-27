# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import copy

# Detector asks for a preprocessed image from the Preprocessor;
# Once received, it finds the contours of the image and returns the set of contours
class Detector:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def next(self):
        # Get the next image from Preprocessor
        image, mask = self.preprocessor.next()
        contours = []
        for i in set(mask.reshape(1,-1).tolist()[0]):
        # Find contours
            if i!=0:
                filt = copy.deepcopy(mask)
                filt[filt != i] = 0
                _, contour, _ = cv2.findContours(filt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for a in contour:
                    if cv2.contourArea(a) > 4:
                        contours.append(a)
        return image, contours
