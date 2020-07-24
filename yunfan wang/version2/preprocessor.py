# COMP 9517 project
# python 3.7
# cv2 3.4
import cv2
import numpy as np
# This file includes different methods to preprocess images for different dataset
# Preprocessor will get image masks as output

class Preprocessor:
    def __init__(self, seq):
        # Save original images
        self.original_images = seq
        # Preprocess images
        self.masks = self.preprocess(seq)
        # Counter is used to keep track of current image and mask on next() call
        self.counter = 0
        # Keep track if finished; 1 = not done
        self.status = 1

    # Process an array of original images and return an array of masks
    def preprocess(self, seq):
        processed = []
        for img in seq:
            kernel = 11
            img = cv2.GaussianBlur(img, (kernel, kernel), 0)
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            th = np.where(hist == np.max(hist))[0]
            ret, thresh = cv2.threshold(img, th[0] + 1, 255, cv2.THRESH_BINARY)
            thresh = cv2.erode(thresh, np.ones((9, 9)))
            thresh = cv2.dilate(thresh, np.ones((9, 9)))
            processed.append(thresh)
        return processed

    def get_masks(self):
        return self.masks

    # Serves the next image and its mask
    def next(self):
        # Indicate when done
        if self.counter >= len(self.original_images) - 1:
            self.status = 0
        
        image, mask = self.original_images[self.counter], self.masks[self.counter]
        self.counter += 1
        return image, mask


