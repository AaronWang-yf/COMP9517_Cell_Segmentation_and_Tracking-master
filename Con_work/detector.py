import cv2

# Detector asks for a preprocessed image from the Preprocessor;
# Once received, it finds the contours of the image and returns the set of contours
class Detector:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
    
    def next(self):
        # Get the next image from Preprocessor
        image, mask = self.preprocessor.next()
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return image, contours
            
            