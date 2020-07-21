import cv2
import numpy as np
import torch 
import sys 
import ast
sys.path.insert(0, './jnet_inference')
from jnet_inference.nets import Model,load_model_from_file
from jnet_inference.evaluate import evaluate

class Preprocessor:
    def __init__(self, images,params):
        # Save original images
        # self.original_images = seq 
        self.params = params 
        self.params.cuda = torch.cuda.is_available() and params.cuda 
        self.model = None 
        self.dataset = None 
        if (params.dataset=="DIC-C2DH-HeLa"):
            self.original_images = images 
            self.model = load_model_from_file(params.model_file)
        else:
            self.original_images = []
            for img_file in images:
                image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) 
                self.original_images.append(image)
        # Preprocess images 
        self.masks = self.preprocess(self.original_images)
        # Counter is used to keep track of current image and mask on next() call
        self.counter = 0
        
    # Process an array of original images and return an array of masks
    def preprocess(self, seq):
        processed = []
        
        return processed

    def get_DIC_masks(self):
        dataset=JNet_Cells(self.original_images,[],self.params.dt_bound,ast.literal_eval(self.params.resolution_levels),load_to_memory=bool(self.params.load_dataset_to_ram))
        return (evaluate(self.model,dataset,self.params))

    
    def get_masks(self):
        return self.masks
            
    def next(self):
        # Serves the next image and its mask
        image, mask = self.original_images[self.counter], self.masks[self.counter]
        self.counter += 1
        return image, mask
       