import cv2
import numpy as np
from scipy import ndimage as ndi 
from libtiff import TIFF
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import torch  # The pytorch version should be 0.3.1
import sys 
import ast
sys.path.insert(0, './jnet_inference')
sys.path.insert(0, './deepwater_inference')
from jnet_inference.nets import Model,load_model_from_file
from jnet_inference.evaluate import evaluate
from jnet_inference.dataset import JNet_Cells
from dw_config import load_config
from deepwater_inference.src.deepwater_object import DeepWater

class Preprocessor:
    def __init__(self, images,params):
        # Save original images
        # self.original_images = images
        self.params = params 
        self.params.cuda = torch.cuda.is_available() and params.cuda 
        self.model = None 
        self.dataset = None 
         
        if (params.dataset=="DIC-C2DH-HeLa" and params.nn_method=="JNet"):
            self.model = load_model_from_file(params.model_file)
        
        # Preprocess images 
        self.masks = self.preprocess(images)
        # Counter is used to keep track of current image and mask on next() call
        self.counter = 0
        
    # Process an array of original images and return an array of masks
    def preprocess(self,images):
        processed = []
        if (self.params.dataset=="DIC-C2DH-HeLa"):
            processed = get_DIC_masks(self,images)
        elif(self.params.dataset=="Fluo-N2DL-HeLa"):
            processed = get_Fluo_masks(self,images)
        elif(self.params.dataset=="PhC-C2DL-PSC"):
            processed = get_Phc_masks(self,images)
        else:
            raise ValueError("Dataset "+self.params.dataset+" is not supported!")
        return processed

    def get_DIC_masks(self,images):
        if (self.params.nn_method=="JNet"):
            dataset=JNet_Cells(self.original_images,[],self.params.dt_bound,ast.literal_eval(self.params.resolution_levels),load_to_memory=bool(self.params.load_dataset_to_ram))
            return (evaluate(self.model,dataset,self.params))
        elif(self.params.nn_method=="DeepWater"):
            config = load_config(self.params,mode=2)
            model = DeepWater(config)
            return (model.test())
        else:
            raise ValueError("Neural Network Method should be JNet or DeepWater!")

    def get_Fluo_masks(self,images):
        def watershed_batch_process_2(img_path): 
            tif = TIFF.open(img_path,mode='r')
            image = tif.read_image()
            sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)# gradient in x direction
            sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)# gradient in y direction
            sobelX = np.uint8(np.absolute(sobelX))
            sobelY = np.uint8(np.absolute(sobelY))
            image = cv2.bitwise_or(sobelX,sobelY)
            ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            image = cv2.medianBlur(image, 9)
            image = cv2.dilate(image, np.ones((3,3), np.uint8))
            image = cv2.erode(image, np.ones((19,19), np.uint8))
            image = cv2.dilate(image, np.ones((13,13), np.uint8))
            distance = ndi.distance_transform_edt(image)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((9, 9)),labels=image)
            markers = ndi.label(local_maxi)[0]
            labels = watershed(-distance, markers, mask=image)
            labels = cv2.normalize(labels,None, 0, 255, cv2.NORM_MINMAX)
            labels[np.where(labels>10)] = 255
            return (labels)
        fluo_masks = [] 
        for img_path in self.original_images:
            fluo_masks.append(watershed_batch_process_2(img_path))
        return(fluo_masks)


    def get_PhC_mask(self,images):
        def d3_threshold_seg(img_path):
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) 
            img = cv2.erode(img,np.ones((2,2)))
            kernel = 3
            img  =cv2.GaussianBlur(img,(kernel,kernel),0)
            hist,bins = np.histogram(img.flatten(),256,[0,256])
            th  = np.where(hist ==np.max(hist))
            th = np.array([th[0].max()]) 
            ret, thresh = cv2.threshold(img,th+10,255,cv2.THRESH_BINARY)
            return (thresh)
        if (self.params.nn_method == "DeepWater"):
            config = load_config(self.params,mode=2)
            model = DeepWater(config)
            return (model.test()) 
        else:
            phc_masks = []
            for img_path in images:
                phc_masks.append(d3_threshold_seg(img_path))
            return(phc_masks)

    
    # Mask Getter
    def get_masks(self):
        return self.masks
            
    def next(self):
        # Serves the next image and its mask
        image, mask = self.original_images[self.counter], self.masks[self.counter]
        self.counter += 1
        return image, mask
       