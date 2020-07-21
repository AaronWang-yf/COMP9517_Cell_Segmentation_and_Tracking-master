import os
class Params():
    def __init__(self):
        #---------- General Parameters ----------#
        self.dataset_root = "./DIC-C2DH-HeLa" 
        self.dataset = "DIC-C2DH-HeLa"
        #---------- JNet Parameters ----------#
        self.cuda = True 
        self.images_idx = {"Sequence 1":[]}
        self.output_dir = "./results/segmentations/"
        self.mode = 'vis'
        self.resolution_levels = [-2,-1,0]#List of resolutions in the pipeline. 0 means the original resolution, -1 downscale by factor 2, -2 downscale by factor 4 etc
        self.dt_bound = 6 
        self.model_file = "./jnet_inference/DIC_model"
        self.load_dataset_to_ram = 0 
        self.num_workers = 0 

