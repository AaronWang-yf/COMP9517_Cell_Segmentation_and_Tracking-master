class Params:
    def __init__(self):
        #---------- General Parameters ----------#
        self.dataset_root = "./datasets/DIC-C2DH-HeLa" 
        self.dataset = "DIC-C2DH-HeLa"
        self.nn_method = "JNet" # choose JNet or DeepWater
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
        # Parameters below are trivial JNet parameters
        self.structure = None 
        self.aug_crop_params = [] 
        self.aug_elastic_params = [] 
        self.aug_intensity_params = [] 
        self.aug_rotation = True 
        self.aug_rotation_flip = True 
        self.batch_size = 1 
        self.batchnorm_momentum = 0.1 
        self.learning_rate = 0.0001 
        self.dataset_len_multiplier = 1 
        self.non_decreasing_output_file = "" 
        self.num_epochs = 3000 
        self.num_workers = 0 
        self.save_model_frequency = 200 
        self.validation_percentage = 0.0
        #---------- DeepWater Parameters ----------#



