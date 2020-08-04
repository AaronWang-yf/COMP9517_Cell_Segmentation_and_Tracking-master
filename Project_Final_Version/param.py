class Params:
    def __init__(self):
        #---------- General Parameters ----------#
        """
        For dataset_root, if you are using neural network method, you should set it to
        be "./datasets/DATASET_NAME". If you are using thresholding algorithm, you should set
        it to be "./datasets/DATASET_NAME/SEQ". SEQ should be 01,02..., etc.
        """
        self.dataset_root = "./datasets/DIC-C2DH-HeLa" 
        self.dataset = "DIC-C2DH-HeLa"
        self.images_idx = {"01":[]}
        self.nn_method = "DeepWater" # choose JNet or DeepWater or None
        #---------- JNet Parameters ----------#
        self.cuda = True 
        self.output_dir = "./results/segmentations"
        self.mode = 'vis'
        self.resolution_levels ='[-2,-1,0]'#List of resolutions in the pipeline. 0 means the original resolution, -1 downscale by factor 2, -2 downscale by factor 4 etc
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



