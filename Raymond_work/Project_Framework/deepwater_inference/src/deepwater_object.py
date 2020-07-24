import os
import numpy as np
import cv2
import subprocess
from tqdm import tqdm
from skimage.morphology import watershed
from datetime import datetime

from .dataset import Dataset
from .models import UNetModel
from .postprocessing import \
    postprocess_markers as pm,\
    postprocess_markers_09,\
    postprocess_foreground
from .utils import \
    get_formatted_shape,\
    overlay_labels,\
    clean_dir,\
    create_tracking,\
    remove_edge_cells, \
    find_sequences


class DeepWater:
    def __init__(self, config):
        self.config = config
        self.mode = config.mode
        self.checkpoint_path = config.checkpoint_path
        self.name = config.DATASET_NAME
        self.model_name = config.MODEL_NAME
        self.seq = config.SEQUENCE
        self.batch_size = config.BATCH_SIZE
        self.version = config.VERSION
        self.tracking = config.TRACKING
        self.display = config.DISPLAY_RESULTS
        self.debug = config.DEBUG
        self.dim = config.DIM
        self.config_path = config.CONFIGURATION_PATH
        self.new_model = config.NEW_MODEL
        
        config.MODEL_MARKER_PATH = "model_markers"
        # print("\nIn deepwater_object now, the config.MODEL_MARKER_PATH is:",config.MODEL_MARKER_PATH,"\n")
        self.m_model_path = self._get_model_path(config.MODEL_MARKER_PATH)
        # print("\nIn deepwater_object now, the self.m_model_path is:",self.m_model_path,"\n")
        # print("\nIn deepwater_object now, the config.MODEL_FOREGROUND_PATH is:",config.MODEL_FOREGROUND_PATH,"\n")
        self.f_model_path = self._get_model_path(config.MODEL_FOREGROUND_PATH)

        self.marker_model = None
        self.foreground_model = None
        self.marker_dataset = None
        self.foreground_dataset = None

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.regenerate_prediction = config.REGENERATE_PREDICTION
        self.digits = self._get_n_digits()
        self.border = config.FRAME_BORDER
        self.dataset = None

        self.train_markers = config.TRAIN_MARKERS
        self.train_foreground = config.TRAIN_FOREGROUND


        print('DeepWater model was created!')

    # def load(self):
    #     self.marker_model = MarkerModel(self.config)
    #     self.foreground_model = MarkerModel(self.config)
    #     print('models were loaded')

    def test(self):

        img_path = self.config.TEST_PATH
        viz_path = self.config.VIZ_PATH
        out_path = self.config.OUT_PATH

        processed_masks = []

        assert os.path.isdir(img_path), img_path

        # create result path
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        clean_dir(out_path)

        # create display path
        if self.display:
            if not os.path.isdir(viz_path):
                os.mkdir(viz_path)
            clean_dir(viz_path)

        self._set_img_shape()

        # create dataset instance
        self.dataset = Dataset(self.config)
        marker_model = UNetModel(self.config)
        foreground_model = UNetModel(self.config)

        # print("\nIn deepwater_object now, the m_model_path is: ",self.m_model_path,"\n")

        marker_model.load(self.m_model_path)
        foreground_model.load(self.f_model_path)


        batch_size = self.batch_size
        n_batches = int(np.ceil(len(self.dataset) / batch_size))

        

        if self.debug > 2:
            self.dataset.save_all()

        # CTC BF datasets compatibility
        foreground_index = 1
        if self.version == 1.0:
            foreground_index = 3
        # CTC SIM+ datasets compatibility
        if self.version == 0.9:
            postprocess_markers = postprocess_markers_09
        else:
            postprocess_markers = pm


        for batch_index in tqdm(range(n_batches)):
            indexes = list(range(batch_index*batch_size, (batch_index + 1) * batch_size))

            marker_prediction = marker_model.predict_dataset(self.dataset,
                                                             batch_index=batch_index)[..., foreground_index]
            foreground_prediction = foreground_model.predict_dataset(self.dataset,
                                                                     batch_index=batch_index)[..., -1]
            n_samples = len(foreground_prediction)

            for i in range(n_samples):
                marker_image = marker_prediction[i, ...]
                foreground_image = foreground_prediction[i, ...]

                # convert markers and foreground to uint8 images
                marker_image = (marker_image * 255).astype(np.uint8)
                foreground_image = (foreground_image * 255).astype(np.uint8)

                _, marker_function = postprocess_markers(marker_image,
                                                         threshold=self.config.THR_MARKERS,
                                                         c=self.config.MARKER_DIAMETER,
                                                         h=self.config.MIN_MARKER_DYNAMICS,
                                                         dic=("DIC-C2DH-HeLa" in self.name))

                foreground = postprocess_foreground(foreground_image,
                                                    threshold=self.config.THR_FOREGROUND)
                foreground = np.maximum(foreground, (marker_function > 0) * 255)

                if self.version == 1.0:
                    # imposing markers into segmenation function
                    segmentation_function = np.maximum((255 - foreground_image), (255 - foreground))
                else:
                    segmentation_function = -foreground_image

                labels = watershed(segmentation_function, marker_function, mask=foreground)
                labels = remove_edge_cells(labels, self.border)

                index = str(indexes[i]).zfill(self.digits)

                # store result
                # cv2.imwrite(f'{out_path}/mask{index}.tif', labels.astype(np.uint16))

                processed_masks.append((indexes[i],labels))
        processed_masks.sort(key= lambda x:x[0],reverse=False) 
        processed_masks = [x[1] for x in processed_masks]
        return(processed_masks)

    def _store_visualisations(self,
                              viz_path,
                              out_path):
        print("storing visualisations...\n")
        for i in tqdm(range(len(self.dataset.flist_img))):
            o = self.dataset.get_image(i)
            index = str(i).zfill(self.digits)
            labels = cv2.imread(f'{out_path}/mask{index}.tif', cv2.IMREAD_ANYDEPTH)
            labels = labels.astype(np.uint8)
            overlay = overlay_labels(o, labels)
            cv2.imwrite(f'{viz_path}/color_segmentation{index}.tif',
                        overlay.astype(np.uint8))

    def train(self):

        img_path = self.config.IMG_PATH
        assert os.path.isdir(img_path), img_path

        self._set_img_shape()

        """ create model """
        model_m = UNetModel(self.config)
        model_f = UNetModel(self.config)

        """ load weights """
        if os.path.isfile(self.m_model_path):
            print('loading marker model weights')
            model_m.load(self.m_model_path)
        if os.path.isfile(self.f_model_path):
            print('loading foreground model weights')
            model_f.load(self.f_model_path)

        """ initialize dataset """
        dataset_markers = Dataset(self.config, markers=True)
        dataset_foreground = Dataset(self.config)

        if self.config.TESTMODE:
            self._store_network_inputs(img_path, dataset_markers, tag='markers')
            self._store_network_inputs(img_path, dataset_foreground, tag='foreground')
            exit()

        """ train and store"""
        if self.train_markers:
            print("training marker model")
            print(f"dataset length: {len(dataset_markers)}")
            model_m.train_model(dataset_markers)
            model_m.save_weights(self.m_model_path)
        if self.train_foreground:
            print("training foreground model")
            print(f"dataset length: {len(dataset_foreground)}")
            model_f.train_model(dataset_foreground)
            model_f.save_weights(self.f_model_path)

    def eval(self):

        gt_path = os.path.join(self.config.DATA_PATH, self.name, f'{self.seq}_GT/SEG')
        assert os.path.isdir(gt_path), f'GT path do not exists {gt_path}'

        if self.regenerate_prediction or not self._verify_results():
            print('Regenerating results...')
            self.test()
            assert self._verify_results(), 'Results are not consistent.'

        dataset_path = os.path.join(self.config.DATA_PATH, self.name)
        cmd = ("./measures/SEGMeasure", dataset_path, self.seq, str(self.digits))
        seg_measure = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
        seg = seg_measure.decode('utf8')

        cmd = ("./measures/DETMeasure", dataset_path, self.seq, str(self.digits))
        det_measure = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()[0].strip()
        det = det_measure.decode('utf8')

        print(f'Score of {dataset_path}, seq {self.seq}:')
        print(f'\t{seg}\n\t{det}')

    def _set_img_shape(self):
        if self.config.MODE == 1:

            seq = find_sequences(self.config.IMG_PATH)
            assert len(seq) > 0, f'there are no image sequences in {self.config.IMG_PATH}'

            for s in seq:
                img_path = os.path.join(self.config.IMG_PATH, s)
                dim, dim_original = get_formatted_shape(img_path)
        else:
            img_path = self.config.TEST_PATH
            dim, dim_original = get_formatted_shape(img_path)
        self.config.DIM, self.config.DIM_ORIGINAL = dim, dim_original

    def _get_n_digits(self):
        path = self.config.TEST_PATH
        img_name = [name for name in os.listdir(path) if '.txt' not in name][0]
        digits = len(img_name.split('.')[0]) - 1
        return digits

    def _verify_results(self):
        # gt_path = os.path.join(self.config.DATA_PATH, self.name, f'{self.seq}_GT/SEG')
        res_path = os.path.join(self.config.DATA_PATH, self.name, f'{self.seq}_RES')

        if not os.path.isdir(res_path):
            return False

        # TODO: test every gt file if it has proper mask
        return True

    def _get_model_path(self, model_name):
        self.config_path = "./deepwater_inference/checkpoints" 
        path = os.path.join(self.config_path, self.model_name)
        files = [f for f in os.listdir(path) if model_name in f]
        files.sort()
        if len(files) == 0 or self.new_model:
            date = datetime.now().strftime("%y%m%d_%H%M")
            if self.mode == 1:
                print(f"created new model: {os.path.join(path, f'{model_name}_{date}')}.h5")
            return os.path.join(path, f'{model_name}_{date}.h5')
        if self.mode == 2:
            print(f'loaded model {files[-1]}')
        return os.path.join(path, files[-1])

    def _store_network_inputs(self, img_path, dataset, tag='marker'): 
        debug_path = os.path.join(img_path, 'DEBUG')
        if not os.path.isdir(debug_path):
            os.mkdir(debug_path)
        print('test mode')
        print(f'Storing markers dataset to {debug_path}')
        images, gts = dataset.get_all()

        cwd = os.getcwd()

        for s in range(images.shape[0]):
            for d in range(images.shape[-1]):
                img = (images[s, ..., d] + .5) * 255
                cv2.imwrite(os.path.join(cwd, debug_path, f'{tag}_{s:03}_img_{d:03}.png'), img)
        for s in range(gts.shape[0]):
            for d in range(gts.shape[-1]):
                gt = gts[s, ..., d] * 255
                cv2.imwrite(os.path.join(cwd, debug_path, f'{tag}_{s:03}_gt_{d:03}.png'), gt)

    def _get_flists(self, dataset):
        return dataset._get_flists()
