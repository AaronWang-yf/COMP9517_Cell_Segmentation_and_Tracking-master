import numpy as np
import cv2
import os
from .postprocessing import postprocess_markers, postprocess_cell_mask
from skimage.morphology import watershed
from .preprocessing import load_images, shrink_markers, shrink_markers_inv

import tensorflow as tf

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model, load_model

from keras.optimizers import Adam


def topdist_merge(marker_fc, cell_mask):
    
    mi, ni = marker_fc.shape[:2]
    
    free = (cell_mask > 0) ^ (marker_fc > 0)

    labels = np.unique(marker_fc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    
    result = marker_fc
    
    progress = True
    
    while progress:
        loss = np.sum(free)
        
        new_result = np.zeros((mi,ni))
        
        for index in labels:
            mask = result == index
            new_mask = cv2.dilate(mask.astype(np.uint8), kernel)
            new_mask = new_mask * (free == 1)
            new_result += (mask + new_mask) * index
            free = free & (new_result == 0)
            
        progress = loss != np.sum(free)   

        result = new_result
    return result.astype(np.uint8)


def topdist_merge(marker_fc, cell_mask):
    
    dt = cv2.distanceTransform((marker_fc == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dtws = watershed(dt, marker_fc, mask=cell_mask)
    
    return dtws


def eucldist_merge(marker_fc, cell_mask):
    
    mi, ni = marker_fc.shape[:2]
    
    free = (marker_fc == 0)

    labels = np.unique(marker_fc)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    result = marker_fc
    progress = True
    
    while progress:
        loss = np.sum(free)
        
        new_result = np.zeros((mi,ni))
        
        for index in labels:
            mask = result == index
            new_mask = cv2.dilate(mask.astype(np.uint8), kernel)
            new_mask = new_mask * (free == 1)
            new_result += (mask + new_mask) * index
            free = free & (new_result == 0)
            
        progress = loss != np.sum(free)   

        result = new_result
    return (result * cell_mask // 255).astype(np.uint8)


def threshold_and_store_markers(predictions,
                                original=None,
                                out_path='.',
                                thr_markers=240,
                                thr_cell_mask=230,
                                viz=False,
                                circular=False,
                                erosion_size=12,
                                step=4,
                                merge_method='watershed',
                                m_size=100):

    viz_path = out_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        if predictions.shape[3] > 2:
            m = (predictions[i, :, :, 1] * 255).astype(np.uint8)
            c = (predictions[i, :, :, 3] * 255).astype(np.uint8)
        else:
            m = predictions[i, :, :, 0]
            c = predictions[i, :, :, 1]

        # postprocess the result of prediction
        idx, marker_function = postprocess_markers(m,
                                                   threshold=thr_markers,
                                                   c=erosion_size,
                                                   circular=circular,
                                                   h=step)
        seg = np.zeros(m.shape)
        markers = watershed(seg, marker_function)
        # markers = shrink_markers_inv(marker_function, m_size)

        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(out_path, i), markers.astype(np.uint16))
        if viz:
            cv2.imwrite('{}/c{:03d}.tif'.format(viz_path, i), c.astype(np.uint8))
            cv2.imwrite('{}/m{:03d}.tif'.format(viz_path, i), m.astype(np.uint8))

            cv2.imwrite('{}/marker_fc{:03d}.tif'.format(viz_path, i), marker_function)


def threshold_and_store_markers_threshold(predictions,
                                original=None,
                                out_path='.',
                                thr_markers=240,
                                thr_cell_mask=230,
                                viz=False,
                                circular=False,
                                erosion_size=12,
                                step=4,
                                merge_method='watershed',
                                m_size=100,
                                otsu=False):

    viz_path = out_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        if predictions.shape[3] > 2:
            m = (predictions[i, :, :, 1] * 255).astype(np.uint8)
            c = (predictions[i, :, :, 3] * 255).astype(np.uint8)
        else:
            m = predictions[i, :, :, 0]
            c = predictions[i, :, :, 1]

        # postprocess the result of prediction
        marker_function = ((m > thr_markers) * 255).astype(np.uint8)
        if otsu:
            _, marker_function = ret2,th2 = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        idx, marker_function = cv2.connectedComponents(marker_function)
        seg = np.zeros(m.shape)
        markers = watershed(seg, marker_function)
        # markers = shrink_markers_inv(marker_function, m_size)

        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(out_path, i), markers.astype(np.uint16))
        if viz:
            cv2.imwrite('{}/c{:03d}.tif'.format(viz_path, i), c.astype(np.uint8))
            cv2.imwrite('{}/m{:03d}.tif'.format(viz_path, i), m.astype(np.uint8))

            cv2.imwrite('{}/marker_fc{:03d}.tif'.format(viz_path, i), marker_function)


def threshold_and_store_gtmarkers(predictions,
                                  original=None,
                                  out_path='.',
                                  thr_markers=240,
                                  thr_cell_mask=230,
                                  viz=False,
                                  circular=False,
                                  erosion_size=12,
                                  step=4,
                                  merge_method='watershed',
                                  m_size=100):

    viz_path = out_path.replace('_RES', '_VIZ')
    gt_path = out_path.replace('RES', 'man_seg')

    for i in range(predictions.shape[0]):

        gt_name = os.path.join(gt_path, 'man_seg{:03}.tif'.format(i))
        assert os.path.isfile(gt_name), gt_name

        gt = cv2.imread(gt_name, cv2.IMREAD_ANYDEPTH)
        marker_function = shrink_markers(gt, m_size)
        _, marker_function = cv2.connectedComponents(marker_function.astype(np.uint8))

        if predictions.shape[3] > 2:
            c = predictions[i, :, :, 3]
        else:
            c = predictions[i, :, :, 1]

        # print(np.unique(c), thr_cell_mask)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, marker_function)

        # segmentation function
        # clipping
        segmentation_function = np.maximum((255 - c), (255 - cell_mask))

        assert merge_method in ['watershed', 'topdist', 'eucldist'], merge_method

        if merge_method == 'watershed':
            labels = watershed(segmentation_function, marker_function, mask=cell_mask)
        elif merge_method == 'topdist':
            labels = topdist_merge(marker_function, cell_mask)
        elif merge_method == 'eucldist':
            labels = eucldist_merge(marker_function, cell_mask)
        else:
            labels = None

        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(out_path, i), labels.astype(np.uint16))

        if viz:
            # vizualize result in rgb
            o = (original[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
            m_rgb = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)

            fg_mask = (labels != 0).astype(np.uint8)
            bg_mask = (labels == 0).astype(np.uint8)

            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] * fg_mask
            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] + bg_mask * 180
            labels_rgb[:, :, 1] = labels_rgb[:, :, 1] + bg_mask * 180
            labels_rgb[:, :, 2] = labels_rgb[:, :, 2] + bg_mask * 180

            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            mf_rgb = cv2.cvtColor(((marker_function > 0) * 255 ).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            sf_rgb = cv2.cvtColor(segmentation_function.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, mf_rgb, c_rgb, sf_rgb, overlay), 1)
            cv2.imwrite('{}/res{:03d}.tif'.format(viz_path, i), result)
            cv2.imwrite('{}/c{:03d}.tif'.format(viz_path, i), c.astype(np.uint8))
            cv2.imwrite('{}/m{:03d}.tif'.format(viz_path, i), gt.astype(np.uint8))

            cv2.imwrite('{}/marker_fc{:03d}.tif'.format(viz_path, i), marker_function)
            cv2.imwrite('{}/segmen_fc{:03d}.tif'.format(viz_path, i), segmentation_function)
            cv2.imwrite('{}/cell_mask{:03d}.tif'.format(viz_path, i), cell_mask)


def threshold_and_store(predictions,
                        original=None,
                        out_path='.',
                        thr_markers=240,
                        thr_cell_mask=230,
                        viz=False,
                        circular=False,
                        erosion_size=12,
                        step=4,
                        merge_method='watershed',
                        start_index=0,
                        n_digits=3):

    viz_path = out_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        if predictions.shape[3] > 2:
            m = (predictions[i, :, :, 1] * 255).astype(np.uint8)
            c = (predictions[i, :, :, 3] * 255).astype(np.uint8)
        else:
            m = predictions[i, :, :, 0]
            c = predictions[i, :, :, 1]

        # postprocess the result of prediction
        idx, marker_function = postprocess_markers(m,
                                                   threshold=thr_markers,
                                                   c=erosion_size,
                                                   circular=circular,
                                                   h=step)

        # print(np.unique(c), thr_cell_mask)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, marker_function)

        # segmentation function
        # clipping
        segmentation_function = np.maximum((255 - c), (255 - cell_mask))

        assert merge_method in ['watershed', 'topdist', 'eucldist'], merge_method

        if merge_method == 'watershed':
            labels = watershed(segmentation_function, marker_function, mask=cell_mask)
        elif merge_method == 'topdist':
            labels = topdist_merge(marker_function, cell_mask)
        elif merge_method == 'eucldist':
            labels = eucldist_merge(marker_function, cell_mask)
        else:
            labels = None


        # store result
        cv2.imwrite('{}/mask{:03d}.tif'.format(out_path, i), labels.astype(np.uint16))

        if viz:
            # vizualize result in rgb
            o = (original[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
            m_rgb = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)

            fg_mask = (labels != 0).astype(np.uint8)
            bg_mask = (labels == 0).astype(np.uint8)

            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] * fg_mask
            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] + bg_mask * 180
            labels_rgb[:, :, 1] = labels_rgb[:, :, 1] + bg_mask * 180
            labels_rgb[:, :, 2] = labels_rgb[:, :, 2] + bg_mask * 180

            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            mf_rgb = cv2.cvtColor(((marker_function > 0) * 255 ).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            sf_rgb = cv2.cvtColor(segmentation_function.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, mf_rgb, c_rgb, sf_rgb, overlay), 1)
            cv2.imwrite('{}/res{:03d}.tif'.format(viz_path, i + start_index), result)
            cv2.imwrite('{}/c{:03d}.tif'.format(viz_path, i + start_index), c.astype(np.uint8))
            cv2.imwrite('{}/m{:03d}.tif'.format(viz_path, i + start_index), m.astype(np.uint8))

            cv2.imwrite('{}/marker_fc{:03d}.tif'.format(viz_path, i + start_index), marker_function)
            cv2.imwrite('{}/segmen_fc{:03d}.tif'.format(viz_path, i + start_index), segmentation_function)
            cv2.imwrite('{}/cell_mask{:03d}.tif'.format(viz_path, i + start_index), cell_mask)


def threshold_and_store_bf(predictions,
                        original=None,
                        out_path='.',
                        thr_markers=240,
                        thr_cell_mask=230,
                        viz=False,
                        circular=False,
                        erosion_size=12,
                        step=4,
                        merge_method='watershed',
                        start_index=0,
                        n_digits=3):

    viz_path = out_path.replace('_RES', '_VIZ')

    for i in range(predictions.shape[0]):

        if predictions.shape[3] > 2:
            m = (predictions[i, :, :, 1] * 255).astype(np.uint8)
            c = (predictions[i, :, :, 3] * 255).astype(np.uint8)
        else:
            m = predictions[i, :, :, 0]
            c = predictions[i, :, :, 1]

        # postprocess the result of prediction
        idx, marker_function = postprocess_markers(m,
                                                   threshold=thr_markers,
                                                   c=erosion_size,
                                                   circular=circular,
                                                   h=step)

        # print(np.unique(c), thr_cell_mask)
        cell_mask = postprocess_cell_mask(c, threshold=thr_cell_mask)

        # correct border
        cell_mask = np.maximum(cell_mask, marker_function)

        # segmentation function
        # clipping
        segmentation_function = np.maximum((255 - c), (255 - cell_mask))

        assert merge_method in ['watershed', 'topdist', 'eucldist'], merge_method

        if merge_method == 'watershed':
            labels = watershed(segmentation_function, marker_function, mask=cell_mask)
        elif merge_method == 'topdist':
            labels = topdist_merge(marker_function, cell_mask)
        elif merge_method == 'eucldist':
            labels = eucldist_merge(marker_function, cell_mask)
        else:
            labels = None


        # store result
        cv2.imwrite('{}/mask{:04d}.tif'.format(out_path, i + start_index), labels.astype(np.uint16))

        if viz:
            # vizualize result in rgb
            o = (original[i, :, :, 0] + .5) * 255
            o_rgb = cv2.cvtColor(o, cv2.COLOR_GRAY2RGB)
            m_rgb = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            c_rgb = cv2.cvtColor(c.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            labels_rgb = cv2.applyColorMap(labels.astype(np.uint8) * 15, cv2.COLORMAP_JET)

            fg_mask = (labels != 0).astype(np.uint8)
            bg_mask = (labels == 0).astype(np.uint8)

            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] * fg_mask
            labels_rgb[:, :, 0] = labels_rgb[:, :, 0] + bg_mask * 180
            labels_rgb[:, :, 1] = labels_rgb[:, :, 1] + bg_mask * 180
            labels_rgb[:, :, 2] = labels_rgb[:, :, 2] + bg_mask * 180

            overlay = cv2.addWeighted(o_rgb.astype(np.uint8), 0.5, labels_rgb, 0.5, 0)

            mf_rgb = cv2.cvtColor(((marker_function > 0) * 255 ).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            sf_rgb = cv2.cvtColor(segmentation_function.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            result = np.concatenate((m_rgb, mf_rgb, c_rgb, sf_rgb, overlay), 1)
            cv2.imwrite('{}/res{:04d}.tif'.format(viz_path, i + start_index), result)
            cv2.imwrite('{}/c{:04d}.tif'.format(viz_path, i + start_index), c.astype(np.uint8))
            cv2.imwrite('{}/m{:04d}.tif'.format(viz_path, i + start_index), m.astype(np.uint8))

            cv2.imwrite('{}/marker_fc{:04d}.tif'.format(viz_path, i + start_index), marker_function)
            cv2.imwrite('{}/segmen_fc{:04d}.tif'.format(viz_path, i + start_index), segmentation_function)
            cv2.imwrite('{}/cell_mask{:04d}.tif'.format(viz_path, i + start_index), cell_mask)


def make_model(mi=512, ni=512, loss_function='mse'):

    assert str(mi).isdigit(), 'argument mi has to be integer'
    assert str(ni).isdigit(), 'argument ni has to be integer'
    mi = int(mi)
    ni = int(ni)

    os.chdir('/home/xlux/cbia/')

    # adapt this if using `channels_first` image data format
    input_img = Input(shape=(mi, ni, 1))

    # layers definition
    c1e = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1e)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2e = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2e)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3e = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3e)
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4e = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4e)
    p4 = MaxPooling2D((2, 2), padding='same')(c4)

    c5e = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5e)

    u4 = UpSampling2D((2, 2), interpolation='bilinear')(c5)
    a4 = Concatenate(axis=3)([u4, c4])
    c6e = Conv2D(256, (3, 3), activation='relu', padding='same')(a4)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6e)

    u3 = UpSampling2D((2, 2), interpolation='bilinear')(c6)
    a3 = Concatenate(axis=3)([u3, c3])
    c7e = Conv2D(128, (3, 3), activation='relu', padding='same')(a3)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7e)

    u2 = UpSampling2D((2, 2), interpolation='bilinear')(c7)
    a2 = Concatenate(axis=3)([u2, c2])
    c8e = Conv2D(64, (3, 3), activation='relu', padding='same')(a2)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8e)

    u1 = UpSampling2D((2, 2), interpolation='bilinear')(c8)
    a1 = Concatenate(axis=3)([u1, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(a1)

    c10 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)
    markers = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    cell_mask = Conv2D(2, (1, 1), activation='softmax', padding='same')(c10)
    output = Concatenate(axis=3)([markers, cell_mask])

    model = Model(input_img, output)

    model.compile(optimizer=Adam(lr=0.0001), loss=loss_function)

    print('Model was created')

    return model


def predict_images(images, model_path, mi=512, ni=512):
    """
    process images by NN
    :param images: np array in format (n_images, x, y, depth)
    :param model_path:
    :return:
    """
    model = make_model(mi=mi, ni=ni)
    model.load_weights(model_path)

    result = model.predict(images, batch_size=8)
    del model
    return result


def find_cell_mask_threshold(prediction, dataset, sequence):
    """
    reads GT and choose the most appropriate threshold
    :param dataset:
    :param sequence:
    :return:
    """

    PAD = 50
    gt_path = os.path.join(dataset, f'{sequence}_GT', 'SEG')
    gt_names = os.listdir(gt_path)

    score = np.zeros(256)

    for name in gt_names:

        gt = cv2.imread(os.path.join(gt_path, name), cv2.IMREAD_ANYDEPTH)
        gt = gt > 0

        frame_number = int(name[7:10])

        r3 = prediction[frame_number, :, :, -1]

        cv2.imwrite(os.path.join('tmp', name), gt*255)
        cv2.imwrite(os.path.join('tmp', name + '.png'), r3)
        for i in range(256):
            cross_section = r3 > i
            intersection = cross_section[PAD:-PAD, PAD:-PAD] & gt[PAD:-PAD, PAD:-PAD]
            union = cross_section[PAD:-PAD, PAD:-PAD] | gt[PAD:-PAD, PAD:-PAD]
            score[i] += np.sum(intersection) / np.sum(union)

    score /= len(gt_names)
    print('evaluation', np.argmax(score), np.max(score))
    return np.argmax(score)


if __name__ == '__main__':

    os.chdir('/home/xlux/cbia/')
    model_path = '/home/xlux/cbia/SUBMISSION/SW_new11/DIC-C2DH-HeLa/unet_model320_nord8+8_s1_1202_PREDICT.h5'
    dataset = 'DIC-C2DH-HeLa'
    sequence = '01'

    model = make_model()
    model.load_weights(model_path)
    img_path = os.path.join(dataset, f'{sequence}')
    images = load_images(img_path)
    prediction = model.predict(images)*255

    print(find_cell_mask_threshold(prediction, dataset, sequence))
