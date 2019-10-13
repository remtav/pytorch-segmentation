'''
Author: Remi Tavon
Date: 5 juin 2019

To run:
module load cuda cudnn/7.4
source $HOME/pytorch_env/bin/activate
cd pytorch-segmentation-master/src
python pipelinev2.py --base_dir ../data/sherbrooke/leftImg8bit/val
'''

#Class: GSV
#Attributes: metadata (date, orientation, latlong, ...), split
#TODO: add all attributes of dataframe to GSV images used in pipeline
#TODO: in vis.segmentation overlay sidewalk mask and default bbox

#Class: defaults
#Attributes: bbox, class,

import argparse
import datetime

import numpy as np
#import pandas as pd
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt #importing matplotlib

import cv2
from PIL import Image
from pathlib import Path
import os

import torch

from models.net import SPPNet
from utils.metrics import compute_iou_batch
from utils.preprocess_pipeline import minmax_normalize, meanstd_normalize, topcrop, rescale
from utils.custum_aug import PadIfNeededRightBottom

print('Modules and packages imported')

#df = pd.read_csv('../data/sherbrooke/meta_centro_rad5,6,8.csv', sep = ';', parse_dates =['imageDate'], index_col=['index'], encoding='latin-1')

def create_img_list(base_dir='../data/GSV_sherb_centro_2012-2609'):
    base_dir = Path(base_dir)
    #img_paths = sorted(base_dir.glob('leftImg8bit/val/mtl-marche-central/*.*g'))
    img_paths = sorted(base_dir.glob('**/*.*g'))
    if len(img_paths) == 0:
        raise AssertionError(f'No images found. Check current working directory.')
    return img_paths


def img_preproc(image):

    image = minmax_normalize(image, norm_range=(-1, 1))
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image)

    return image


def load_model(output_channels=2, model_path='../model/sherbrooke_deeplabv3p_lgpu30-2/model.pth'):
    '''
    Returns torch model with pretrained weights

    :param output_channels:
    :param model_path:
    :return:
    '''
    model_path = Path(model_path)
    path, model_dir = os.path.split(model_path.parent)  # separate path and filename
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #work on GPU if available

    print(f'Device: {device}')
    if 'mnv2' in model_dir:
        model = SPPNet(enc_type='mobilenetv2', dec_type = 'maspp', output_channels = output_channels).to(device)
    else:
        model = SPPNet(output_channels=output_channels).to(device)

    if device == torch.device('cpu'):
        param = torch.load(model_path, map_location='cpu')  # parameters saved in checkpoint via model_path
    else:
        param = torch.load(model_path)  # parameters saved in checkpoint via model_path

    print(f'Parameters loaded from {model_path}')

    model.load_state_dict(param) #apply method load_state_dict to model?
    del param # why delete parameters? Reduce memory usage?

    return model

def GrayScaleHistogram(img_array):
    nb_bins = len(np.unique(img_array))
    min_bin_val = min(np.unique(img_array))
    max_bin_val = max(np.unique(img_array))
    plt.hist(img_array.ravel(), bins=nb_bins, range=(min_bin_val, max_bin_val), fc='k', ec='k')  # calculating histogram
    # find frequency of pixels in range 0-255
    #histr = cv2.calcHist([img_array], [0], None, [nb_bins], [min_bin_val, max_bin_val])

    # show the plotting graph of an image
    #plt.plot(histr)
    #plt.show()
    return plt

def filter_by_contour(mask, thresh_area=200):
    '''

    :param image: image (np.array) on which we will draw bounding boxes
    :param mask: mask we will extract bboxes from. Must be binary (0 and 1 values only)
    :return: image (np.array) with bboxes drawn on top, in specified color and thickness.
    '''
    #_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #ncontours = sorted(contours, key=cv2.contourArea, reverse=True)
    #rect = cv2.minAreaRect(ncontours[0])

    bin_ct = np.bincount(mask.flatten())
    print(f'Bin count BEFORE filtering by contour area: {bin_ct}')

    for j, contour in enumerate(contours):
        #bbox = cv2.boundingRect(contour)

        #if contour's area is smaller than 200 pixels, change value to 0 (background)
        cont_area = cv2.contourArea(contour)
        if cont_area < thresh_area: #if contour area smaller than threshold
            cv2.drawContours(mask, contour, -1, (0, 0, 0), 0) #fill with 0 values
            print(f'Contour {j} was filtered out and converted to background due to area smaller than {thresh_area} ({cont_area})')

    bin_ct = np.bincount(mask.flatten())
    print(f'Bin count AFTER filtering by contour area: {bin_ct}')

    return mask


def filter_by_activation(pred, percentile=99, debug = True, raw_output = False):
    '''

    :param pred: torch tensor corresponding to output of model
    :return: numpy array as float 64 (or 32?), normalized
    '''
    if debug:
        pred_copy = pred.argmax(dim=1)
        pred_copy = pred_copy[0].numpy()  # send back to cpu, convert to numpy
        pred_copy = pred_copy.astype(np.uint8)
        bin_ct = np.bincount(pred_copy.flatten())
        print(f'Bin count if using "argmax" instead of filtering by activation: {bin_ct}')

    pred = pred.numpy() # convert pred to numpy ndarray of dtype float32

    # take first pred of single-item list of preds. Reduce item to 3D.
    pred = pred[0, 1, :, :]

    # Compute value at given percentile
    perc = round(len(np.unique(pred)) * (percentile / 100))
    val_at_perc = np.unique(pred)[perc]

    if raw_output:
        pred[pred < val_at_perc] = val_at_perc
        print(f'Value of {percentile}th percentile in prediction is: {val_at_perc}. All values smaller will be given this value.')

    else:
        pred[pred < val_at_perc] = 0
        pred[pred >= val_at_perc] = 1

        pred = pred.astype(int)

        if debug:
            bin_ct = np.bincount(pred.flatten())
            print(f'Bin count AFTER filtering by activation to {percentile}th percentile in activation values: {bin_ct}')
            pass
    return pred


def threshold(pred, threshold_value=253):
    threshold = threshold_value
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    return pred


def seg_mask(image, model, top_crop = True, downscale=True, softmax_filter_perc=99, debug=True):
    '''
    Returns prediction mask as np array for sidewalk on image with value = 1 for sidewalk
    :param model_path: image as np array
    :param model: pytorch model to use for semantic segmentation
    :param vis: if True, will save labels as .png files and create .jpg overlays with image
    :return: per-pixel mask for sidewalk as np array of same dimensions as image
    '''
    model.eval()  # set model mode to eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # work on GPU if available

    if debug:
        print(f'initial image shape: {image.shape}')

    if downscale:
        image_shape = (image.shape[0], image.shape[1])
        if max(image_shape) >= 6656:  # most probably a 360 panorama
            image = rescale(image, max_size=4096)
        elif max(image_shape) > 2048:
            image = rescale(image, max_size=1536)
    if top_crop:
        croppix = round(image.shape[0] * 0.5)  #crop top half of image
        image = topcrop(image, croppix)

    if debug:
        print(f'final image shape: {image.shape}')

    image_torch = img_preproc(image)
    image_torch = image_torch[None]  # mimick dataloader with 4th channel (batch channel)
    image_torch = image_torch.to(device)  # send image to device

    # output: predictions (segmentation maps with horizontal flip, i.e. test time augmentation)
    torch_pred = model.tta(image_torch, net_type='deeplab')  # TODO try without tta. don't think it will have a significant effect on iou
    torch_pred = torch_pred.detach().cpu()
    softmax = torch.nn.Softmax(dim=1)
    softmax_pred = softmax(torch_pred)

    if softmax_filter_perc:
        bin_pred = filter_by_activation(softmax_pred, percentile=softmax_filter_perc)

    else:
        bin_pred = softmax_pred.argmax(dim=1)  # take channel with highest pixel value as winning class
        bin_pred = softmax_pred[0].numpy()  # send back to cpu, convert to numpy

    # set all pixel in pred corresponding to an ignore_pixel in label to 0
    # pred[label == dataset.ignore_index] = 0

    if top_crop:
        bin_pred = topcrop(bin_pred, croppix=croppix, reverse=True)  # restore top half of prediction mask by filling with black values
    if downscale:
        if max(image_shape) > 2048:
            #pred = rescale(pred, max_size=max_dim)
            bin_pred = cv2.resize(src=bin_pred, dsize=(image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        #if round(pred.shape[1] / pred.shape[0]) == round(image.shape[1] / image.shape[0]):
            #pred = cv2.resize(src=pred, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            #pred = rescale(pred, max_size=max(image.shape[0], image.shape[1]))
        # resize pred to image dimensions
        #else:
        #    print(f'Unable to rescale two arrays with different aspect ratios.\n'
        #          f'Prediction shape is {pred.shape[1]}x{pred.shape[0]} '
        #          f'whereas image shape is {image.shape[1]}x{image.shape[0]}.')
    if debug:
        print(f'final prediction shape: {bin_pred.shape}') #\nimage shape: {image.shape}')
    return bin_pred, softmax_pred.numpy()


def vis_segmentation(pred, img, bbox = False, cont_thresh = 200):
    '''

    :param pred: prediction mask as numpy array
    :param img: image used for inference as numpy array
    :return: image as PIL image with overlay of bbox from mask (if bbox) or of prediction (in color) (if not bbox)
    '''
    assert pred.shape == img.shape[:2]
    if cont_thresh:
        assert bbox, 'Activate bbox option if thresholding by contour area'
    if bbox:
        img = draw_bbox(pred.astype(np.uint8), img, bbox_or_cont='bbox', contour_thresh=cont_thresh)
        img_pil = Image.fromarray(img)
        return img_pil
    else:
        colormap = np.asarray([
            [255, 255, 255, 0],
            [244, 35, 232, int(255 * 0.3)], #cityscapes sidewalk-pink
            [0, 0, 205, int(255 * 0.3)], #blue3
            [0, 205, 0, int(255 * 0.3)], #green3
            [190, 153, 153, int(255 * 0.3)],
            [153, 153, 153, int(255 * 0.3)]])
        pred_color = colormap[pred]
        pred_pil = Image.fromarray(pred_color.astype(np.uint8))
        img_pil = Image.fromarray(img)
        background = img_pil
        foreground = pred_pil
        background.paste(foreground, (0, 0), foreground)

        return background


def encode_mask(lbl):
    lbl[lbl == 1] = 8
    return lbl


def bbox_from_mask(pred, image, hue = 1):
    '''

    :param pred: numpy array
    :param image: image as numpy array
    :param hue: grayscale value to use in blob creation
    :return: blobs of image for corresponding pixels in prediction mask where value is [hue]
    '''

    mask = cv2.inRange(pred, np.array(hue), np.array(hue))
    blob = cv2.bitwise_and(image, image, mask=mask)

    return blob


def draw_bbox(mask, image, bbox_or_cont = 'bbox', contour_thresh=200): #TODO get this function to work
    '''

    :param image: image (np.array) on which we will draw bounding boxes
    :param mask: mask we will extract bboxes from. Must be binary (0 and 1 values only)
    :return: image (np.array) with bboxes drawn on top, in specified color and thickness.
    '''
    #TODO: filter out small contours under a certain area in pixels
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for j, contour in enumerate(contours):
        cont_area = cv2.contourArea(contour)
        if contour_thresh and cont_area>=contour_thresh:
            image = draw_contour(image, contour, j, bbox_or_cont=bbox_or_cont)
            print(f'Contour {j+1} was printed on image. Area ({cont_area}) bigger than {contour_thresh}')
        elif contour_thresh and cont_area < contour_thresh:
            print(f'Contour {j+1} was filtered out due to area ({cont_area}) smaller than {contour_thresh}')
        else:    # if no contour_thresh, print all 'em darn contours.
            image = draw_contour(image, contour, j, bbox_or_cont=bbox_or_cont)
            print(f'Printed contour {j+1} with area ({cont_area})')
    return image


def draw_contour(image, c, i, debug=True, bbox_or_cont='bbox'):
    # compute the center of the contour area and draw a circle
    # representing the center
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # set values as what you need in the situation
        cX, cY = 0, 0

    cont_area = cv2.contourArea(c)

    if bbox_or_cont == 'bbox':
        bbox = cv2.boundingRect(c)
        # Create a mask for this contour
        # contour_mask = np.zeros_like(mask)

        # result = cv2.bitwise_and(image, image, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])
        # cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    elif bbox_or_cont == 'cont':
        cv2.drawContours(image, c, -1, (255, 0, 0), 2)
    else:
        raise NotImplementedError

    # draw the contour number on the image
    cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY - 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2)

    # draw the area value on the image
    cv2.putText(image, "Area: {}".format(cont_area), (cX - 20, cY - 80), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 2)

    if debug:
        print(f'Printed area #{i+1} with contour area of {cont_area}')

    # return the image with the contour number drawn on it
    return image


def save_to_image(image, img_path, output_path, suffix):
    _, filename = os.path.split(img_path)
    file, _ = os.path.splitext(filename)  # separate name and extension

    if isinstance(image, np.ndarray):
        print('This function handles PIL images, not numpy arrays. Trying to convert to PIL image...')
        image = Image.fromarray(image.astype(np.uint8))

    filename = f'{file}_{str(suffix)}.jpg'
    image.save(output_path.joinpath(f'{filename}'))
    print(f'Files were saved to {output_path}/{filename}')

if __name__ == '__main__':

    now = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir')
    args,leftover = parser.parse_known_args()

    if args.base_dir is None:
        basedir = '/lustre04/scratch/remit/panoimg[]_1-2609'
    else:
        basedir = args.base_dir

    model_path = '../model/sherbrooke_deeplabv3p_lgpu32-1/model.pth'
    crack_model_path = '../model/sherbrooke_deeplabv3p_lgpu_mnv2_205-2/model.pth'

    print(f'Arguments used: \nmodel path : {model_path}')

    img_paths = create_img_list(base_dir=basedir)
    print(f'Base directory: {basedir}')

    output_dir_pref = str(Path(model_path).parent.stem).split('deeplabv3p_')[-1]
    output_dir_suf = str(Path(crack_model_path).parent.stem).split('deeplabv3p_')[-1]
    output_dir = output_dir_pref+'_'+output_dir_suf+'_'+now

    output_path = Path(f'{img_paths[0].parent.parent}/{output_dir}/{img_paths[0].parent.stem}')
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_path}')

    model = load_model(model_path=model_path)
    crack_model = load_model(model_path=crack_model_path)
    activation_perc = 99
    cont_thresh = 400

    count = 0
    crack_count = 0
    with torch.no_grad():
        #dataloader is a 2 element list with images and labels as torch tensors
        print('Generating predictions...')
        for img_path in img_paths:
            count += 1
            if count % 10 == 0:
                print(f'Evaluation progress: {count}/{len(img_paths)}')

            ## Load image to infer on.

            #1. Load image from dataset, make copy of original image for later use
            orig_img = np.array(Image.open(img_path))

            if orig_img.shape[1] >= 6656 and orig_img.shape[0] >= 3328:
                crop = True
            else:
                crop = False

            ##Sidewalk segmentation

            #2. Create inference for image with specified torch model
            pred, _ = seg_mask(orig_img, model=model, top_crop=crop, downscale=True)

            #3 Set all values other than 1 (sidewalk) to 0. Sends back black images with only predicted sidewalks visible
            blob = bbox_from_mask(np.int8(pred), orig_img)

            # Optionally save blob.
            #save_to_image(Image.fromarray(blob), img_path, Path(model_path).parent.stem, 'blob')



            ## Crack segmentation

            #4 Create segmentation prediction for cracks on sidewalks previously segmented
            crack_pred, softmax_crack_pred = seg_mask(blob, model=crack_model, top_crop=False, downscale=False, softmax_filter_perc=activation_perc)
            # Convert all black values in blob red channel to black values in prediction.
            # Indirect way to blacken prediction outside region of interest
            crack_pred[blob[:, :, 0] == 0] = 0

            # If softmax_filter returns raw outputs with all softmax probabilities *not* converted to 0 and 1 values.
            softmax_crack_pred = softmax_crack_pred[0, 1, :, :]
            # Convert all black values in blob red channel to black values in prediction.
            # Indirect way to blacken prediction outside region of interest
            softmax_crack_pred = minmax_normalize(softmax_crack_pred, norm_range=(0,255), orig_range=(min(np.unique(softmax_crack_pred)), max(np.unique(softmax_crack_pred))))
            softmax_crack_pred[blob[:, :, 0] == 0] = 0
            softmax_crack_pred = softmax_crack_pred.astype(np.uint8)

            if 'TESTS' in basedir:
                # Optionally save sidewalk segmentation.
                save_to_image(Image.fromarray(pred.astype(np.uint8)), img_path, output_path, '1_sidewalk_seg')
                # Save heatmap
                save_to_image(Image.fromarray(softmax_crack_pred.astype(np.uint8)), img_path, output_path, '2_heatmap_crack_seg')
                # Save crack prediction segmentation
                save_to_image(Image.fromarray(crack_pred.astype(np.uint8)), img_path, output_path, '3_crack_seg')

                # Optional: print stats on crack predictions and show plot
                print(f'Bin count: {np.bincount(crack_pred.flatten())[-16:]}')
                #plot = GrayScaleHistogram(crack_pred)
                plt.imshow(crack_pred, cmap='flag')

                plot_fname = Path(f'{output_path}/{img_path.stem}_perc{activation_perc}.png')
                plt.savefig(plot_fname)
                plt.close()

            #if working with blobs already
            #crack_pred = seg_mask(orig_img, model=crack_model, top_crop=crop, downscale=False, heatmap=True)
            #crack_pred[orig_img[:, :, 0] == 0] = 0

            if 1 in np.unique(crack_pred):
                crack_count += 1
                #pass
                #overlay = vis_segmentation(pred, orig_img)
                #save_to_image(overlay, img_path, Path(model_path).parent.stem, 'sidewalk_overlay')
                #crack_overlay = vis_segmentation(crack_pred, np.array(overlay), bbox=True)
                #crack_overlay = vis_segmentation(crack_pred, blob, bbox=True)
                crack_overlay = vis_segmentation(crack_pred, orig_img, bbox=True, cont_thresh=cont_thresh)
                save_to_image(crack_overlay, img_path, output_path, f'filter_{activation_perc}th_c{cont_thresh}')

                #if crack_count % 10 == 0:
                #    break

        print('Job done!')