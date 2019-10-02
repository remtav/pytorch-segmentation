import numpy as np
import cv2
import torch
from utils.preprocess_pipeline import minmax_normalize
from PIL import Image


def threshold(pred, value=253):
    '''Apply threshold on numpy array.

    Args:
        pred (np.array): array on which threshold will be applied
        value: value under which array values will be converted to 0

    return: prediction as numpy array with 0 and 1 values only.
    '''
    threshold = value
    pred[pred < threshold] = 0
    pred[pred >= threshold] = 1
    return pred


def filter_by_activation(pred, percentile=99, norm_range=(0, 255), threshold = True, debug=True):
    '''Filter feature map given by logits according to probability of pixel belonging to class c

    pred (np.array): 3d array to filter activation values on (class probabilities after softmax, height, width)
    percentile (int): percentile used to find threshold probability value.
        The higher the percentile, the higher the probability of value chosen as threshold
    norm_rang (tuple): range to spread values across after filtering operation

    return: 2D numpy array as float 64 (or 32?), normalized
    '''
    #pred = softmax_from_feat_map(pred)

    # take first pred of single-item list of preds. Reduce item to 3D.
    #pred = pred[1, :, :]

    # pred = preprocessing.scale(pred, axis=0, with_mean=True, with_std=True, copy=True)
    # pred = meanstd_normalize(pred, np.mean(pred), np.std(pred))

    perc = round(len(np.unique(pred)) * (int(percentile) / 100))
    val_at_perc = np.unique(pred)[perc]
    if debug:
        print(f'Value of {percentile}th percentile in prediction is: {val_at_perc}. All values smaller will be given this value.')
    pred[pred < val_at_perc] = val_at_perc

    pred = minmax_normalize(pred, norm_range=norm_range, orig_range=(min(np.unique(pred)), max(np.unique(pred))))
    # pred = preprocessing.minmax_scale(pred, feature_range=(0, 255))
    pred = pred.astype(np.uint8)

    pred = threshold(pred, value=253)
    return pred


def filter_by_contour(mask, thresh_area=200, debug = True): #TODO get this function to work
    '''Filter objects in prediction by size in square pixel

    Args:
        mask (np.array): array with zero and non-zero values (to filter)
        thresh_area (int): area to consider while filtering. Anything smaller will be converted to zero value.

    return: np.array
    '''
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #ncontours = sorted(contours, key=cv2.contourArea, reverse=True)
    #rect = cv2.minAreaRect(ncontours[0])

    bin_ct = np.bincount(mask.flatten())
    if debug:
        print(f'Bin count BEFORE filtering by contour area: {bin_ct}')

    for j, contour in enumerate(contours):
        #bbox = cv2.boundingRect(contour)

        #if contour's area is smaller than 200 pixels, change value to 0 (background)
        cont_area = cv2.contourArea(contour)
        if cont_area < int(thresh_area):
            cv2.drawContours(mask, contour, -1, (0), 0)
            if debug:
                print(f'Contour {j} was filtered out and converted to background due to area smaller than {thresh_area} ({cont_area})')

    if debug:
        bin_ct = np.bincount(mask.flatten())
        print(f'Bin count AFTER filtering by contour area: {bin_ct}')

    return mask
