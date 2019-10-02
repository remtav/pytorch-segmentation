#adapted from: https://stackoverflow.com/questions/50051916/bounding-box-on-objects-based-on-color-python

import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os

def encode_mask(lbl):
    # create mask for values other than 0 (background) and 1(sidewalk)
    for i in range(34, 37):
        lbl[lbl == i] = 8  # convert these classes to sidewalk-no-default
    return lbl

def contour_proc(binary_array, image):
    '''compute bboxes and model scores from a mask/image pair

    :param binary_array:
    :param image:
    :param img_path:

    return: dictionnary with bbox coordinates and model scores for contours in each image
        Format: {'filename': {'boxes': [l,t,r,b][][][]}, 'scores': [], 'filename': {...}}
    '''


    # let's find the shape matching each dominant hue
    # First we create a mask selecting all the pixels of this hue
    mask = cv2.inRange(binary_array, np.array(1), np.array(1))
    # And use it to extract the corresponding part of the original colour image


    blob = cv2.bitwise_and(image, image, mask=mask)

    # lbl_blob = cv2.bitwise_and(mask, mask, mask=mask)
    # blob_name = f"{filename}-{i}-hue_{hue:.3f}-blob.png"
    # cv2.imwrite(os.path.join(output_dir, blob_name), blob)

    # extract contours from binary array
    #_, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bbox_list = []
    scores_list = []

    with tqdm(contours) as _tqdm:
        for contour in _tqdm:
            x,y,w,h = cv2.boundingRect(contour)
            bbox = [x, y, x + w, y + h]
            contour_mask = np.zeros_like(mask)
            # And draw a bounding box
            # top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])
            # cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
            # cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

            bbox_list.append(bbox)

            # Extract and save the area of the contour on label
            blob_region = blob.copy()[y:y + h, x:x + w]
            # region_mask = contour_mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # blob_crop = cv2.bitwise_and(blob_region, blob_region, mask=region_mask)
            blob_region = np.where(blob_region != 0, blob_region, np.nan)

            score = np.nanmean(blob_region)
            scores_list.append(float(score))

    return bbox_list, scores_list