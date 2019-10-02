'''
Author: Remi Tavon
Date: 5 juin 2019

To run:
module load cuda cudnn/7.4
source $HOME/pytorch_env/bin/activate
cd pytorch-segmentation-master/src
python pipeline.py --base_dir '../data/sherbrooke/leftImg8bit/val/*'
'''

#Class: GSV
#Attributes: metadata (date, orientation, latlong, ...), split
#TODO: add all attributes of dataframe to GSV images used in pipeline
#TODO: in vis.segmentation overlay sidewalk mask and default bbox

#Class: defaults
#Attributes: bbox, class,

import argparse
import numpy as np
#import pandas as pd
import matplotlib
matplotlib.use('Agg')
import cv2
from PIL import Image
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.net import SPPNet
from src.utils.metrics import compute_iou_batch
from src.utils import minmax_normalize, meanstd_normalize, topcrop, rescale
from utils.custum_aug import PadIfNeededRightBottom

print('Modules and packages imported')

#df = pd.read_csv('../data/sherbrooke/meta_centro_rad5,6,8.csv', sep = ';', parse_dates =['imageDate'], index_col=['index'], encoding='latin-1')

class GSVDataset(Dataset):
    def __init__(self, base_dir='../data/GSV_sherb_centro_2012-2609', net_type='deeplab', debug = False):
        self.debug = debug
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.img_paths = sorted(self.base_dir.glob(f'*.*g'))
        if len(self.img_paths) == 0:
            raise AssertionError(f'No images found. Check current working directory.')

    def __len__(self):
        return len(self.img_paths)

    def img_preproc(self, image, top_crop = True, downscale = True):

        if top_crop:
        #pre-processing for original GSV panoramas (6656*3328)
        #if img.shape[0] >= 3328 and img.shape[1] >= 6656:
            image = topcrop(image, topcrop_prop=0.5)
        if downscale:
            if max(image.shape[0], image.shape[1]) >= 6656: # most probably a 360 panorama
                image = rescale(image, max_size=4096)
            elif max(image.shape[0], image.shape[1]) > 2048:
                image = rescale(image, max_size=1536)

        image = minmax_normalize(image, norm_range=(-1, 1))
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)
        return image

    def seg_mask(self, image, model, top_crop = True, downscale=True):
        '''
        Returns prediction mask as np array for sidewalk on image with value = 1 for sidewalk
        :param model_path: image as np array
        :param model: pytorch model to use for semantic segmentation
        :param vis: if True, will save labels as .png files and create .jpg overlays with image
        :return: per-pixel mask for sidewalk as np array of same dimensions as image
        '''
        model.eval()  # set model mode to eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # work on GPU if available

        image_torch = self.img_preproc(image, top_crop=top_crop, downscale=downscale)
        image_torch = image_torch[None]  # mimick dataloader with 4th channel (batch channel)
        image_torch = image_torch.to(device)  # send image to device

        # output: predictions (segmentation maps with horizontal flip, i.e. test time augmentation)
        pred = model.tta(image_torch,
                         net_type='deeplab')  # TODO try without tta. don't think it will have a significant effect on iou
        pred = pred.argmax(dim=1)  # take channel with highest pixel value as winning class
        pred = pred.detach().cpu().numpy()  # send back to cpu, convert to numpy

        # take first pred of single-item list of preds. Reduce item to 3D.
        pred = pred[0]

        # set all pixel in pred corresponding to an ignore_pixel in label to 0
        # pred[label == dataset.ignore_index] = 0

        if topcrop:
            #if pred.shape[1] / pred.shape[0] == 4:
            pred = topcrop(pred, reverse=True)  # restore top half of prediction mask by filling with black values
        if downscale:
            #if pred.shape != image.shape[:2] and pred.shape[1] / pred.shape[0] == image.shape[1] / image.shape[0]:
            if pred.shape[1] / pred.shape[0] != image.shape[1] / image.shape[0]:
                print(f'Unable to rescale two arrays with different aspect ratios.\n'
                      f'Prediction shape is {pred.shape[1]}x{pred.shape[0]} '
                      f'whereas image shape is {image.shape[1]}x{image.shape[1]}.')
            # resize pred to image dimensions
            else:
                pred = cv2.resize(src=pred, dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        return pred

def save_to_image(image, img_path, output_dir, suffix):
    _, filename = os.path.split(img_path)
    file, _ = os.path.splitext(filename)  # separate name and extension

    output_path = Path(f'{img_path.parent.parent}/{str(output_dir)}/{img_path.parent.stem}')
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        print('This function handles PIL images, not numpy arrays. Trying to convert to PIL image...')
        image = Image.fromarray(image)

    filename = f'{file}_{str(suffix)}.jpg'
    image.save(output_path.joinpath(f'{filename}'))
    print(f'Files were saved to {output_path}/{filename}')

def vis_segmentation(pred, img, bbox = False):
    '''

    :param pred: prediction mask as numpy array
    :param img: image used for inference as numpy array
    :return: image as PIL image with overlay of bbox from mask (if bbox) or of prediction (in color) (if not bbox)
    '''
    assert pred.shape == img.shape[:2]
    if bbox:
        img = draw_bbox(pred.astype(np.uint8), img)
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

def load_model(output_channels=2, model_path='../model/sherbrooke_deeplabv3p_lgpu30-2/model.pth'):
    '''
    Returns torch model with pretrained weights

    :param dataset:
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

def bbox_from_mask(pred, image, hue = 1):
    '''

    :param pred:
    :param image:
    :param hue: value for sidewalk class in pred mask
    :return:
    '''

    #reconvert to RGB from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Minimum percentage of pixels of same hue to consider dominant colour
    #MIN_PIXEL_CNT_PCT = (1.0/10000.0)

    # Let's count the number of occurrences of each hue
    #bins = np.bincount(lbl.flatten())
    # And then find the dominant hues
    #lbl_hues = np.where(bins > (lbl.size * MIN_PIXEL_CNT_PCT))[0])
    #pred_hues = np.unique(pred)

    # Now let's find the shape matching each dominant hue
    #for i, hue in enumerate(pred_hues):
    #    if hue == 0:
    #        continue
        # First we create a mask selecting all the pixels of this hue
    #    mask = cv2.inRange(lbl_encoded, np.array(hue), np.array(hue))
        # And use it to extract the corresponding part of the original colour image
    #    blob = cv2.bitwise_and(image, image, mask=mask)

    mask = cv2.inRange(pred, np.array(hue), np.array(hue))
    blob = cv2.bitwise_and(image, image, mask=mask)

    return blob

def draw_bbox(mask, image): #TODO get this function to work
    '''

    :param image:
    :param mask:
    :return:
    '''
    #TODO: filter out small contours under a certain area in pixels
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for j, contour in enumerate(contours):
        bbox = cv2.boundingRect(contour)
        # Create a mask for this contour
        contour_mask = np.zeros_like(mask)

        #result = cv2.bitwise_and(image, image, mask=contour_mask)
        # And draw a bounding box
        top_left, bottom_right = (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3])
        #cv2.rectangle(result, top_left, bottom_right, (255, 255, 255), 2)
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

    return image

#class streetview_pano(Image):
#    def __init__(self):
        #self.debug = debug
#        self.img_paths = sorted(self.base_dir.glob(f'*.*g'))

#for gsv in gsv_dataset:
#produire un masque pour le trottoir avec modèle
#utiliser ce masque pour extraire les blobs de trottoirs
#détecter les défauts avec petit modèle
#signaler lorsqu'un défaut est trouvé (copier l'image? imprimer la bbox, imprimer la coord et l'orientation absolue?)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--output_ch', type=int)
    parser.add_argument('--base_dir')
    args,leftover = parser.parse_known_args()

    if args.base_dir is None:
        basedir = '../data/sherbrooke/leftImg8bit/val/*'
    else:
        basedir = args.base_dir

    output_channels = 2

    model_path = '../model/sherbrooke_deeplabv3p_lgpu32-1/model.pth'
    crack_model_path = '../model/sherbrooke_deeplabv3p_lgpu_mnv2_202-1/model.pth'

    dataset = GSVDataset(base_dir=basedir, net_type='deeplab')  # reach cityscapes dataset, validation split

    print(f'Arguments used: \n'
          f'output channels : {output_channels}\nmodel path : {model_path}')
    #output_path = Path(args.output_path)
    #output_path.parent.mkdir()

    model = load_model(model_path=model_path)
    crack_model = load_model(model_path=crack_model_path)

    count=0
    with torch.no_grad():
        #dataloader is a 2 element list with images and labels as torch tensors
        print('Generating predictions...')
        for i in range(len(dataset)):
            count += 1
            if count % 10 == 0:
                print(f'Evaluation progress: {count}/{len(dataset.img_paths)}')

            #1. Load image from dataset, make copy of original image for later use
            img_path = dataset.img_paths[i]
            orig_img = np.array(Image.open(img_path))
            #img = dataset.img_preproc(orig_img)
            #image = dataset[i]

            #2. Create inference for image with specified torch model
            pred = dataset.seg_mask(orig_img, model=model, top_crop=True, downscale=True)

            overlay = vis_segmentation(pred, orig_img)
            #save_to_image(overlay, img_path, Path(model_path).parent.stem, 'sidewalk_overlay')

            blob = bbox_from_mask(np.int8(pred), orig_img)

            #save_to_image(Image.fromarray(blob), img_path, Path(model_path).parent.stem, 'blob')

            crack_pred = dataset.seg_mask(blob, model=crack_model, top_crop=True, downscale=False)

            #convert all black values in blob to black values in prediction
            crack_pred[blob[:, :, 0] == 0] = 0
            if 1 in np.unique(crack_pred):
                crack_overlay = vis_segmentation(crack_pred, np.array(overlay), bbox=True)
                save_to_image(crack_overlay, img_path, Path(model_path).parent.stem, 'default_overlay')
                print('Found crack!')




