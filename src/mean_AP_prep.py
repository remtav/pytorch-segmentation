'''
To run:
module load cuda cudnn/7.4
source $HOME/pytorch_env/bin/activate
cd pytorch-segmentation-master/src

python mean_AP_prep.py --split val --output_ch 2 --model_path ../model/sherbrooke_deeplabv3p_lgpu_mnv2_202-1/model.pth
'''

#TODO: adapt script for images with no corresponding labels.
#TODO: save all predictions to .png. Then evaluate with evaluation module in cityscapescripts.
#TODO: find correspondance to original image from image saved in figure, at end of script.

print('Importing modules and packages...')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
from models.net import SPPNet
from dataset.sherbrooke import SherbrookeDataset
from utils.metrics import compute_iou_batch
from utils.preprocess import minmax_normalize, meanstd_normalize, topcrop, rescale
from utils.postprocess import filter_by_activation, filter_by_contour, threshold, softmax_from_feat_map
from utils.contour_processing import contour_proc

from PIL import Image
from pathlib import Path
import os
import json
from tqdm import tqdm
from collections import OrderedDict

print('Modules and packages imported')

#Global variables
filetype = 'pred'
colortype = 'color'
output_folder = f'colormap_{colortype}' #only used in print. To change, go to save_colormap fct


def eval_from_model(split, output_channels, model_path, postproc = False, vis = True, debug = True):

    model_path = Path(model_path)
    path, model_dir = os.path.split(model_path.parent)  # separate path and filename
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #work on GPU if available

    print(f'Device: {device}')

    if 'mnv2' in model_dir:
        model = SPPNet(enc_type='mobilenetv2', dec_type = 'maspp', output_channels = output_channels).to(device)
        defaults = True
    else:
        model = SPPNet(output_channels=output_channels).to(device)
        defaults = False

    if device == torch.device('cpu'):
        param = torch.load(model_path, map_location='cpu')  # parameters saved in checkpoint via model_path
    else:
        param = torch.load(model_path)  # parameters saved in checkpoint via model_path

    print(f'Parameters loaded from {model_path}')

    model.load_state_dict(param) #apply method load_state_dict to model?
    del param # delete parameters? Reduce memory usage?

    dataset = SherbrookeDataset(split=split, net_type='deeplab', defaults = defaults) #reach cityscapes dataset, validation split
    classes = np.arange(1, dataset.n_classes)
    img_paths = dataset.img_paths
    base_dir = dataset.base_dir
    split = dataset.split
    if len(img_paths) == 0:
        raise ValueError('Your dataset seems empty...')
    else:
        print(f'{len(img_paths)} images found in {base_dir}\\{split}')

    model.eval() #apply eval method on model. ?

    #print(f'Files containing \'{filetype}\' will be converted to \'{colortype}\' colormap and saved to:\n{output_folder}')

    valid_ious = []
    count = 0
    predicted_boxes = {}
    ground_truth_boxes = {}

    with torch.no_grad():
        #dataloader is a 2 element list with images and labels as torch tensors
        print('Generating predictions...')

        with tqdm(range(len(dataset))) as _tqdm:
            for i in _tqdm:
                count += 1
                image, label = dataset[i]
                img_path = dataset.img_paths[i]
                #filename = img_path.stem
                filename = img_path.name

                #if isinstance(image, tuple): #take only image in label is also returned by __getitem__
                #    image = image[0]

                image = image[None] # mimick dataloader with 4th channel (batch channel)
                image = image.to(device)
                # next line reaches to tta.py --> net.py --> xception.py ...
                # output: predictions (segmentation maps)
                pred = model.tta(image, net_type='deeplab')
                # pred = model(image)
                # pred = F.interpolate(pred, size=label.shape, mode='bilinear', align_corners=True)
                # pred = pred.argmax(dim=1)
                pred = pred.detach().cpu().numpy()
                label = label.numpy()

                # take first pred of single item list of preds...
                pred = pred[0]

                pred = softmax_from_feat_map(pred)

                # take channel corresponding to softmax scores in class 1. Reduces array to 2D
                pred = pred[1, :, :]

                if pred.shape[1] / pred.shape[0] == 4:
                    pred = topcrop(pred, reverse=True)
                    label = topcrop(label, reverse=True)

                if debug:
                    print(f'Prediction shape after evaluation: {pred.shape}\nLabel shape: {label.shape}')

                if defaults:
                    # set all pixel in pred corresponding to an ignore_pixel in label to 0
                    pred[label == dataset.ignore_index] = 0

                #perc = round(len(np.unique(pred)) *0.5) #find index at median
                #val_at_perc = np.unique(pred)[perc]
                val_at_perc = 0.0002
                #print(
                #    f'Value at median in prediction is: {val_at_perc}')


                pred_masked = np.where(pred >= val_at_perc, pred, np.nan)
                pred_binary = threshold(pred.copy(), value=val_at_perc) # set values under 0.5 to 0, else to 1. result: binary array
                bbox_list, scores_list = contour_proc(pred_binary, pred_masked)

                #add key to predicted_boxes: {'filename': {'boxes':bbox_list, 'scores':scores_list}}
                predicted_boxes.update({filename: {"boxes": bbox_list, "scores": scores_list}})

                #pred = filter_by_activation(pred, percentile=90)
                #pred = threshold(pred)

                bbox_list_lbl, _ = contour_proc(label, label.copy())

                #add key to predicted_boxes: {'filename': {'boxes':bbox_list, 'scores':scores_list}}
                ground_truth_boxes.update({filename: bbox_list_lbl})

                if debug:
                    print(f'Label unique values: {np.unique(label)}')

                _tqdm.set_postfix(OrderedDict(last_image=f'{filename}'))

    with open('predicted_boxes_GSV.json', 'w') as json_file:
        json.dump(predicted_boxes, json_file, sort_keys=True)

    with open('ground_truth_boxes_GSV.json', 'w') as json_file:
        json.dump(ground_truth_boxes, json_file, sort_keys=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split')
    parser.add_argument('--output_ch', type=int)
    parser.add_argument('--model_path')
    args,leftover = parser.parse_known_args()

    if args.split is None:
        split = 'val'
    else:
        split = args.split
    if args.output_ch is None:
        output_channels = 2
    else:
        output_channels = args.output_ch
    if args.model_path is None:
        model_path = '../model/sherbrooke_deeplabv3p_lgpu30-2/model.pth'
    else:
        model_path = args.model_path

    print(f'Arguments used: \nsplit : {split}\n'
          f'output channels : {output_channels}\nmodel path : {model_path}')
    #output_path = Path(args.output_path)
    #output_path.parent.mkdir()
    eval_from_model(split, output_channels, model_path, postproc=True, vis=False, debug=False)

