'''
To run:
module load cuda cudnn/7.4
source $HOME/pytorch_env/bin/activate
cd pytorch-segmentation-master/src

python eval_remiv6.py --split val --output_ch 2 --model_path ../model/sherbrooke_deeplabv3p_lgpu_mnv2_202-1/model.pth
python eval_remiv6.py --split val --output_ch 2 --model_path ../model/sherbrooke_deeplabv3p_lgpu30-2/model.pth
'''

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
from utils.postprocess import filter_by_activation, filter_by_contour, threshold
from utils.contour_processing import contour_proc

from PIL import Image
from pathlib import Path
import os
import json
from collections import OrderedDict
from tqdm import tqdm

print('Modules and packages imported')

#Global variables
filetype = 'pred'
colortype = 'color'
output_folder = f'colormap_{colortype}' #only used in print. To change, go to save_colormap fct

def vis_segmentation(pred, img_path):
    colormap = np.asarray([
        [255, 255, 255, 0],
        [227, 207, 87, int(255 * 0.3)], #banana
        [0, 0, 205, int(255 * 0.3)], #blue3
        [0, 205, 0, int(255 * 0.3)], #green3
        [190, 153, 153, int(255 * 0.3)],
        [153, 153, 153, int(255 * 0.3)]])
    pred_color = colormap[pred]
    pred_pil = Image.fromarray(pred_color.astype(np.uint8))
    img_pil = Image.open(img_path)
    if pred_pil.size != img_pil.size:
        pred_pil = pred_pil.resize((img_pil.size[0], img_pil.size[1]), Image.NEAREST)
    background = img_pil
    foreground = pred_pil

    background.paste(foreground, (0, 0), foreground)
    return background


def encode_mask(lbl):
    lbl[lbl == 1] = 8
    return lbl


def eval_from_model(split, output_channels, model_path, postproc = False, vis = True, debug = True, mean_AP=False):

    model_path = Path(model_path)
    path, model_dir = os.path.split(model_path.parent)  # separate path and filename
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #work on GPU if available

    print(f'Device: {device}')

    if 'mnv2' in model_dir:
        model = SPPNet(enc_type='mobilenetv2', dec_type = 'maspp', output_channels = output_channels).to(device)
        defects = True
    else:
        model = SPPNet(output_channels=output_channels).to(device)
        defects = False

    if device == torch.device('cpu'):
        param = torch.load(model_path, map_location='cpu')  # parameters saved in checkpoint via model_path
    else:
        param = torch.load(model_path)  # parameters saved in checkpoint via model_path

    print(f'Parameters loaded from {model_path}')

    model.load_state_dict(param) #apply method load_state_dict to model?
    del param # delete parameters? Reduce memory usage?

    dataset = SherbrookeDataset(split=split, net_type='deeplab', defects = defects) #reach cityscapes dataset, validation split
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
                if count % 1 == 10:
                    print(f'Evaluation progress: {count}/{len(img_paths)}')
                image, label = dataset[i]
                img_path = dataset.img_paths[i]
                orig_image = np.array(Image.open(img_path))
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

                # take first pred of single item list of preds...
                pred = pred[0]

                softmax = torch.nn.Softmax(dim=1)
                pred = softmax(pred)
                #pred = softmax_from_feat_map(pred)

                pred = pred.detach().cpu().numpy()
                label = label.numpy()

                if pred.shape[1] / pred.shape[0] == 2:
                    pred, label = dataset.postprocess(pred,label)

                if mean_AP and not postproc:
                    raise Exception ('postproc argument in eval_from_model function must be true if mean_AP is set to True')
                elif postproc:
                    # take channel corresponding to softmax scores in channel 1 (class 1). Reduces array to 2D
                    pred = pred[1, :, :]

                    if dataset.defects:
                        # set all pixel in pred corresponding to an ignore_pixel in label to 0
                        pred[label == dataset.ignore_index] = 0

                    if mean_AP:
                        val_at_perc = 0.0002
                        # print(
                        #    f'Value at median in prediction is: {val_at_perc}')

                        #create array copy and wreplace all values under threshold by nan values
                        pred_masked = np.where(pred >= val_at_perc, pred, np.nan)

                        #create copy of pred array and set all values above threshold to 1 and under to 0
                        pred_binary = threshold(pred.copy(),
                                                value=val_at_perc)

                        # set values under 0.5 to 0, else to 1. result: binary array
                        bbox_list, scores_list = contour_proc(pred_binary, pred_masked)

                        # add key to predicted_boxes: {'filename': {'boxes':bbox_list, 'scores':scores_list}}
                        predicted_boxes.update({filename: {"boxes": bbox_list, "scores": scores_list}})

                        # pred = filter_by_activation(pred, percentile=90)
                        # pred = threshold(pred)

                        bbox_list_lbl, _ = contour_proc(label, label.copy())

                        # add key to predicted_boxes: {'filename': {'boxes':bbox_list, 'scores':scores_list}}
                        ground_truth_boxes.update({filename: bbox_list_lbl})

                        pred_masked = np.where(pred >= val_at_perc, pred, np.nan)
                        pred_binary = threshold(pred.copy(), value=val_at_perc) # set values under 0.5 to 0, else to 1. result: binary array
                        bbox_list, scores_list = contour_proc(pred_binary, pred_masked)

                        #add key to predicted_boxes: {'filename': {'boxes':bbox_list, 'scores':scores_list}}
                        predicted_boxes.update({filename: {"boxes": bbox_list, "scores": scores_list}})

                    pred = filter_by_activation(pred, percentile=90)

                else:
                    pred = np.argmax(pred, axis=0)

                if debug:
                    print(f'Label unique values: {np.unique(label)}')

                # print(np.unique(pred))
                if output_channels == 19:
                    # create mask for values other than 0 (background) and 1(sidewalk)
                    for i in range(2,19):
                        pred[pred == i] = 0 # convert these classes to background value

                if dataset.split == 'val':
                    # compute iou
                    iou = compute_iou_batch(pred, label, classes)
                    print(f'Iou for {filename}: {iou}')
                    valid_ious.append(iou)

                if vis:

                    output_dir = Path(f'../data/output/{model_dir}/{split}/{os.path.split(img_path.parent)[1]}')
                    output_dir.mkdir(parents=True, exist_ok=True)

                    folder = output_dir.joinpath('figures')
                    folder.mkdir(parents=True, exist_ok=True)
                    label[label == 255] = 0
                    conf_overlay = np.add(label, pred*2)
                    print(np.unique(conf_overlay))
                    confus_overlay = vis_segmentation(conf_overlay, img_path)
                    confus_overlay.save(folder.joinpath(f'{filename}_overlay.jpg'))

                elif dataset.split == 'bootstrap':
                    # convert 1 values to 8. For bootstrapping.
                    pred = encode_mask(pred)

                    pred_pil = Image.fromarray(pred.astype(np.uint8))
                    img_pil = Image.open(img_path)
                    if pred_pil.size != img_pil.size:
                        pred_pil = pred_pil.resize((img_pil.size[0], img_pil.size[1]), Image.NEAREST)

                    pred_pil.save(output_dir.joinpath(f'{filename}_gtFine_labelIds.png'))
                    #save_colormap(pred[0], savename, output_dir, filetype, colortype=colortype)
                else:
                    raise NotImplementedError

            _tqdm.set_postfix(OrderedDict(last_image=f'{filename}'))

    if mean_AP:
        with open('predicted_boxes_GSV.json', 'w') as json_file:
            json.dump(predicted_boxes, json_file, sort_keys=True)

        with open('ground_truth_boxes_GSV.json', 'w') as json_file:
            json.dump(ground_truth_boxes, json_file, sort_keys=True)

    if dataset.split == 'val':
        valid_iou = np.nanmean(valid_ious)
        print(f'mean valid iou: {valid_iou}')
        #print(f'Confusion matrix: \n{conf_mat}')

    with open('predicted_boxes_GSV.json', 'w') as json_file:
        json.dump(predicted_boxes, json_file)  # , sort_keys=True)

    with open('ground_truth_boxes_GSV.json', 'w') as json_file:
        json.dump(ground_truth_boxes, json_file, sort_keys=True)

    if vis:
        print(f'Files were be saved to {output_dir.parent}')

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
    eval_from_model(split, output_channels, model_path, postproc=True, vis=True, debug=True, mean_AP=True)

