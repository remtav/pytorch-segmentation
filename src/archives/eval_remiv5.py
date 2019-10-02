# To run:
# cd ./data/pytorch-segmentation-master/src
# python eval_remiv3.py test 19 ../model/sherbrooke_deeplabv3p_lgpu8/model.pth
# python eval_remiv5.py test 2 ../model/sherbrooke_deeplabv3p_lgpu30-2/model.pth

#TODO: adapt script for images with no corresponding labels.
#TODO: save all predictions to .png. Then evaluate with evaluation module in cityscapescripts.
#TODO: find correspondance to original image from image saved in figure, at end of script.

print('Importing modules and packages...')

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
from src.models.net import SPPNet
from src import SherbrookeDataset

from PIL import Image
from pathlib import Path
import os

print('Modules and packages imported')

#Global variables
filetype = 'pred'
colortype = 'color'
output_folder = f'colormap_{colortype}' #only used in print. To change, go to save_colormap fct

def vis_segmentation(pred, img_path):
    colormap = np.asarray([
        [255, 255, 255, 0],
        [244, 35, 232, int(255 * 0.3)],
        [70, 70, 70, int(255 * 0.3)],
        [102, 102, 156, int(255 * 0.3)],
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

def eval_from_model(split, output_channels, model_path, vis = True):

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
    del param # delete parameters? Reduce memory usage?

    dataset = SherbrookeDataset(split=split, net_type='deeplab') #reach cityscapes dataset, validation split
    img_paths = dataset.img_paths
    base_dir = dataset.base_dir
    split = dataset.split
    print(f'{len(img_paths)} images found in {base_dir}\\{split}')

    model.eval() #apply eval method on model. ?

    print(f'Files containing \'{filetype}\' will be converted to \'{colortype}\' colormap and saved to:\n{output_folder}')

    count=0
    with torch.no_grad():
        #dataloader is a 2 element list with images and labels as torch tensors
        print('Generating predictions...')
        for i in range(len(dataset)):
            count += 1
            if count % 10 == 0:
                print(f'Evaluation progress: {count}/{len(img_paths)}')
            image = dataset[i]
            img_path = dataset.img_paths[i]
            _tmp_, filename = os.path.split(img_path)
            file, ext = os.path.splitext(filename)  # separate name and extension
            image = image[None]
            image = image.to(device)
            #next line reaches to tta.py --> net.py --> xception.py ...
            #output: predictions (segmentation maps)
            pred = model.tta(image, net_type='deeplab')
            pred = pred.argmax(dim=1)
            pred = pred.detach().cpu().numpy()
            pred = pred[0]

            #print(np.unique(pred))
            if output_channels == 19:
                #create mask for values other than 0 (background) and 1(sidewalk)
                for i in range(2,19):
                    pred[pred == i] = 0 #convert these classes to background value

            if pred.shape[0] == 1024 and pred.shape[1] == 4096:
                top_restore = np.zeros((1024,pred.shape[1]), dtype=int)
                #bot_restore = np.zeros((82, pred.shape[1]), dtype=int)
                pred = np.concatenate((top_restore, pred), axis=0)
                #pred = np.concatenate((top_restore, pred, bot_restore), axis=0)

            #savename = f'{count:03d}_{filetype}_{yyyy_mm_dd}'
            savename = file

            #Image.fromarray(pred[0].astype(np.uint8)).save(output_dir.joinpath(f'{savename}_pred.png'))
            #print(np.unique(pred))

            output_dir = Path(f'../data/output/{model_dir}/{split}/{os.path.split(img_path.parent)[1]}')
            output_dir.mkdir(parents=True, exist_ok=True)

            if vis:
                overlay = vis_segmentation(pred, img_path)
                folder = output_dir.joinpath('figures')
                folder.mkdir(parents=True, exist_ok=True)
                overlay.save(folder.joinpath(f'{savename}_overlay.jpg'))

            #convert 1 values to 8. For bootstrapping.
            pred = encode_mask(pred)

            pred_pil = Image.fromarray(pred.astype(np.uint8))
            img_pil = Image.open(img_path)
            if pred_pil.size != img_pil.size:
                pred_pil = pred_pil.resize((img_pil.size[0], img_pil.size[1]), Image.NEAREST)

            pred_pil.save(output_dir.joinpath(f'{savename}_gtFine_labelIds.png'))

            #save_colormap(pred[0], savename, output_dir, filetype, colortype=colortype)

    print(f'Files were be saved to {output_dir.parent}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('split')
    parser.add_argument('output_channels', type=int)
    parser.add_argument('model_path')
    args = parser.parse_args()

    split = args.split
    output_channels = args.output_channels
    model_path = args.model_path
    #output_path = Path(args.output_path)
    #output_path.parent.mkdir()
    eval_from_model(split, output_channels, model_path, vis=True)

