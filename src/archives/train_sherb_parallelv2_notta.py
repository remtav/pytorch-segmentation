#to execute: python train_sherb_parallelv2.py ../config/sherbrooke_deeplabv3p.yaml

print('Importing modules and packages...')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pickle
import argparse
import yaml
import numpy as np
import albumentations as albu
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import csv
import datetime

from src.models.net import EncoderDecoderNet, SPPNet, EfficientUnet
from losses.multi import MultiClassCriterion
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from src.utils.metrics import compute_iou_batch

print('Modules and packages imported')

#deals with arguments submitted next to .py filename in terminal
parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()
config_path = Path(args.config_path)
config = yaml.load(open(config_path))
net_config = config['Net']
data_config = config['Data']
train_config = config['Train']
loss_config = config['Loss']
opt_config = config['Optimizer']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t_max = opt_config['t_max']
best_epoch = 0

max_epoch = train_config['max_epoch']
batch_size = train_config['batch_size']
fp16 = train_config['fp16']
resume = train_config['resume']
pretrained_path = train_config['pretrained_path']
try:
    freeze_to_layer = train_config['freeze_to_layer']
    freeze_bn = train_config['freeze_bn']
    load_logits = train_config['load_logits']
except KeyError:
    freeze_to_layer = False
    freeze_bn = False
    load_logits = True

def deterministic_mode(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed=seed)
    import random
    random.seed(a=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def load_model(out_channels=2, enc_type='efficientnet', dec_type='unet', pretrained=True, output_stride=8):
    # Network
    global net_type
    if 'unet' in dec_type:
        net_type = 'unet'
        if 'efficient' in enc_type:
            model = EfficientUnet(enc_type, out_channels=out_channels, concat_input=True,
                                         pretrained=pretrained)#, model_name=enc_type)
        else:
            model = EncoderDecoderNet(**net_config)
    else:
        net_type = 'deeplab'
        model = SPPNet(output_channels=out_channels, enc_type=enc_type, dec_type=dec_type, output_stride=output_stride)
    return model

# Deterministic training
fix_seed = False
if fix_seed:
    seed = 1234
    deterministic_mode(seed)

#Network
model = load_model(out_channels=net_config['output_channels'], enc_type=net_config['enc_type'],
                   dec_type=net_config['dec_type'], pretrained=net_config['pretrained'])

dataset = data_config['dataset']
if dataset == 'pascal':
    from src.dataset.pascal_voc import PascalVocDataset as Dataset
    net_config['output_channels'] = 21
    classes = np.arange(1, 21)
elif dataset == 'cityscapes':
    from src.dataset.cityscapes import CityscapesDataset as Dataset
    net_config['output_channels'] = 19
    classes = np.arange(1, 19)
elif dataset == 'sherbrooke':
    from src.dataset.sherbrooke import SherbrookeDataset as Dataset
    #net_config['output_channels'] = 2
    #classes = np.arange(1, 2)
else:
    raise NotImplementedError
del data_config['dataset']

modelname = config_path.stem
output_dir = Path('../model') / modelname
output_dir.mkdir(exist_ok=True)
log_dir = Path('../logs') / modelname
log_dir.mkdir(exist_ok=True)

logger = debug_logger(log_dir)
logger.debug(config)
logger.info(f'Device: {device}')
logger.info(f'Max Epoch: {max_epoch}')
logger.info(f'Batch size: {batch_size}')

if fix_seed:
    logger.info(f'Deterministic mode activated. Seed = {seed}')

# Loss
print('Initializing loss function, optimizer and scheduler...')
loss_fn = MultiClassCriterion(**loss_config).to(device)
params = model.parameters()
optimizer, scheduler = create_optimizer(params, **opt_config)

# history
if resume:
    with open(log_dir.joinpath('history.pkl'), 'rb') as f:
        history_dict = pickle.load(f)
        best_metrics = history_dict['best_metrics']
        loss_history = history_dict['loss']
        iou_history = history_dict['iou']
        start_epoch = len(iou_history)
        for _ in range(start_epoch):
            if scheduler:
                scheduler.step()
else:
    start_epoch = 0
    best_metrics = 0
    loss_history = []
    iou_history = []

# Dataset
print('Initializing dataset and data loader...')
affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                 # Rotate(5, p=.5)
                                 ])
# image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
#                                 albu.RandomBrightnessContrast(p=.5)])
image_augmenter = albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5)
#image_augmenter = None
train_dataset = Dataset(affine_augmenter=affine_augmenter, image_augmenter=image_augmenter,
                        net_type=net_type, **data_config)
classes = np.arange(1, train_dataset.n_classes)
#logger.info(f'{train_dataset.resizer_info}')
valid_dataset = Dataset(split='valid', net_type=net_type, **data_config)

if torch.cuda.device_count() > 1:
    num_workers = torch.cuda.device_count()*4
else:
    num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

logger.info(f'Number of workers for train Dataloader: {num_workers}')

# Pretrained model
if pretrained_path:
    logger.info(f'Resume from {pretrained_path}')
    if device.type == 'cpu':
        param = torch.load(pretrained_path, map_location='cpu')
    else:
        param = torch.load(pretrained_path)
    pretrained_dict = param
    model_dict = model.state_dict()

    if load_logits==False or model.state_dict()['logits.weight'].shape != pretrained_dict['logits.weight'].shape:

        #logits_sidewalks = pretrained_dict['logits.weight'][1]
        #logits_sky = pretrained_dict['logits.weight'][10]
        # 1. filter out unnecessary keys
#        for k,v in pretrained_dict.items():
#            if k.find('logits') == -1:
#                pretrained_dict[k]=v
#            else:
#                pretrained_dict[k][0] = logits_sky
#                pretrained_dict[k][1] = logits_sidewalks
        #dict_variable = {key:value for (key,value) in dictonary.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('logits') == -1}
        logger.info('Pretrained model loaded without logits layer.')
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        #model_dict['logits.weight'][0].update(logits_sky)
        #model_dict['logits.weight'][1].update(logits_sidewalks)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        #set logits parameters to parameters in 19 class model, but keeping only logits for sidewalk and sky
        #model.logits.weight.data[0] = logits_sidewalks
        #model.logits.weight.data[1] = logits_sky
    else:
        model.load_state_dict(param)
    del param

# Restore model
if resume:
    model_path = output_dir.joinpath(f'model_tmp.pth')
    logger.info(f'Resume from {model_path}')
    if device.type == 'cpu':
        param = torch.load(model_path, map_location='cpu')
    else:
        param = torch.load(model_path)
    # new_state_dict = OrderedDict()
    # for k, v in param.items():
    #     if k.startswith('encoder._'):
    #     #    continue
    #     #elif k.startswith('encoder'):
    #         #name = k[:8]+'_'+k[8:] # add '_'
    #         name = k[:8]+k[9:] # remove '_'
    #         new_state_dict[name] = v
    #     #elif k.startswith('_'):
    #     #    name = k[1:] # remove '_'
    #     #    new_state_dict[name] = v
    #     elif k.startswith('conv' or 'blocks' or 'bn' or 'fc'):
    #         name = '_' + k  # add '_'
    # # load params
    # model.load_state_dict(new_state_dict)

    model.load_state_dict(param)

    del param
    opt_path = output_dir.joinpath(f'opt_tmp.pth')
    param = torch.load(opt_path)
    optimizer.load_state_dict(param)

    #added for type error "expected type torch.FloatTensor but got torch.cuda.FloatTensor"
    #appeared when training from checkpoint saved with Dataparallel. 
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    del param

# Freeze layers
freeze_switch = 0
if freeze_to_layer:
    parameter_dict = dict(model.named_parameters())
    #param_count = 0
    for name, param in parameter_dict.items():
    #    layer_count+=1
    #    for tensor in param:
    #        param_count += tensor.detach().numpy().size
        #if layer_count <= 213:
        if name.find(freeze_to_layer) != -1:
            freeze_switch = 1
        if freeze_switch == 0:
            if isinstance(param, nn.parameter.Parameter):
                print(name, ' : frozen')
                param.requires_grad = False
        else:
            print(name, ': will be trained')
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer, scheduler = create_optimizer(params, **opt_config)
    print('Only parameters from unfrozen layers were submitted to optimizer')

# fp16
if fp16:
    #from apex import fp16_utils
    #model = fp16_utils.BN_convert_float(model.half())
    #optimizer = fp16_utils.FP16_Optimizer(optimizer, verbose=False, dynamic_loss_scale=True)

    model = model.half()  # convert to half precision
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float()

    logger.info('Apply fp16')

#Multi-GPU parallelism
if torch.cuda.device_count() > 1:
    #fp16 = False
    logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

    #torch.distributed.init_process_group(backend="nccl")
    #model = nn.parallel.DistributedDataParallel(model)

model = model.to(device)

# Train
for i_epoch in range(start_epoch, max_epoch):
    logger.info(f'Epoch: {i_epoch}')
    logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

    train_losses = []
    train_ious = []

    model.train()
    if freeze_bn:
        logger.info(f'BatchNorm layers will be turned to "eval" mode, i.e. frozen.')
        model.apply(set_bn_eval)

    #tqdm for progress bar
    # with tqdm(train_loader) as _tqdm:
    #     for batched in _tqdm:
    #         images, labels = batched
    #         if fp16:
    #             images = images.half()
    #         images, labels = images.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         preds = model(images)
    #         if net_type == 'deeplab':
    #             preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
    #         if fp16:
    #             loss = loss_fn(preds.float(), labels)
    #         else:
    #             loss = loss_fn(preds, labels)
    #
    #         preds_np = preds.detach().cpu().numpy()
    #         labels_np = labels.detach().cpu().numpy()
    #         iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)
    #
    #         _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
    #         train_losses.append(loss.item())
    #         train_ious.append(iou)
    #
    #         if fp16:
    #             optimizer.backward(loss)
    #         else:
    #             loss.backward()
    #         optimizer.step()

    if scheduler:
        scheduler.step()

    train_loss = np.mean(train_losses)
    train_iou = np.nanmean(train_ious)
    logger.info(f'train loss: {train_loss}')
    logger.info(f'train iou: {train_iou}')

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), output_dir.joinpath('model_tmp.pth'))
    else:
        torch.save(model.state_dict(), output_dir.joinpath('model_tmp.pth'))
    torch.save(optimizer.state_dict(), output_dir.joinpath('opt_tmp.pth'))

    if (i_epoch + 1) % 4 == 0:
        valid_losses = []
        valid_ious = []

        # Network switch to evaluation mode
        model.eval()

        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    images, labels = batched
                    if fp16:
                        images = images.half()
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)
                    if net_type == 'deeplab':
                        preds = F.interpolate(preds, size=labels.shape[1:], mode='bilinear', align_corners=True)
                    if fp16:
                        loss = loss_fn(preds.float(), labels)
                    else:
                        loss = loss_fn(preds, labels)

                    preds_np = preds.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    iou = compute_iou_batch(np.argmax(preds_np, axis=1), labels_np, classes)

                    _tqdm.set_postfix(OrderedDict(seg_loss=f'{loss.item():.5f}', iou=f'{iou:.3f}'))
                    valid_losses.append(loss.item())
                    valid_ious.append(iou)

        valid_loss = np.mean(valid_losses)
        valid_iou = np.nanmean(valid_ious)
        logger.info(f'valid seg loss: {valid_loss}')
        logger.info(f'valid iou: {valid_iou}')

        if best_metrics < valid_iou:
            best_metrics = valid_iou
            best_epoch = i_epoch
            logger.info('Best Model!')
            torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
            torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))

    else:
        valid_loss = None
        valid_iou = None

    loss_history.append([train_loss, valid_loss])
    iou_history.append([train_iou, valid_iou])
    history_ploter(loss_history, log_dir.joinpath('loss.png'))
    history_ploter(iou_history, log_dir.joinpath('iou.png'))

    history_dict = {'loss': loss_history,
                    'iou': iou_history,
                    'best_metrics': best_metrics}
    with open(log_dir.joinpath('history.pkl'), 'wb') as f:
        pickle.dump(history_dict, f)

    #cuda memory usage
    #print(torch.cuda.max_memory_allocated(device=device))

    #early stopping: if no best model after 200 epochs, break training loop
    if i_epoch == best_epoch + 600:
        logger.info('Early stopping: Last best model was recored 600 epochs ago. No improvements since based on validation iou')
        break
    elif best_metrics < 0.8 and i_epoch >= 1000:
        logger.info('Early stopping: Reached 1000th epoch and best validation iou smaller than 0.8')
        break

#get date
now=datetime.datetime.now()
yyyy_mm_dd= now.isoformat()[:10]

#compute mean for train and valid loss and iou values for all epochs
loss_array = np.array(loss_history, dtype=np.float)
train_loss_mean = np.nanmean(np.array(loss_array)[:,0])
val_loss_mean = np.nanmean(np.array(loss_array)[:,1])

iou_array = np.array(iou_history, dtype=np.float)
train_iou_mean = np.nanmean(np.array(iou_array)[:,0])
val_iou_mean = np.nanmean(np.array(iou_array)[:,1])

# File to save first results
out_file = 'hyperparameter_tuning.csv'

# Write to the csv file ('a' means append)
of_connection = open(out_file, 'a')
writer = csv.writer(of_connection, delimiter=';')
writer.writerow([modelname, net_config['output_channels'], net_config['enc_type'], net_config['dec_type'],
                 net_config['output_stride'], 'sherbrooke', data_config['target_size'], max_epoch,
                 batch_size, resume, pretrained_path, freeze_to_layer, freeze_bn, loss_config['loss_type'], opt_config['mode'], opt_config['base_lr'],
                 opt_config['t_max'], best_metrics, best_epoch, i_epoch, train_loss_mean, val_loss_mean, train_iou_mean, val_iou_mean, yyyy_mm_dd])
of_connection.close()