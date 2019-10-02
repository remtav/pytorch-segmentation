#to execute: 
#python train_multi.py ../config/sherbrooke_deeplabv3p_lgpu40.yaml

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

from models.net_multi import EncoderDecoderNet, SPPNetMulti#SPPNetEncoder, SPPNetDecoder
from losses.multi import MultiClassCriterion
from logger.log import debug_logger
from logger.plot import history_ploter
from utils.optimizer import create_optimizer
from utils.metrics import compute_iou_batch

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
#pretrained_path = train_config['pretrained_path']
try:
    freeze_to_layer = train_config['freeze_to_layer']
    freeze_bn = train_config['freeze_bn']
except KeyError:
    freeze_to_layer = False
    freeze_bn = False

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

# Deterministic training
fix_seed = False
if fix_seed:
    seed = 1234
    deterministic_mode(seed)

# Network
if 'unet' in net_config['dec_type']:
    net_type = 'unet'
    model = EncoderDecoderNet(**net_config)
else:
    net_type = 'deeplab'
    #encoder = SPPNetEncoder(**net_config)
    #decoder_task1 = SPPNetDecoder(**net_config)
    #decoder_task2 = SPPNetDecoder(**net_config)
    #decoder_task3 = SPPNetDecoder(**net_config)
    model = SPPNetMulti(**net_config)

dataset = data_config['dataset']
if dataset == 'pascal':
    from dataset.pascal_voc import PascalVocDataset as Dataset
    net_config['output_channels'] = 21
    classes = np.arange(1, 21)
elif dataset == 'cityscapes':
    from dataset.cityscapes import CityscapesDataset as Dataset
    net_config['output_channels'] = 19
    classes = np.arange(1, 19)
elif dataset == 'sherbrooke':
    from dataset.sherbrooke_multi import SherbrookeDataset as Dataset
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

    iou_history1 = []
    iou_history2 = []
    iou_history3 = []
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
classes = np.arange(1, train_dataset.n_classes) #arange(1,2)
#logger.info(f'{train_dataset.resizer_info}')
valid_dataset = Dataset(split='valid', net_type=net_type, **data_config)

if torch.cuda.device_count() > 1:
    num_workers = torch.cuda.device_count()*4
else:
    num_workers = 0

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)
#valid_loader = DataLoader(valid_dataset, batch_size=torch.cuda.device_count(), shuffle=False, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

logger.info(f'Number of workers for train Dataloader: {num_workers}')

# Pretrained model
# see train_sherb_parallelv2.py. Has been sent to SPP* net classes.

# Set fp16 to false if multi-GPU
if torch.cuda.device_count() > 1:
    fp16 = False

# Restore model
if resume:
    model_path = output_dir.joinpath(f'model_tmp.pth')
    logger.info(f'Resume from {model_path}')
    param = torch.load(model_path)
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
    from apex import fp16_utils
    model = fp16_utils.BN_convert_float(model.half())
    optimizer = fp16_utils.FP16_Optimizer(optimizer, verbose=False, dynamic_loss_scale=True)
    logger.info('Apply fp16')

#Multi-GPU parallelism
if torch.cuda.device_count() > 1:
    fp16 = False
    logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model = model.to(device)

# Train
for i_epoch in range(start_epoch, max_epoch):
    logger.info(f'Epoch: {i_epoch}')
    logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')

    train_losses1 = []
    train_losses2 = []
    train_losses3 = []

    train_losses_mean = []

    train_ious1 = []
    train_ious2 = []
    train_ious3 = []

    train_ious_mean = []

    model.train()
    if freeze_bn:
        logger.info(f'BarchNorm layers will be turned to "eval" mode, i.e. frozen.')
        model.apply(set_bn_eval)

    #tqdm for progress bar
    with tqdm(train_loader) as _tqdm:
        for batched in _tqdm:
            images, labels_sidewalk, labels_defects, labels_maj_defects = batched
            if fp16:
                images = images.half()
            images, labels_sidewalk, labels_defects, labels_maj_defects = images.to(device), labels_sidewalk.to(device), labels_defects.to(device), labels_maj_defects.to(device)
            optimizer.zero_grad()
            #preds = model(images)
            mask1, mask2, mask3 = model(images)
            if net_type == 'deeplab':
                mask1 = F.interpolate(mask1, size=labels_sidewalk.shape[1:], mode='bilinear', align_corners=True)
                mask2 = F.interpolate(mask2, size=labels_sidewalk.shape[1:], mode='bilinear', align_corners=True)
                mask3 = F.interpolate(mask3, size=labels_sidewalk.shape[1:], mode='bilinear', align_corners=True)
            # if fp16:
            #     loss = loss_fn(preds.float(), labels)
            # else:
            #     loss = loss_fn(preds, labels)
            loss1 = loss_fn(mask1, labels_sidewalk)
            loss2 = loss_fn(mask2, labels_defects)
            loss3 = loss_fn(mask3, labels_maj_defects)

            loss = loss1 + 10*loss3

            mask1_np = mask1.detach().cpu().numpy()
            labels_sidewalk_np = labels_sidewalk.detach().cpu().numpy()
            iou1 = compute_iou_batch(np.argmax(mask1_np, axis=1), labels_sidewalk_np, classes)

            mask2_np = mask2.detach().cpu().numpy()
            labels_defects_np = labels_defects.detach().cpu().numpy()
            iou2 = compute_iou_batch(np.argmax(mask2_np, axis=1), labels_defects_np, classes)

            mask3_np = mask3.detach().cpu().numpy()
            labels_maj_defects_np = labels_maj_defects.detach().cpu().numpy()
            iou3 = compute_iou_batch(np.argmax(mask3_np, axis=1), labels_maj_defects_np, classes)

            iou = np.nanmean([iou1, iou2, iou3])

            _tqdm.set_postfix(OrderedDict(seg_loss1=f'{loss1.item():.5f}', seg_loss2=f'{loss2.item():.5f}', seg_loss3=f'{loss3.item():.5f}',
                                          iou_task1=f'{iou1:.3f}', iou_task2=f'{iou2:.3f}', iou_task3=f'{iou3:.3f}'))

            train_losses1.append(loss1.item())
            train_losses2.append(loss2.item())
            train_losses3.append(loss3.item())
            train_losses_mean.append(loss.item())

            train_ious1.append(iou1)
            train_ious2.append(iou2)
            train_ious3.append(iou3)
            train_ious_mean.append(iou)

            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()

    if scheduler:
        scheduler.step()

    train_loss1 = np.mean(train_losses1)
    train_loss2 = np.mean(train_losses2)
    train_loss3 = np.mean(train_losses3)
    train_loss_all = np.mean(train_losses_mean)

    train_iou1 = np.nanmean(train_ious1)
    train_iou2 = np.nanmean(train_ious2)
    train_iou3 = np.nanmean(train_ious3)
    train_iou_all = np.nanmean(train_ious_mean)

    logger.info(f'train loss task 1: {train_loss1.item():.5f}\ntrain loss task 2: {train_loss2.item():.5f}\ntrain loss task 3: {train_loss3.item():.5f}')
    logger.info(f'train iou task 1: {train_iou1:.3f}\ntrain iou task2: {train_iou2:.3f}\ntrain iou task3: {train_iou3:.3f}')

    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), output_dir.joinpath('model_tmp.pth'))
    else:
        torch.save(model.state_dict(), output_dir.joinpath('model_tmp.pth'))
    torch.save(optimizer.state_dict(), output_dir.joinpath('opt_tmp.pth'))

    if (i_epoch + 1) % 2 == 0:
        valid_losses = []
        valid_ious1 = []
        valid_ious2 = []
        valid_ious3 = []
        valid_ious = []

        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    images, labels_sidewalk, labels_defects, labels_maj_defects = batched
                    if fp16:
                        images = images.half()
                    images, labels_sidewalk, labels_defects, labels_maj_defects = images.to(device), labels_sidewalk.to(
                        device), labels_defects.to(device), labels_maj_defects.to(device)
                    #preds = val_model.tta(images, net_type=net_type)
                    mask1, mask2, mask3 = model(images)
                    #print (f'Mask 3 initial shape:{mask3.shape}\nSidewalk label shape:{labels_sidewalk.shape}')
                    if net_type == 'deeplab':
                        mask1 = F.interpolate(mask1, size=labels_sidewalk.shape[1:], mode='bilinear',
                                              align_corners=True)
                        mask2 = F.interpolate(mask2, size=labels_sidewalk.shape[1:], mode='bilinear',
                                              align_corners=True)
                        mask3 = F.interpolate(mask3, size=labels_sidewalk.shape[1:], mode='bilinear',
                                              align_corners=True)

                    #print(f'Mask 3 final shape:{mask3.shape}')

                    val_loss1 = loss_fn(mask1, labels_sidewalk)
                    val_loss2 = loss_fn(mask2, labels_defects)
                    val_loss3 = loss_fn(mask3, labels_maj_defects)
                    val_loss = val_loss1 + val_loss2 + val_loss3

                    mask1_np = mask1.detach().cpu().numpy()
                    labels_sidewalk_np = labels_sidewalk.detach().cpu().numpy()
                    val_iou1 = compute_iou_batch(np.argmax(mask1_np, axis=1), labels_sidewalk_np, classes)

                    mask2_np = mask2.detach().cpu().numpy()
                    labels_defects_np = labels_defects.detach().cpu().numpy()
                    val_iou2 = compute_iou_batch(np.argmax(mask2_np, axis=1), labels_defects_np, classes)

                    mask3_np = mask3.detach().cpu().numpy()
                    labels_maj_defects_np = labels_maj_defects.detach().cpu().numpy()
                    val_iou3 = compute_iou_batch(np.argmax(mask3_np, axis=1), labels_maj_defects_np, classes)

                    val_iou = np.nanmean([val_iou1, val_iou2, val_iou3])

                    _tqdm.set_postfix(OrderedDict(seg_loss1=f'{val_loss1.item():.5f}', seg_loss2=f'{val_loss2.item():.5f}',
                                                  seg_loss3=f'{val_loss3.item():.5f}',
                                                  iou_task1=f'{val_iou1:.3f}', iou_task2=f'{val_iou2:.3f}',
                                                  iou_task3=f'{val_iou3:.3f}'))

                    valid_losses.append(val_loss.item())
                    #valid_losses.append(val_loss3.item())
                    valid_ious.append(val_iou)
                    valid_ious1.append(val_iou1)
                    valid_ious2.append(val_iou2)
                    valid_ious3.append(val_iou3)

        valid_loss = np.mean(valid_losses)
        valid_iou = np.nanmean(valid_ious)
        valid_iou1 = np.nanmean(valid_ious1)
        valid_iou2 = np.nanmean(valid_ious2)
        valid_iou3 = np.nanmean(valid_ious3)

        #logger.info(
            #f'val loss task 1: {val_loss1.item():.5f}\nval loss task 2: {val_loss2.item():.5f}\nval loss task 3: {val_loss3.item():.5f}')
            #f'val loss task 3: {val_loss3.item():.5f}')
        logger.info(f'val iou task 1: {valid_iou1:.3f}\nval iou task2: {valid_iou2:.3f}\nval iou task3: {valid_iou3:.3f}')
        #logger.info(f'val iou task3: {val_iou3:.3f}')

        if best_metrics < valid_iou3:
            best_metrics = valid_iou3
            best_epoch = i_epoch
            logger.info('Best Model!')
            torch.save(model.state_dict(), output_dir.joinpath('model.pth'))
            torch.save(optimizer.state_dict(), output_dir.joinpath('opt.pth'))

    else:
        valid_loss = None
        valid_iou = None
        valid_iou1 = None
        valid_iou2 = None
        valid_iou3 = None

    loss_history.append([train_loss_all, valid_loss])

    iou_history.append([train_iou_all, valid_iou])
    iou_history1.append([train_iou1, valid_iou1])
    iou_history2.append([train_iou2, valid_iou2])
    iou_history3.append([train_iou3, valid_iou3])

    history_ploter(loss_history, log_dir.joinpath('loss.png'))
    history_ploter(iou_history, log_dir.joinpath('iou.png'))
    history_ploter(iou_history1, log_dir.joinpath('iou1.png'))
    history_ploter(iou_history2, log_dir.joinpath('iou2.png'))
    history_ploter(iou_history3, log_dir.joinpath('iou3.png'))

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
                 batch_size, resume, freeze_to_layer, freeze_bn, loss_config['loss_type'], opt_config['mode'], opt_config['base_lr'],
                 opt_config['t_max'], best_metrics, best_epoch, i_epoch, train_loss_mean, val_loss_mean, train_iou_mean, val_iou_mean, yyyy_mm_dd])
of_connection.close()