#TODO: check target_size to set to current resolution

import numpy as np
from PIL import Image
from pathlib import Path
import os
import albumentations as albu

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src.utils import minmax_normalize, meanstd_normalize, topcrop, rescale
    from utils.custum_aug import PadIfNeededRightBottom
except:
    from src.preprocess import minmax_normalize, meanstd_normalize, topcrop, rescale
    from src.utils.custum_aug import PadIfNeededRightBottom

class SherbrookeDataset(Dataset):
    def __init__(self, base_dir='../data/sherbrooke', split='train',
                 affine_augmenter=None, image_augmenter=None, target_size=(544,544),
                 net_type='deeplab', ignore_index=255, defects=False, debug=False):
        if defects is False:
            self.n_classes = 2
            self.void_classes = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                        28, 29, 30, 31, 32, 33, -1]
            self.valid_classes = [0, 8]  # background and sidewalks only. 0 will become background...
        else:
            base_dir = '../data/bbox_mask'
            self.n_classes = 2
            self.void_classes = [0,4] #why 4?
            self.valid_classes = [8, 35]  # background and sidewalks only.
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        self.debug = debug
        self.defects = defects
        self.base_dir = Path(base_dir)
        assert net_type in ['unet', 'deeplab']
        self.net_type = net_type
        self.ignore_index = ignore_index
        self.split = 'val' if split == 'valid' else split

        self.img_paths = sorted(self.base_dir.glob(f'leftImg8bit/{self.split}/*/*leftImg8bit.*'))
        self.lbl_paths = sorted(self.base_dir.glob(f'gtFine/{self.split}/*/*gtFine*.png'))

        #Quality control
        if len(self.img_paths) != len(self.lbl_paths):
            raise AssertionError(f'Length of images (count: {len(self.img_paths)}) '
                                 f'and labels (count: {len(self.lbl_paths)}) don\'t match')
        if len(self.img_paths) == 0:
            raise AssertionError(f'No images found. Check current working directory.')
        count = 0

        for img_path, lbl_path in zip(self.img_paths, self.lbl_paths):
            count += 1
            _, img_path = os.path.split(img_path)
            img_name, img_ext = os.path.splitext(img_path)  # separate name and extension
            _, lbl_path = os.path.split(lbl_path)
            lbl_name, lbl_ext = os.path.splitext(lbl_path)  # separate name and extension
            if img_name.split('_')[0] != lbl_name.split('_')[0]:
                raise AssertionError(f'Image {img_name} and label {lbl_name} don\'t match')
        print(f'Assertion success: image and label filenames in {self.split} split of dataset match.')

        # Resize
        if isinstance(target_size, str):
            target_size = eval(target_size)
        if self.split == 'train':
            if self.net_type == 'deeplab':
                target_size = (target_size[0] + 1, target_size[1] + 1)
            # Resize (Scale & Pad & Crop)
            #self.resizer = None
            self.resizer = albu.Compose([albu.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
                                         #next transform is custom. see src.utils.custom_aug
                                         PadIfNeededRightBottom(min_height=target_size[0], min_width=target_size[1],
                                                                value=0, ignore_index=self.ignore_index, p=1.0),
                                         albu.RandomCrop(height=target_size[0], width=target_size[1], p=1.0)])
            #self.resizer_info = (f'albu.RandomScale(scale_limit={self.resizer.transforms[0].scale_limit}, p=0.5),'
            #                     f'albu.RandomCrop(height={target_size[0]}, width={target_size[1]}, p=1.0)')
        else:
            self.resizer = None

        # Augment
        if self.split == 'train':
            self.affine_augmenter = affine_augmenter
            self.image_augmenter = image_augmenter
        else:
            self.affine_augmenter = None
            self.image_augmenter = None

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        '''Returns preprocessed image and label
        '''

        img_path = self.img_paths[index]
        img = np.array(Image.open(img_path))

        lbl_path = self.lbl_paths[index]
        lbl = np.array(Image.open(lbl_path))

        # if GSV (6656*3328), crop the top half of image and rescale if task is not defect detection
        if img.shape[0] >= 3328 and img.shape[1] >= 6656:
            img = topcrop(img)
            lbl = topcrop(lbl)
            if not self.defects:
                img = rescale(img, max_size=4096)
                lbl = rescale(lbl, max_size=4096)
        elif max(img.shape[0], img.shape[1]) > 2048: #if not GSV, rescale to 2048 if largest side is > 2048
            img = rescale(img, max_size=2048)
            lbl = rescale(lbl, max_size=2048)

        if self.debug:
            print(f'Label path: {lbl_path}')
            print(f'Unique values in label {np.unique(lbl)}')
            # quality control
            valid_values = [0,8,34,35,36,255] #list(self.class_map.values())
            #valid_values.append(self.ignore_index)
            lbl_values = set(np.unique(lbl))
            if not lbl_values.issubset(set(valid_values)):
                print('Oups. There are stranger values in your label...')

        lbl = self.encode_mask(lbl) #overwrite values for to get a 2 class task (sidewalk/background or defect/background)

        # ImageAugment (RandomBrightness, AddNoise...)
        if self.image_augmenter:
            augmented = self.image_augmenter(image=img)
            img = augmented['image']
        # Resize (Scale & Pad & Crop)
        if self.net_type == 'unet':
            img = minmax_normalize(img)
            img = meanstd_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            #ex.: pixel value 144 --> 144/255 if norm range (0,1)
            img = minmax_normalize(img, norm_range=(-1, 1))
        if self.resizer:
            resized = self.resizer(image=img, mask=lbl)
            img, lbl = resized['image'], resized['mask']
        # AffineAugment (Horizontal Flip, Rotate...)
        if self.affine_augmenter:
            augmented = self.affine_augmenter(image=img, mask=lbl)
            img, lbl = augmented['image'], augmented['mask']

        if self.debug:
            print(np.unique(lbl))
            print(lbl.shape)
        else:
            img = img.transpose(2, 0, 1)
            img = torch.FloatTensor(img)
            lbl = torch.LongTensor(lbl)

        return img, lbl

    def encode_mask(self, lbl):
        if self.defects == False:
            # create mask for values other than 0 (background) and 1(sidewalk)
            for i in range(34, 37):
                lbl[lbl == i] = 8  # convert these classes to sidewalk-no-default
        else:
            lbl[lbl == 34] = 8  # convert class 2 to class 1 (sidewalk-no-default)
            lbl[lbl == 36] = 35  # convert class 4 to class 3 (sidewalk-default)
        for c in self.void_classes:
            lbl[lbl == c] = self.ignore_index #assign ignore_index value to all void class values in label
        for c in self.valid_classes: #assign class_map[c] value to c classes in valid_classes
            lbl[lbl == c] = self.class_map[c]
        return lbl

    def crop_to_default(self, img, lbl): #unused, replaced by bbox_mask processing
        where = np.array(np.where(lbl == 1)) #spot pixels where class is sidewalk-default
        if where.size != 0: #if more than 0 pixels of sidewalk-default
            x1, y1 = np.amin(where, axis=1)
            x2, y2 = np.amax(where, axis=1)
            img = img[max(0,x1-256):min(img.shape[0],x2+256), max(0,y1-1024):min(img.shape[1],y2+1024)]
            lbl = lbl[max(0,x1-256):min(lbl.shape[0],x2+256), max(0,y1-1024):min(lbl.shape[1],y2+1024)]
        return img, lbl

    def postprocess(self, pred, lbl):
        '''Returns image and label with original dimensions

        pred (np.array): prediction array
        lbl (np.array): label array
        '''
        assert isinstance(pred,np.ndarray) and isinstance(lbl, np.ndarray)
        # if GSV (6656*3328), add top half of image with 0 values and upscale to
        # original dimensions if task is not defect detection
        pred = topcrop(pred, reverse=True)
        lbl = topcrop(lbl, reverse=True)
        if not self.defects:
            pred_pil = Image.fromarray(pred.astype(np.uint8))
            lbl_pil = Image.fromarray(lbl.astype(np.uint8))
            pred = pred_pil.resize((3328, 6656), Image.NEAREST)
            lbl = lbl_pil.resize((3328, 6656), Image.NEAREST)

        return pred, lbl

    def bbox2(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

if __name__ == '__main__':
    import matplotlib
    import datetime, re

    now = datetime.datetime.now()
    yyyy_mm_dd = now.isoformat().split('T')[0]
    time = now.isoformat().split('T')[1].split('.')[0].strip(':')
    time = re.sub(':', '', time)

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        from utils.custum_aug import Rotate
    except:
        from src.utils.custum_aug import Rotate

    affine_augmenter = albu.Compose([albu.HorizontalFlip(p=.5),
                                     # Rotate(5, p=.5)
                                     ])
    # image_augmenter = albu.Compose([albu.GaussNoise(p=.5),
    #                                 albu.RandomBrightnessContrast(p=.5)])
    image_augmenter = albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=0.5)
    #image_augmenter = None
    # why ignore_index = 19 ?
    dataset = SherbrookeDataset(split='test', net_type='deeplab', debug=True,
                                #dataset = SherbrookeDataset(split='train', net_type='deeplab', ignore_index=255, debug=True,
                                affine_augmenter=affine_augmenter, image_augmenter=image_augmenter, defects= False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    print(len(dataset))

    for i, batched in enumerate(dataloader):
        images, labels = batched
        if i == 0:
            fig, axes = plt.subplots(8, 2, figsize=(20, 48))
            plt.tight_layout()
            for j in range(8):
                axes[j][0].imshow(minmax_normalize(images[j], norm_range=(0, 1), orig_range=(-1, 1)))
                axes[j][1].imshow(labels[j], cmap = 'flag')
                axes[j][0].set_xticks([])
                axes[j][0].set_yticks([])
                axes[j][1].set_xticks([])
                axes[j][1].set_yticks([])
            plt.savefig(f'dataset/sherbrooke_{time}.png')
            plt.close()
        break
