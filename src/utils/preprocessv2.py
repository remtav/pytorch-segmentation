import numpy as np
import cv2
from PIL import Image

def topcrop(img_array, topcrop_prop = 0.5):
    img = Image.fromarray(img_array)
    # keep only region with sidewalk
    top_crop = int(img.size[1] * topcrop_prop)
    #bottom_crop = int(img.size[1] * 0.96)
    img = img.crop((0, top_crop, img.size[0], img.size[1]))#, bottom_crop))
    preproc_img = np.array(img)
    return preproc_img

def rescale(img_array, max_size=4096):
    img = Image.fromarray(img_array)
    if (img.size[0] > img.size[1]):
        resized_width = max_size
        # resize largest side to basewidth, keeping same aspect ratio
        wpercent = (resized_width / float(img.size[0]))
        resized_height = int(round((float(img.size[1]) * float(wpercent))))
    else:
        resized_height = max_size
        # resize largest side to basewidth, keeping same aspect ratio
        wpercent = (resized_height / float(img.size[1]))
        resized_height = int(round((float(img.size[0]) * float(wpercent))))
    if img.mode != 'L':
        resc_img = img.resize((resized_width, resized_height), Image.ANTIALIAS)
    else:
        resc_img = img.resize((resized_width, resized_height), Image.NEAREST)
    preproc_img = np.array(resc_img)
    return preproc_img

def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img


def meanstd_normalize(img, mean, std):
    mean = np.asarray(mean)
    std = np.asarray(std)
    norm_img = (img - mean) / std
    return norm_img


def padding(img, pad, constant_values=0):
    pad_img = np.pad(img, pad, 'constant', constant_values=constant_values)
    return pad_img


def clahe(img, clip=2, grid=8):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    img_yuv[:, :, 0] = _clahe.apply(img_yuv[:, :, 0])
    img_equ = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2BGR)
    return img_equ
