from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import glob, os
#from get_dataset_colormap import label_to_color_image
from pathlib import Path

def vis_segmentation(image, seg_map, save_file):
  """Visualizes input image, segmentation map and overlay view."""
  background = image
  foreground = seg_map

  background.paste(foreground, (0, 0), foreground)
  background.save(save_file)

#EXECUTION BLOCK

#path to segmentation predictions with values 0-18
#path = Path('..\data\output\lindau')
split = 'test'
base_dir=Path(f'../../data/output/sherbrooke_deeplabv3p_lgpu8/{split}')
images = sorted(base_dir.glob(f'../../../sherbrooke/leftImg8bit/{split}/*/*'))
preds = sorted(base_dir.glob(f'*/colormap_color/*color.png'))

print ("Current working directory is: %s" % (os.getcwd()))

#loop through all files with "filetype" inside file name
count = 0

for img, pred in zip(images, preds):
  count+=1
  if count % 10 == 0:
    print(f'Figure creation progress: {count}/{len(images)}')
  _tmp_, filename = os.path.split(img)
  file, ext = os.path.splitext(filename)  # separate name and extension
  img_pil = Image.open(img)  # open image with pillow
  pred_pil = Image.open(pred)

  if pred_pil.size != img_pil.size:
    pred_pil = pred_pil.resize((img_pil.size[0],img_pil.size[1]), Image.NEAREST)

  folder = pred.parent.parent.joinpath('figures')
  folder.mkdir(parents=True, exist_ok=True)
  vis_segmentation(img_pil, pred_pil, folder.joinpath(f'{file}_fig.jpg'))
  #if count==4:
  #  break

