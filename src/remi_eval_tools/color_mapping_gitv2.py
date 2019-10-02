# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Visualizes the segmentation results via specified color map.
Visualizes the semantic segmentation results by the color map
defined by the different datasets. Supported colormaps are:
* ADE20K (http://groups.csail.mit.edu/vision/datasets/ADE20K/).
* Cityscapes dataset (https://www.cityscapes-dataset.com).
* Mapillary Vistas (https://research.mapillary.com).
* PASCAL VOC 2012 (http://host.robots.ox.ac.uk/pascal/VOC/).
"""

from PIL import Image
import glob, os
import numpy as np
from pathlib import Path

from remi_eval_tools.get_dataset_colormap import label_to_color_image

def save_colormap(img_np, filename, output_dir, filetype, colortype='color'):
    """Converts a 2D numpy array with corresponding colormap
    Args:
      img_np: image as numpy array to be converted and saved
      filename: filename of original image. Saved image will be [filename]_color.png
      output_dir: output directory where images will be saved
      colortype: colormap to use when converting. 'color' or 'gray'. Default: 'color'
    Returns:
      Saved image converted to corresponding colormap in specified output directory.
      Consider returning only PIL Image for further modification...
    """
    #path to segmentation predictions with values 0-18
    output_folder = f'colormap_{colortype}'
    #output_dir = input_dir / output_folder
    output_dir = output_dir / output_folder

    output_dir.mkdir(exist_ok=True)

    #print(f'Files containing \'{filetype}\' will be converted to \'{colortype}\' colormap and saved to:\n{output_dir}')

    colormap = label_to_color_image(img_np, dataset='cityscapes', type=colortype)  # generate colormap with given args
    if colortype == 'gray':
        Image.fromarray(colormap.astype(np.uint8)).save(
            output_dir.joinpath(f'{filename}_id.png'))  # save np.array back as pil image, in file
    else:
        img = Image.fromarray(colormap.astype(np.uint8))
        img = img.convert("RGBA")

        pixdata = img.load()

        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (255, 255, 255, 255):
                    pixdata[x, y] = (255, 255, 255, 0)
                else:
                    pixdata[x, y] = (pixdata[x, y][0], pixdata[x, y][1], pixdata[x, y][2], int(255*0.3))

        img.save(output_dir.joinpath(f'{filename}_color.png'))
    # break

def color_mapping(input_dir, filetype='pred', colortype='color'): #no tested yet.
    '''
    Takes 'input_dir' as input which will be globbed with particular filetype (ex.: pred)
    Output: colormap file saved in "filetype"_"colortype" folder
    Args:
         colortype: color or gray (for evaluation with official cityscapes scripts)
         filetype: name contained in files to be colormapped. In this case, 'pred' or 'lbl'.
    '''

    #loop through all files with "filetype" inside file name
    count = 0
    for infile in glob.glob(f'{input_dir}/*{filetype}*.png'):
        count +=1
        if count % 10 == 0:
            print(f'Conversion progress: {count}')
        tmp, full_filename = os.path.split(infile)  # separate path and filename
        filename, ext = os.path.splitext(full_filename)  # separate name and extension
        #print(file)
        img = Image.open(infile) #open image with pillow
        img_np=np.array(img) #open image as np array
        save_colormap(img_np, filename, input_dir, filetype, colortype)

if __name__ == '__main__':

    input_path = Path('..\data\output\\2019-03-08')
    color_mapping(input_path, colortype='color', filetype='pred')

