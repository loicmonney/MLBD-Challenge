# coding: utf8

#######################################################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015 Loïc Monney <loic.monney@master.hes-so.ch>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#######################################################################################################################

import numpy as np
from PIL import Image
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from PIL import Image, ImageChops


def get_image_data(filename):
    img = Image.open(filename)
    # img = img.getdata()
    # img = img.resize(STANDARD_SIZE)
    # img = map(list, img)
    img = img.convert('L')
    return img
    # s = img.shape[0] * img.shape[1]
    # img_wide = img.reshape(1, s)
    # return img_wide[0]


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def center_on_white(im, w, h):
    trimmed = trim(im)

    ## create a new WxH pixel image surface
    ## make the background white (default bg=black)
    bg = Image.new("RGB", [w, h], (255, 255, 255))

    ## Paste de trimmed image in the middle of background image
    x = (bg.size[0] - trimmed.size[0]) / 2
    y = (bg.size[1] - trimmed.size[1]) / 2
    bg.paste(trimmed, (x, y))

    return bg


# pca = RandomizedPCA(n_components=10)
# std_scaler = StandardScaler()
#
#
# def apply_pca(data):
#    flat = []
#    for d in data:
#        flat += d.flatten()
#    data = pca.fit_transform(data)
#    data = std_scaler.fit_transform(data)
#    return data


def extract_features(paths):
    all_features = []
    for f in paths:
        im = get_image_data(f)
        im = center_on_white(im, 256, 256)
        data = np.array(im.convert('L'))
        features = []
        fd = hog(data, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.extend(fd)
        # features = apply_pca(data)
        all_features.append(features)

    return all_features
