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


def get_image_data(filename):
    img = Image.open(filename)
    img = img.getdata()
    #img = img.resize(STANDARD_SIZE)
    #img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def apply_pca(data):
    pca = RandomizedPCA(n_components=10)
    std_scaler = StandardScaler()
    data = pca.fit_transform(data)
    data = std_scaler.fit_transform(data)
    return data

def extract_features(paths):
    data = []
    for f in paths:
        data.append(get_image_data(f))
    features = apply_pca(data)
    return features