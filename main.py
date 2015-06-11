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

import os
import numpy as np
from PIL import Image
from metrics import plot_confusion_matrix, print_classification_report
from sklearn.metrics.metrics import f1_score
from data_loader import load_numbers
from sklearn.cross_validation import train_test_split
from svm import load_or_train
from feature_extraction import extract_features

## Vars
force_train = True
enable_plot = False

## Load and split the data in train and test sets
print "Loading data..."
X, y = load_numbers()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## Get trained classifier
print "Training classifier..."
clf = load_or_train(X_train, y_train, force_train, enable_plot)

## Compute the features of the test set and predict
print "Predicting test set..."
features = extract_features(X_test)
y_pred = clf.predict(features)

## Score
f1 = f1_score(y_test, y_pred)
print "f1-score for is {}%".format(f1)
if enable_plot:
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred, range(0, 10))
