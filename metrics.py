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

import pylab as pl
from pylab import *
from sklearn.metrics import confusion_matrix, classification_report


def draw_confusion_matrix(cm_normal, cm_adapted_for_color, labels_names, ax, title):
    """
    Draw the given confusion matrix in the given subplot
    :param cm_normal:               the original confusion matrix
    :param cm_adapted_for_color:    the eventuel adapted confusion matrix (ex: cm_adapted_for_color = sqrt(cm))
    :param labels_names:            all available classes
    :param ax:                      the given subplot on which the chart will be drawn
    :param title:                   the title for this subplot
    """

    cmim = ax.matshow(cm_adapted_for_color, interpolation='nearest')

    for i in xrange(cm_normal.shape[0]):
        for j in xrange(cm_normal.shape[1]):
            value = cm_normal[i, j]

            # Color
            # green for diagonal, red otherwise
            color = 'red'
            if i == j:
                color = 'green'
            elif value == 0:
                continue

            # Add text
            ax.annotate(str(value), xy=(j, i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8,
                        color=color)
    ax.set_xticks(np.arange(cm_normal.shape[0]))
    ax.set_xticklabels([labels_names[l] for l in xrange(cm_normal.shape[0])], rotation='vertical')
    ax.set_yticks(np.arange(cm_normal.shape[1]))
    _ = ax.set_yticklabels([labels_names[l] for l in xrange(cm_normal.shape[1])])
    ax.set_xlabel('predicted label')
    ax.set_ylabel('true label')
    ax.set_title(title)
    pl.colorbar(cmim, shrink=0.5)


def plot_confusion_matrix(y_true, y_predicted, labels_names):
    """Utility function to plot a confusion matrix"""

    cm = confusion_matrix(y_predicted, y_true)

    cm_log = cm.astype('float')
    for i in xrange(cm_log.shape[0]):
        for j in xrange(cm_log.shape[1]):
            if cm_log[i, j] > 0:
                cm_log[i, j] = sqrt(cm_log[i, j])

    cm_norm = cm.astype('float')
    for i in xrange(cm_norm.shape[0]):
        a = 0
        for j in xrange(cm_norm.shape[1]):
            a += cm_norm[i, j]
        for j in xrange(cm_norm.shape[1]):
            if a == 0:
                cm_norm[i, j] = 0
            else:
                cm_norm[i, j] = float(cm_norm[i, j]) / float(a)

    # display
    axOriginal = pl.subplot(131)
    draw_confusion_matrix(cm, cm, labels_names, axOriginal, 'Normal')

    axLog = pl.subplot(132)
    draw_confusion_matrix(cm, cm_log, labels_names, axLog, 'SQRT')

    axNormalized = pl.subplot(133)
    draw_confusion_matrix(cm, cm_norm, labels_names, axNormalized, 'Normalized per number')

    pl.show()


def print_classification_report(y_true, y_pred, title=''):
    if title != '':
        print title
    print(classification_report(y_true, y_pred))
