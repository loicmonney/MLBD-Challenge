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
import matplotlib.pyplot as plt
import os
import pickle
from metrics import plot_confusion_matrix, print_classification_report
from sklearn import svm, cross_validation
from feature_extraction import extract_features
from sklearn.learning_curve import learning_curve


def train(clf, X_train, y_train):
    """ Train and return an SVM classifier """

    clf.fit(X_train, y_train)
    return clf


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def load_or_train(X, y, force_train=False, enable_plot=False):
    """
    Load an existing one or train a new SVM classifier, and return it.
    Once the classifier is trained, it is saved through pickle.
    """

    clf_path = './clf.pkl'

    if not force_train and os.path.exists(clf_path):
        print "> loading from file..."
        clf = pickle.load(open(clf_path, 'rb'))
        print "> loaded"
    else:
        print "> training..."

        X = extract_features(X)

        ## Instantiate a classifier
        clf = svm.SVC(kernel='linear')

        ## Cross-validation
        from sklearn.cross_validation import ShuffleSplit

        cv = ShuffleSplit(len(X), n_iter=10, test_size=0.2, random_state=0)

        from sklearn.grid_search import GridSearchCV
        import numpy as np

        gammas = np.logspace(-6, -1, 10)
        estimator = svm.SVC(gamma=0.001)
        classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas), n_jobs=6, verbose=2)
        classifier.fit(X, y)

        title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' % classifier.best_estimator_.gamma
        estimator = svm.SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
        plot_learning_curve(estimator, title, X, y, cv=cv)
        plt.show()

        ## Cross validation
        # kfolds = 10
        # scores = cross_validation.cross_val_score(clf, X, y, cv=kfolds)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        ## Train the classifier on the whole data set, and save it
        clf = train(clf, X, y)

        ## show information about the training
        if enable_plot:
            outputs = clf.predict(X)
            plot_confusion_matrix(y, outputs, range(0, 10))
            print_classification_report(y, outputs)

        ## Save the model
        pickle.dump(clf, open(clf_path, 'wb'))

    return clf
