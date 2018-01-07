# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:06:19 2016

@author: caoloi
"""
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import scipy.spatial
import numpy as np

#%% CENTROID
class CentroidBasedOneClassClassifier:
    def __init__(self, threshold = 0.95, metric="euclidean"):
        self.threshold = threshold
        self.scaler = preprocessing.StandardScaler()
        self.metric = metric

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        # because we are using StandardScaler, the centroid is a
        # vector of zeros, but we save it in shape (1, n) to allow
        # cdist to work happily later.
        self.centroid = np.zeros((1, X.shape[1]))
        # transform relative threshold (eg 95%) to absolute
        dists = self.get_density(X, scale=False) # no need to scale again
        self.abs_threshold = np.percentile(dists, 100 * self.threshold)

    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        dists = scipy.spatial.distance.cdist(X, self.centroid, metric=self.metric)
        dists = np.mean(dists, axis=1)
        return dists

    def predict(self, X):
        dists = self.get_density(X)
        return dists > self.abs_threshold

#%% NEGATIVE MEAN DISTANCE
class NegativeMeanDistance:
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def fit(self, X):
        self.X = X

    def score_samples(self, X):
        dists = scipy.spatial.distance.cdist(X, self.X, metric=self.metric)
        return -np.mean(dists, axis=1)

#%% KERNEL DENSITY ESTIMATION
class DensityBasedOneClassClassifier:
    def __init__(self, threshold=0.95, kernel="gaussian", bandwidth=1.0,
                 metric="euclidean",
                 should_downsample=False, downsample_count=1000):

        self.should_downsample = should_downsample
        self.downsample_count = downsample_count
        self.threshold = threshold
        self.scaler = preprocessing.StandardScaler()
        if kernel == "really_linear":
            self.dens = NegativeMeanDistance(metric=metric)
        else:
            self.dens = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric=metric)

    def fit(self, X):
        # scale
        self.scaler.fit(X)
        self.X = self.scaler.transform(X)
        # downsample?
        if self.should_downsample:
            self.X = self.downsample(self.X, self.downsample_count)
        # fit
        self.dens.fit(self.X)
        # transform relative threshold (eg 95%) to absolute
        dens = self.get_density(self.X, scale=False) # no need to scale again
        self.abs_threshold = np.percentile(dens, 100 * (1 - self.threshold))

    def get_density(self, X, scale=True):
        if scale:
            X = self.scaler.transform(X)
        # in negative log-prob (for KDE), in negative distance (for NegativeMeanDistance)
        return self.dens.score_samples(X)

    def predict(self, X):
        dens = self.get_density(X)
        return dens < self.abs_threshold # in both KDE and NMD, lower values are more anomalous

    def downsample(self, X, n):
        # we've already fit()ted, but we're worried that our X is so
        # large our classifier will be too slow in practice. we can
        # downsample by running a kde on X and sampling from it (this
        # will be slow, but happens only once), and then using those
        # points as the new X.
        if len(X) < n:
            return X
        kde = KernelDensity()
        kde.fit(X)
        return kde.sample(n)