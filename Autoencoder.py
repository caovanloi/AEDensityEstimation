# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:29:11 2016

@author: caoloi
"""
import theano as th
from theano import tensor as T
from numpy import random as rng
import numpy as np
from sklearn.base import BaseEstimator
from collections import OrderedDict
#%% Define activation functions
#def logistic(x):
#    return 1.0/(1+T.exp(-x))
#
#def logistic_prime(x):
#    ex=T.exp(-x)
#    return ex/(1+ex)**2

def identity(x):
    return x

#def identity_prime(x):
#    return 1

#def tansig(x):
#    ex = T.exp(-2*x)
#    return ((1-ex)/(1+ex))

#%% define Autoencoder Class
class AutoEncoder(BaseEstimator):

    def __init__(self,
                 input_size,
                 hidden_size = 6,
                 n_epochs=100,
                 mini_batch_size=1,
                 learning_rate=0.1,
                 K = 1.0):

        #Steepness parameter k for logistic function Sigmoid_ex
        self.K = K
        #Input_size is the same number of the input dimension, int.
        self.input_size = input_size
        #Hidden_size is the number of neurons in the hidden layer, int.
        assert type(hidden_size) is int
        assert hidden_size > 0
        self.hidden_size = hidden_size

        #Create random seed for randomly generate Weight matrix
        rng.seed(0)
        initial_W = np.asarray(rng.uniform(
                 low=-4 * np.sqrt(6. / (self.hidden_size + self.input_size)),
                 high=4 * np.sqrt(6. / (self.hidden_size + self.input_size)),
                 size=(self.input_size, self.hidden_size)), dtype=th.config.floatX)

        self.W = th.shared(value=initial_W, name='W', borrow=True)
        self.b1 = th.shared(name='b1', value=np.zeros(shape=(self.hidden_size,),
                            dtype=th.config.floatX),borrow=True)
        self.b2 = th.shared(name='b2', value=np.zeros(shape=(self.input_size,),
                            dtype=th.config.floatX),borrow=True)

        self.n_epochs = n_epochs
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate

    #%% Sigmoid function with steepness parameter K
    def sigmoid_ex(self,x):
        return 1.0/(1.0 + T.exp(-self.K*x))
    #%% Fit data, define cost function and compute grad
    def fit(self, X):
        self.activation_function = self.sigmoid_ex   #hidden activation = sigmoid
        self.output_function = identity              #output activation = identity

        #X is an numpy matrix, rows and cols correspond to datapoints and features.
        #assert type(X) is np.ndarray
        assert len(X.shape)==2
        self.X = X                #Training set
        self.X = th.shared(name='X', value=np.asarray(self.X,
                         dtype=th.config.floatX),borrow=True)
        # Wrap data in theano variable which is to get this to run fast on the gpu.
        self.n = X.shape[1]    #Dimension
        self.m = X.shape[0]    #Size of data

        "****************** Compute hidden and output data *******************"
        index = T.lscalar()
        x=T.matrix('x')
        params = [self.W, self.b1, self.b2]  #List of parameters
        #Hidden data
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        #Reconstruction data at output layer
        output = self.output_function(T.dot(hidden, T.transpose(self.W))+self.b2)

        "********************** Define cost function *************************"
        "Use cross-entropy loss, suitable for binary data"
        #L = -T.sum(x*T.log(output) + (1-x)*T.log(1-output), axis=1)
        #cost=T.mean(L)

        "MEAN-MSE is used in this work, for real-value data "
        cost = (((x - output)**2).mean(1)).mean()

        #"1.RMSE-MAE:  RMSE of whole dataset based on MAE of each sample"
        #cost = T.sqrt(((T.abs_(x-output).mean(1))**2).mean())

        #"2.RMSE-RMSE: RMSE of whole dataset based on RMSE of each sample"
        #cost = T.sqrt(T.mean((x - output)**2))

        #"3.MEAN-RMSE: Mean of whole dataset based on RMSE of each example"
        #cost = (T.sqrt(((x - output)**2).mean(1))).mean()

        #"4. MEAN -MAE"
        #cost = (T.abs_(x-output).mean(1)).mean()

        "************************** Update cost ******************************"
        updates=[]
        #Return gradient with respect to W, b1, b2.
        gparams = T.grad(cost,params)

        #SGD
#        for param, gparam in zip(params, gparams):
#            updates.append((param, param-self.learning_rate*gparam))
        "============== ADAGRAD is used in this work ============="
        eps  = 1e-8
        accugrads = [th.shared(name='accugrad', value=np.zeros(shape=param.shape.eval(),
                            dtype=th.config.floatX),borrow=True) for param in params]
        # compute list of weights updates
        updates = OrderedDict()
        for accugrad, param, gparam in zip(accugrads, params, gparams):
            # c.f. Algorithm 1 in the Adadelta paper (Zeiler 2012)
            agrad = accugrad + gparam * gparam
            updates[param] = param - (self.learning_rate /(T.sqrt(agrad) + eps)) * gparam
            updates[accugrad] = agrad
        "========================================================="
        fit = th.function(inputs=[index],
                          outputs=[cost],
                          updates=updates,
                          givens={x:self.X[index:index+self.mini_batch_size,:]})

        for epoch in range(self.n_epochs):
            for row in range(0,self.m, self.mini_batch_size):
                fit(row)

        """it may return cost of the last mini_batch in the last epoch"""
        return cost

    #%% Get data from hidden layer
    def get_hidden(self,data):
        x=T.dmatrix('x')
        hidden = self.activation_function(T.dot(x,self.W)+self.b1)
        transformed_data = th.function(inputs=[x], outputs=[hidden])
        return transformed_data(data)
    #%% Get data from the output layer
    def get_output(self,data):
        x = T.dmatrix('x')
        hidden = self.activation_function(T.dot(x, self.W)+self.b1)
        output = self.output_function(T.dot(hidden,T.transpose(self.W))+self.b2)
        transformed_data = th.function(inputs = [x], outputs = [output])
        return transformed_data(data)
    #%% Get parameters
    def get_weights(self):
        return [self.W.get_value(), self.b1.get_value(), self.b2.get_value()]

    # This function is used to do gridsearch for AE
    def score(self, X, y=None):
        output =  self.get_output(X)
        "MEAN-MSE"
        RE = (((X - output)**2).mean(1)).mean()
        return  (1/(RE+1))




























