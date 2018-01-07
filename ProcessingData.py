# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:47:44 2016
@author: caoloi
"""
import numpy as np
from sklearn import preprocessing

#%% Compute distance to the origin
def norm_of_each_row(x):
    # Euclidean norm of each row in a 2D array
    return np.array([np.linalg.norm(xi) for xi in x])

#%% Generate uniformly in terms of radius
def uniform_radius(d, n, r):
    U = np.random.uniform(0,r, (n,))
    X = np.random.normal(0, 1, (n, d))
    p = X / norm_of_each_row(X).reshape((-1, 1))
    p1 =U.reshape((-1, 1))*p
    return p1

#%% Save data to arff file for SVM
def Save_To_Arff(XT, X0, X1):
    Norm  = np.full((len(XT)),0,dtype = np.int)
    Norm1 = np.full((len(X0)),0,dtype = np.int)
    Anom  = np.full((len(X1)),0,dtype = np.int)

    XT_L = np.column_stack([XT,Norm])
    np.savetxt("train.arff", XT_L, delimiter=",",fmt='%f')
    X0_L = np.column_stack([X0,Norm1])
    np.savetxt("testX0.arff", X0_L, delimiter=",",fmt='%f')
    X1_L = np.column_stack([X1,Anom])
    np.savetxt("testX1.arff", X1_L, delimiter=",",fmt='%f')

#%% Normalize training and testing sets
def normalize_data(train_X, test_X, scale = "standard"):
    if ((scale == "standard") | (scale == "maxabs") | (scale == "minmax")):
        if (scale == "standard"):
            scaler = preprocessing.StandardScaler()
        elif (scale == "maxabs"):
            scaler = preprocessing.MaxAbsScaler()
        elif (scale == "minmax"):
            scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        test_X  = scaler.transform(test_X)
    else:
        print ("No scaler")
    return train_X, test_X

#%% Process NSL-KDD, training and testing sets are drawn from different distribution
def process_KDD(group_attack):
    #Load normal training data and normal testing data from csv files
    d = np.genfromtxt("NSL-KDD/KDDTrain_Normal.csv", delimiter=",")
    dX0 = np.genfromtxt("NSL-KDD/KDDTest_Normal.csv", delimiter=",")

    if (group_attack =="DoS"):
        dX1 = np.genfromtxt("NSL-KDD/KDDTest_Dos.csv", delimiter=",")
    elif (group_attack =="R2L"):
        dX1 = np.genfromtxt("NSL-KDD/KDDTest_R2L.csv", delimiter=",")
    elif (group_attack =="U2R"):
        dX1 = np.genfromtxt("NSL-KDD/KDDTest_U2R.csv", delimiter=",")
    elif (group_attack =="Probe"):
        dX1 = np.genfromtxt("NSL-KDD/KDDTest_Probe.csv", delimiter=",")

    num_train = 350
    #Train(67343);                    2000
    #KDDTest_Normal(9711);            2000
    #KDDTest_Dos   7458 (5700);       2000
    #KDDTest_R2L   2887(2199);        2000
    #KDDTest_U2R   67 (37)            37
    #KDDTest_Probe 2421 (1106);       1000

    d = d[~np.isnan(d).any(axis=1)]
    np.random.seed(0)
    np.random.shuffle(d)

    # Random Sample 350 examples for train_X
    train_X = d[:num_train]

    np.random.shuffle(dX0)
    np.random.shuffle(dX1)
    test_X0 = dX0      #normal test
    test_X1 = dX1     #anomaly test

    #normal and anomaly test
    test_X = np.concatenate((test_X0, test_X1))

    #Create label for normal and anomaly test examples, and then combine two sets
    test_y0 = np.full((len(test_X0)), False, dtype=bool)    #Normal: False
    test_y1 = np.full((len(test_X1)), True, dtype=bool)     #Anomaly: True
    test_y =  np.concatenate((test_y0, test_y1))

    """We consider normal class and anomaly class are positive class and negative class
	respectively. Thus, we put true label equal 1 for normal and 0 for anomly examples
	in testing set. The roc_curve function will calculate FPR and TPR based on moving
	a threshold from high prediction values to low prediction values"""
    actual = (~test_y).astype(np.int)
	#return training set, testing set (normal and anomaly)
    #and the label set of testing set
    return train_X, test_X, test_y, actual

#%% Process data and return training set, testing set and labels
def process_Data(dataset):
    if (dataset == "WBC"):
        d = np.genfromtxt("UCIData/wobc.csv", delimiter=",")
        label_threshold = 2
        # 9 attributes + 1 class[2 - benign(458); 4 - malignant(241)]
    elif (dataset == "WDBC"):
        d = np.genfromtxt("UCIData/wdbc.csv", delimiter=",")
        label_threshold = 2
        # 30 attributes + 1 class [2 - benign(357); 4 - malignant(212)]
    elif (dataset == "C-heart"):
        d = np.genfromtxt("UCIData/C-heart.csv", delimiter=",")
        label_threshold = 0
        # 13 attributes + 1 class [0 - Level0(164); level 1,2,3,4 - (139), 6 missing)
    elif (dataset == "ACA"):
        d = np.genfromtxt("UCIData/australian.csv", delimiter=",")
        #Australia: 14 feature + 1 class [ 0 - 383, 1 - 307]
        label_threshold = 0

    elif ((dataset =="DoS") or (dataset =="R2L") or (dataset =="U2R") or (dataset == "Probe")):
        train_X, test_X, test_y, actual = process_KDD(dataset)
        return train_X, test_X, test_y, actual

    else:
        print("No dataset is chosen")
        #German: 24 features + 1 class [700 (1-googd), 300(2-bad)]

    "*************************Chosing dataset*********************************"

    # discard the '?' values in wdbc
    d = d[~np.isnan(d).any(axis=1)]

    # shuffle
    np.random.seed(0)
    np.random.shuffle(d)

    dX = d[:,0:-1]      # discard the first column (ids) and the last (labels)
    dy = d[:,-1]

    dy = dy >label_threshold   # dataset 1-(2,4), 2-(2,4), 3-(0,1,2,3,4), 4-(0, 1)

    # separate into normal and anomaly
    dX0 = dX[~dy]   # Normal data
    dX1 = dX[dy]    # Anomaly data
    dy0 = dy[~dy]   # Normal label
    dy1 = dy[dy]    # Anomaly label

    split = 0.7    #split 70% for training and 30% for testing
    idx  = int(split * len(dX0))
    idx1 = int(split* len(dX1))
    # train_X contains only normal examples
    train_X = dX0[:idx]

    # test set is 30% of the normal class and 30% of anomaly class
    test_X = np.concatenate((dX0[idx:], dX1[idx1:]))  # 30% of normal and 30% anomal data
    test_y = np.concatenate((dy0[idx:], dy1[idx1:]))  # 30% of normal and 30% anomal labels

    """We consider normal class and anomaly class are positive class and negative class
	respectively. Thus, we put true label equal 1 for normal and 0 for anomly examples
	in testing set. The roc_curve function will calculate FPR and TPR based on moving
	a threshold from high prediction values to low prediction values"""
    actual = (~test_y).astype(np.int)

    return train_X, test_X, test_y, actual

