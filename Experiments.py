# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:31:20 2016
@author: caoloi
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
from Autoencoder import AutoEncoder
from ProcessingData import process_Data, normalize_data
from BaseOneClass import CentroidBasedOneClassClassifier, DensityBasedOneClassClassifier
from Plot_curve import Plotting_AUC, Plotting_AUC_HZ, Plotting_hidden_data

#%%
def Call_Autoencoder(data, train_X, test_X, actual,
					 h_size, epoch, k_log, l_rate, bw, exp):

    #******************* Call autoencoder **********************
    i_size = train_X.shape[1]
    AE = AutoEncoder(input_size      = i_size,
                     hidden_size     = h_size,
                     n_epochs        = epoch,
                     learning_rate   = l_rate,
                     K               = k_log)

    AE.fit(train_X)
	#get reconstruction of train_X and test_X, get hidden data of train_X and test_X
    output_train  = AE.get_output(train_X)
    output_test  = AE.get_output(test_X)
    train_hidden = AE.get_hidden(train_X)
    test_hidden  = AE.get_hidden(test_X)

    #*************** RE-based one-class classifier *************"
    RE_MSE = (((test_X - output_test[0])**2).mean(1))
    """RE (MSE) between output and input is used as anomalous score.
	We put minus "-" to RE_MSE to for computing FPR and TPR using roc_curve"""
    predictions_auto = -RE_MSE
    FPR_ae, TPR_ae, thresholds_auto = roc_curve(actual, predictions_auto)
    auc_ae = auc(FPR_ae, TPR_ae)

    #***************** Centroid on hidden data *****************"
    CEN = CentroidBasedOneClassClassifier()
    CEN.fit(train_hidden[0])
    predictions_cen = -CEN.get_density(test_hidden[0])
    FPR_cen, TPR_cen, thresholds_cen = roc_curve(actual, predictions_cen)
    auc_cen = auc(FPR_cen, TPR_cen)

    #****************** KDE on hidden layer *****************"
    KDE = DensityBasedOneClassClassifier(bandwidth = bw, kernel="gaussian", metric="euclidean")
    KDE.fit(train_hidden[0])
    predictions_kde = KDE.get_density(test_hidden[0])
    FPR_kde, TPR_kde, thresholds_kde = roc_curve(actual, predictions_kde)
    auc_kde = auc(FPR_kde, TPR_kde)

    #************ RE (MEAN-MSE) on training set *************"
    RE  = (((train_X - output_train)**2).mean(1)).mean()

    if (exp == "ME"):         #main experiment
        Plotting_AUC(FPR_ae,  TPR_ae,  auc_ae,
				 FPR_cen, TPR_cen, auc_cen,
				 FPR_kde, TPR_kde, auc_kde, data)
    elif (exp == "HD"):      #investigate hidden size
        #Save hidden data to csv file"
        np.savetxt("Results/Hidden_data/" + data + "_train_" + str(k_log) + ".csv", train_hidden[0], delimiter=",",fmt='%f')
        np.savetxt("Results/Hidden_data/" + data + "_test_"  + str(k_log) + ".csv", test_hidden[0],  delimiter=",",fmt='%f')
        return train_hidden[0], test_hidden[0]

    return auc_ae, auc_cen, auc_kde, RE

#%% Preliminary Experiment for investigating hidden size
def Investigate_hidden_size(data):
    train_X, test_X, test_y, actual= process_Data(data)  # 4 datasets in UCI, and NSL-KDD
    train_X, test_X = normalize_data(train_X, test_X)

    epoch   = 5
    h_size  = [2,3,4,5,6,7,8]
    k       = 1.0
    lr      = 0.01

    AUC_RE = np.empty([0,6])
    print("\nDataset:" + data)
    for i in range(0, len(h_size)):
        bw = (h_size[i]/2.0)**0.5
        print ("Hidden_siz: ", h_size[i], " bw: ", bw)

        ae, cen, kde, re = Call_Autoencoder(data, train_X, test_X, actual,
									    h_size[i], epoch, k, lr, bw, "HZ")
        temp = np.column_stack([h_size[i], epoch, ae, cen, kde, re])
        AUC_RE = np.append(AUC_RE, temp)

    AUC_RE = np.reshape(AUC_RE, (len(h_size), 6))
    print(AUC_RE)
    np.savetxt("Results/Hidden_size/" + data + "_hidden_size.csv",
			   AUC_RE, delimiter=",",fmt='%f')
    Plotting_AUC_HZ(AUC_RE, data)

#%% Preliminary Experiment to visualize hidden data
def Visualize_hidden_data(data):
	train_X, test_X, test_y, actual= process_Data(data)  # 4 datasets in UCI, and NSL-KDD
	train_X, test_X = normalize_data(train_X, test_X)

	epoch   = 5
	h_size  = 2
	k       = [0.1, 0.5, 1.0]
	lr      = 0.01

    #Default in One-class SVM
	bw = (h_size/2.0)**0.5
	print("Hidden_siz: ", h_size, " bw: ", bw)

	for i in range(0, len(k)):
		train_hidden, test_hidden = Call_Autoencoder(data, train_X, test_X, actual,
									    h_size, epoch, k[i], lr, bw, "HD")
		_, test_h = normalize_data(train_hidden, test_hidden)
		test_h_X0 = test_h[actual==1]
		test_h_X1 = test_h[actual==0]
		Plotting_hidden_data(test_h_X0, test_h_X1, data, k[i])

#%% Main experiment on NSL-KDD dataset
def Main_Experiment(data):
	train_X, test_X, test_y, actual= process_Data(data)  # 4 datasets in UCI, and NSL-KDD
	train_X, test_X = normalize_data(train_X, test_X)

	epoch   = 5
	h_size  = 7
	k       = 1.0
	lr      = 0.01
	bw = (h_size/2.0)**0.5   #Default in One-class SVM
	ae, cen, kde, re = Call_Autoencoder(data, train_X, test_X, actual,
									    h_size, epoch, k, lr, bw, "ME")

	print("********************************************************")
	print("Data: %s \nNormal train: %d \nNormal test: %d \nAnomaly test: %d"
	   %(data, len(train_X), len(test_X[actual==1]), len(test_X[actual == 0])))
	print("Hidden_size: %d \nBandwidth: %f \nLearning rate: %0.3f \nEpochs: %d"
		   %(h_size, bw, lr, epoch))
	print("Training error:%0.4f" %re)


#%%
list_uci    = ["WBC", "WDBC", "C-heart", "ACA"]
list_nslkdd = ["DoS", "R2L", "U2R", "Probe"]
if __name__ == '__main__':
	#Visualize hidden data in 2D on WBC
	Visualize_hidden_data("WBC")
	#Investigate hidden size on four UCI datasets
	for data in list_uci:
		Investigate_hidden_size(data)
	#Evaluate the proposed model on NSL-KDD	dataset
	for data in list_nslkdd:
		Main_Experiment(data)
