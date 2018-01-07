# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 20:39:44 2018

@author: VANLOI
"""
import matplotlib.pyplot as plt
#%%" Plot AUC
def Plotting_AUC(FPR_ae, TPR_ae, auc_ae, FPR_cen, TPR_cen, auc_cen, FPR_kde, TPR_kde, auc_kde, data):
    plt.figure(figsize=(6,6))
    plt.title('The ROC curves - '+ data, fontsize=16)
    plt.plot(FPR_ae, TPR_ae,   'g-^' , label='OCAE     (AUC = %0.3f)'% auc_ae, markevery = 150 , markersize = 6)
    plt.plot(FPR_cen, TPR_cen, 'b-o' ,  label='OCCEN   (AUC = %0.3f)'% auc_cen, markevery = 150 , markersize = 6)
    plt.plot(FPR_kde, TPR_kde, 'r-x' , label='OCKDE    (AUC = %0.3f)'% auc_kde, markevery = 150 , markersize = 6)

    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)

    plt.savefig("Results/Main_Experiment/fig_" + data + "_AUC.pdf")
    plt.show()

#%% Plot AUC against hidden size, using four UCI datasets.
def Plotting_AUC_HZ(AUC_RE, data):
	plt.figure(figsize=(6, 3))
	plt.title("AUC of the three classifiers on " + data, fontsize=14)

	ax = plt.subplot(111)
	plt.plot(AUC_RE[:,0], AUC_RE[:,2], 'g-^' ,label="OCAE")
	plt.plot(AUC_RE[:,0], AUC_RE[:,3], 'b-o' ,label="OCCEN")
	plt.plot(AUC_RE[:,0], AUC_RE[:,4], 'r-x' ,label="OCKDE")


	ax.legend(bbox_to_anchor=(1.01, 0.2), ncol=3)
	#plt.legend(loc='lower right')

	plt.ylim([0.6,1.05])
	plt.xlim([1.8,8.2])
	#plt.xlabel('Hidden size', fontsize=14)
	plt.ylabel('AUC', fontsize=14)
	plt.savefig("Results/Hidden_size/" + "fig_" + data + "_AUC_hiddensize.pdf")
	plt.close

#%% Plot normal and anomaly examples from hidden tesing data
def Plotting_hidden_data(test_h_X0, test_h_X1, data, k):

	plt.figure(figsize=(4, 4))
	plt.title("Hidden data on WBC, k = " + str(k), fontsize=14)
	#plt.title('Generate data from Gaussian distribution', fontsize=14)
	#plt.plot(train_sl[:,0],   train_sl[:,1],   'go')
	ax = plt.subplot(111)
	plt.plot(test_h_X0[:,0], test_h_X0[:,1], 'go' ,label="Normal")
	plt.plot(test_h_X1[:,0], test_h_X1[:,1], 'r^' ,label= "Anomaly")

	ax.legend(bbox_to_anchor=(1.04, 1.04), ncol=2)
	#plt.legend(loc='lower right'

	plt.axis('equal')
	if (k == 0.1):
		plt.ylim((-2.0, 5.0))
		plt.xlim((-2.0, 5.0))
	else:
		plt.ylim((-1.0, 3.5))
		plt.xlim((-1.0, 3.5))
	plt.savefig("Results/Hidden_data/" + "fig_" + data + "_" + str(k) + "_hidden.pdf")
	plt.close
