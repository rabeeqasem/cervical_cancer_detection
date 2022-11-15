import GWO as gwo
import csv
import numpy
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import svm


# algo: 0, ["FN1", -1, 1], 20, 20, 
def selector(algo,func_details,popSize,Iter,completeData):
    function_name=func_details[0]
    lb=func_details[1] # -1
    ub=func_details[2] # 1
   
    
    DatasetSplitRatio=0.3  #Training 70%, Testing 30%
    
    DataFile=completeData
      
    data_set=numpy.loadtxt(open(DataFile,"rb"),delimiter=",",skiprows=0)
    numRowsData=numpy.shape(data_set)[0]    # number of instances in the  dataset
    numFeaturesData=numpy.shape(data_set)[1]-1 #number of features in the  dataset

    dataInput=data_set[0:numRowsData,0:-1] # 1: for class in first column, 0:-1 for class in last column
    dataTarget=data_set[0:numRowsData,-1]  # -1 for class in last column, 0 for class in first column
    trainInput, testInput, trainOutput, testOutput = train_test_split(dataInput, dataTarget, 
                                                        test_size=DatasetSplitRatio, random_state=1)#, stratify=dataTarget)

    dim=numFeaturesData
    
    x=gwo.GWO(getattr(fitnessFUNs, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput)
            
    reducedfeatures=[]
    for index in range(0,dim):
        if (x.bestIndividual[index]==1):
            reducedfeatures.append(index)
    reduced_data_train_global=trainInput[:,reducedfeatures]
    reduced_data_test_global=testInput[:,reducedfeatures]
    # reduced_data_validation_global=validationInput[:,reducedfeatures]
    
    #svc=svm.SVC(kernel='rbf').fit(reduced_data_train_global,trainOutput)
    svc=svm.SVC(kernel='rbf').fit(trainInput,trainOutput)

    # Compute the accuracy of the prediction
    #red_data_dict = {'train': reduced_data_train_global, 'test': reduced_data_test_global}
    red_data_dict = {'train': trainInput, 'test': testInput}
    labels_dict = {'train': trainOutput, 'test': testOutput}

    scores_dict = {'train':{'acc': 0., 'prec': {'macro': 0., 'micro': 0.}, 'rec': {'macro': 0., 'micro': 0.}, 'f1': {'macro': 0., 'micro': 0.}},
                   'test':{'acc': 0., 'prec': {'macro': 0., 'micro': 0.}, 'rec': {'macro': 0., 'micro': 0.}, 'f1': {'macro': 0., 'micro': 0.}},
                  }
    methods_dict = {'acc': accuracy_score, 'prec': precision_score, 'rec': recall_score, 'f1': f1_score}

    for dat in red_data_dict.keys():
        tgt_pred = svc.predict(red_data_dict[dat])
        mets = scores_dict[dat]
        for m in mets:
            if m == 'acc':
                scores_dict[dat][m] = float(methods_dict[m](labels_dict[dat], tgt_pred))
            else:
                for sub_m in scores_dict[dat][m].keys():
                    scores_dict[dat][m][sub_m] = float(methods_dict[m](labels_dict[dat], tgt_pred,average = sub_m))

    x.scores_dict = scores_dict
    return x
    
#####################################################################    
