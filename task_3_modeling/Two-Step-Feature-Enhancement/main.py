import csv
import numpy
import time
import selector as slctr
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument('--export', type=bool, default = True, help='Export results to a csv file? True/False')
parser.add_argument('--num_csv', type=int, required = True, help='How many csv files of features do you have?')
parser.add_argument('--num_runs', type=int, default = 30, help='How many independent runs do you want?')
parser.add_argument('--pop_size', type=int, default = 20, help='Number of individuals in a population of grey wolves.')
parser.add_argument('--num_iter', type=int, default = 20, help='Number of iterations of GWO algorithm.')
args = parser.parse_args()

csv_list = []
for i in range(args.num_csv):
        csv_list.append(str(input("Enter name of csv number %d: "%(i+1))))

def join_csv(csv_list,dset_name):
        for num,i in enumerate(csv_list):
                if '.csv' not in i:
                        i=i+'.csv'
                if num==0:
                    df = np.asarray(pd.read_csv(i,header=None))
                    target = df[:,0] # target is the first column
                    df = df[:,1:] # features are from 2nd column onwards
                    target = np.expand_dims(target, axis=1)
                    target = target.astype(int)
                    print("target", target.shape, target.dtype)
                    print("arr", df.shape)
                else:
                    df2 = np.asarray(pd.read_csv(i,header=None))
                    df2 = df2[:,1:] # target is in the first column
                    df = np.concatenate((df,df2),axis=1)
                    print("Concatenated shape", df.shape)
        
        scaler = StandardScaler()
        df = scaler.fit_transform(df)

        pca = PCA(0.99)
        fit = pca.fit(df)
        df = fit.transform(df)
        print(df.shape)

        df = np.hstack([df, target])
        print(df.shape)
        np.savetxt(dset_name+".csv", df, delimiter=",")
        return dset_name

datasets=[join_csv(csv_list,dset_name='final_feat')]
# datasets=['final_feat']
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=args.num_runs

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = args.pop_size
Iterations= args.num_iter

#Export results ?
Export = args.export

#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated file name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 


# CSV Header for for the convergence 
CnvgHeader1=[]
CnvgHeader2=[]
Flag = False

for l in range(0,Iterations):
	CnvgHeader1.append("Iter"+str(l+1))

for l in range(0,Iterations):
	CnvgHeader2.append("Iter"+str(l+1))


for k in range (0,NumOfRuns): # run it 30 times
        print("Run #", k)
        func_details=fitnessFUNs.getFunctionDetails(0)
        print("FUN DETAILS", func_details)
        completeData=datasets[0]+".csv"
        # parameters are: 
        x=slctr.selector(0,func_details,PopulationSize,Iterations,completeData)
          
        if(Export==True):
            with open(ExportToFile, 'a',newline='\n') as out:
                writer = csv.writer(out,delimiter=',')
                if (Flag==False): # just one time to write the header of the CSV file
                    header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime",
                        "trainAcc","testAcc",
                        "trainPrec Micro","testPrec Micro",
                        "trainPrec Macro","testPrec Macro",
                        "trainRec Micro","testRec Micro",
                        "trainRec Macro","testRec Macro",
                        "trainF1 Micro","testF1 Micro",
                        "trainF1 Macro","testF1 Macro",
                        ],CnvgHeader1,CnvgHeader1])
                    writer.writerow(header)
                a=numpy.concatenate([[x.optimizer,datasets[0],x.objfname,k+1,x.startTime,x.endTime,x.executionTime,
                    x.scores_dict['train']['acc'],x.scores_dict['test']['acc'],
                    x.scores_dict['train']['prec']['micro'],x.scores_dict['test']['prec']['micro'],
                    x.scores_dict['train']['prec']['macro'],x.scores_dict['test']['prec']['macro'],
                    x.scores_dict['train']['rec']['micro'],x.scores_dict['test']['rec']['micro'],
                    x.scores_dict['train']['rec']['macro'],x.scores_dict['test']['rec']['macro'],
                    x.scores_dict['train']['f1']['micro'],x.scores_dict['test']['f1']['micro'],
                    x.scores_dict['train']['f1']['macro'],x.scores_dict['test']['f1']['macro']],                   
                    x.convergence1,x.convergence2])
                writer.writerow(a)
            out.close()
        Flag=True
