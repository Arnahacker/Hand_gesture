import numpy as np
import csv
from nn.model import NeuralNetwork
from nn.optimizers_functions import Adam
from nn.activation_function import Softmax,Tanh
from nn.loss import cross_entropy,cross_entropy_diff
from nn.dense import Dense
import pandas as pd

df=pd.read_csv("/Users/anoopchhabra/Documents/College/Projects/handgesture/data/final_data.csv")

df=df.sample(frac=1, random_state=42).reset_index(drop=True)

X=np.array(df.iloc[:,:63])

Y=np.zeros(df.shape[0])

categories={}
count=0

for i in range(df.shape[0]):
    if df.iloc[i,63] in categories:
        Y[i]=categories[df.iloc[i,63]]
    else:
        categories[df.iloc[i,63]]=count
        count+=1

X_train=X[:((len(X)*4)//5)]
X_test=X[((len(X)*4)//5):]
Y_train=Y[:((len(X)*4)//5)]
Y_test=Y[((len(X)*4)//5):]

