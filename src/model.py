import numpy as np
from nn.model import NeuralNetwork
from nn.optimizers_functions import Adam
from nn.activation_function import Softmax,Tanh
from nn.loss import cross_entropy,cross_entropy_diff
from nn.dense import Dense
import pandas as pd
import csv

df=pd.read_csv("/Users/anoopchhabra/Documents/College/Projects/handgesture/data/final_data.csv")

df=df.sample(frac=1, random_state=42).reset_index(drop=True)

X=np.array(df.iloc[:,:63])

Y_before=np.zeros(df.shape[0])

categories={}
count=0

file = open("/Users/anoopchhabra/Documents/College/Projects/handgesture/src/label.csv","w+")
writer= csv.writer(file)

for i in range(df.shape[0]):
    if df.iloc[i,63] in categories:
        Y_before[i]=categories[df.iloc[i,63]]
    else:
        writer.writerow([df.iloc[i,63],count])
        categories[df.iloc[i,63]]=count
        Y_before[i]=count
        count+=1

count-=1

Y_new=np.zeros((df.shape[0],count+1))

for i in range(df.shape[0]):
    a=[0]*(count+1)
    a[categories[df.iloc[i,63]]]=1
    Y_new[i]=a


X_train=X[:((len(X)*4)//5)]
X_test=X[((len(X)*4)//5):]
Y_train=Y_new[:((len(X)*4)//5)]
Y_test=Y_new[((len(X)*4)//5):]

network=NeuralNetwork([
    Dense(63,30),
    Tanh(),
    Dense(30,15),
    Tanh(),
    Dense(15,count+1),
    Softmax()],cross_entropy,cross_entropy_diff
)

optimizer = Adam(lr=0.01)

network.fit(X_train, Y_train, epochs=500, optimizer=optimizer, verbose=True)

def predict_class(model, x):
    outputs = model.predict(x)
    preds = [np.argmax(o) for o in outputs]
    return np.array(preds)

y_pred = predict_class(network, X_test)
y_true = np.argmax(Y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
network.save("/Users/anoopchhabra/Documents/College/Projects/handgesture/models/gesture_model.pkl")




