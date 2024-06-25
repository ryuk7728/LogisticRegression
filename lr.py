import numpy as np
import matplotlib.pyplot as plt

def computeCost(w,b,y_train,X_train,m):
    lin=np.dot(X_train,w)+b
    y_pred=1/(1+np.exp(-lin))
    cost=-(1/m)*np.sum((y_train*np.log(y_pred))+((1-y_train)*np.log(1-y_pred)))
    return cost

def derW(w,b,y_train,X_train,m):
    lin=np.dot(X_train,w)+b
    y_pred=1/(1+np.exp(-lin))
    derw=(1/m)*np.dot(X_train.T,y_pred-y_train)
    return derw

def derB(w,b,y_train,X_train,m):
    lin=np.dot(X_train,w)+b
    y_pred=1/(1+np.exp(-lin))
    derb=(1/m)*np.sum(y_pred-y_train)
    return derb

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
m=X_train.shape[0]
n=X_train.shape[1]
w=np.zeros(n,)
b=0
computeCost(w,b,y_train,X_train,m)