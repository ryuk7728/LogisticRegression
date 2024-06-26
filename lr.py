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
    derw=(1/m)*np.dot(X_train.T,y_pred-y_train) #Multiplying x values associated with weight to the error to determine to what extent to change w to reduce error
    return derw

def derB(w,b,y_train,X_train,m):
    lin=np.dot(X_train,w)+b
    y_pred=1/(1+np.exp(-lin))
    derb=(1/m)*np.sum(y_pred-y_train) #Calculating average error to determine to what extent to change w to reduce error
    return derb

def gradient_descent(w,b,y_train,X_train,m,iter,alpha,costx,costs):
    
    finloss=0
    for i in range(iter):
        temp_w=w-alpha*derW(w,b,y_train,X_train,m)
        temp_b=b-alpha*derB(w,b,y_train,X_train,m)
        w=temp_w
        b=temp_b
        costx.append(i+1)
        costs.append(computeCost(w,b,y_train,X_train,m))
        if(i==iter-1):
            finloss=computeCost(w,b,y_train,X_train,m)
    return w,b,costx,costs,finloss



X_train = np.array([[0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5],
    [0.5, 3],
    [1.5, 2],
    [2.5, 1.5],
    [3.5, 1],
    [4, 2],
    [2, 3],
    [0.5, 0.5],
    [1.5, 1.5],
    [2.5, 2.5],
    [3, 3],
    [0, 2],
    [4, 0],
    [3, 1.5],
    [1, 3]])
y_train = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0])
m=X_train.shape[0]
n=X_train.shape[1]
w=np.zeros(n,)
b=0
iter=100000
alpha=1
costx=[0]
costs=[]
costs.append(computeCost(w,b,y_train,X_train,m))
finloss=0

w,b,costx,costs,finloss=gradient_descent(w,b,y_train,X_train,m,iter,alpha,costx,costs)
plt.plot(costx,costs,color='red')
plt.show()
print(f"The final weight and bias is:{w}, {b}. The final loss is {finloss}")
lin=np.dot(X_train,w)+b
y_pred=1/(1+np.exp(-lin))
print(y_pred)
for i in range(m):
    if y_train[i]==0:
        plt.scatter(X_train[i][0],X_train[i][1],color='blue',marker='o')
    else:
        plt.scatter(X_train[i][0],X_train[i][1],color='red',marker='x')

decx=np.linspace(-5,5,1000)
decy=-(w[0]*decx+b)/w[1]
plt.plot(decx,decy,color='black')
plt.show()
