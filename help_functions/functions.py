import numpy as np
import sys

def sigmoid(x, der=False):
    '''sigmoid funtion or derivative'''
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        return sigmoid(x) * (1-sigmoid(x))

def softmax(x):
    exps=np.exp(x)
    return exps / np.sum(exps)

def relu(x, der=False):
    y=np.copy(x)
    y[x<=0]=0
    if not der:
        return y
    else:
        y[x>0]=1
        return y

def Loss(y, y_hat, loss="square", der=False):
    """
    - y: one hot encoded true label of this form [0 ... 0 ..0 1 0 ...0]
    - y_hat: the output neural's network last layer [ dim(y)==dim(y_hat) ]
    """
    if loss=="square":
        if not der:
            return ((y-y_hat)**2).sum() # return scaler vector
        else:
            return 2*(y-y_hat) # return vector with shape = dim(y) 
    elif loss=="entropy":
        if not der:
            return -np.dot(y, np.log(y_hat)) # return scaler vector
        else:
            return y-y_hat # return vector with shape = dim(y)
    else:
        assert loss in ["square", "entropy"]

def accuracy(net, testX, testY):
    acc=[]
    for x, y in zip(testX, testY):
        out = net.forward(x)
        acc.append(all([out["guess"][k]==y[k] for k in range(len(y))]))
    return round(sum(acc) / len(acc), 5)

def scale(X):
    S=[]
    for x in X:
        s=(x-0.0) / (255.0 - 0.0)
        S.append(s.tolist())
    return np.array(S)

           
if __name__ == "__main__":
    # y=np.zeros(2, dtype="int"); y[np.random.randint(0, 1, size=1).item()]=1
    # y=np.array([0, 1])
    # y_hat=np.random.uniform(0.01, 0.99, size=10)
    # loss=F.Loss(y, y_hat, loss="entropy", der=False)
    # print(loss)
    X=np.random.randint(0, 255, size=(10, 1024))
    print(X)
    E=scale(X)
    print(E)
    print(E.shape)

    # y=np.array([0, 1])
    # y_hat=np.array([0.09242108, 0.9075789 ])
    # loss=F.Loss(y, y_hat, loss="entropy", der=False)
    # print(loss)