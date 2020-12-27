import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import help_functions.functions as F
from help_functions.helpers import shuffle, unpickle, one_hot_classes, DataBase

class MyNetwork:
    def __init__(self, X, y, neurons, param_path=None,lr=0.01, epoch=100, val_set_per=0.2, train=False, retrain=False):
        self.X = X
        self.y = y
        if neurons[-1]>1:
            self.y=one_hot_classes(self.y)
        self.param_path = param_path
        self.lr = lr
        self.epoch = epoch
        self.val_per = val_set_per
        self.layers = len(neurons)-1
        self.neurons = neurons
        self.Z=[]
        self.output={}

        if self.param_path!=None:
            if os.path.exists(self.param_path):
                if not train:
                    self.w, self.b = DataBase(self.param_path).load_par(num_w=len(self.neurons)-1)
                else:
                    if not retrain:
                        self.w, self.b = DataBase(self.param_path).load_par(num_w=len(self.neurons)-1)
                        self.train()
                    else:
                        self.w = [np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]),size=[self.neurons[l],self.neurons[l+1]]) for l in range(self.layers)]
                        self.b = [np.expand_dims(np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]), size=self.w[l].shape[1]), axis=0) for l in range(self.layers)]
                        self.train()
            else:
                self.w = [np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]),size=[self.neurons[l],self.neurons[l+1]]) for l in range(self.layers)]
                self.b = [np.expand_dims(np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]),size=self.w[l].shape[1]), axis=0) for l in range(self.layers)]
                self.train()
        else:
            self.w = [np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]),size=[self.neurons[l],self.neurons[l+1]]) for l in range(self.layers)]
            self.b = [np.expand_dims(np.random.uniform(-np.sqrt(1/self.neurons[l]),np.sqrt(1/self.neurons[l]),size=self.w[l].shape[1]), axis=0) for l in range(self.layers)]
            self.train()

    def split(self, X, y):
        trX, valX = np.split(X, [int(X.shape[0]*(1-self.val_per))])
        trY, valY = np.split(y, [int(X.shape[0]*(1-self.val_per))])
        self.train_examples=trX.shape[0]
        self.valid_examples=valX.shape[0]
        return trX, trY, valX, valY
    
    def iter_data(self,data, labels):
        for (x, y) in zip(data, labels):
            if x.ndim<2:
                x = np.expand_dims(x,axis=0)
            yield (x, y)
                
    def forward(self, x):
        if len(self.Z)>0: self.Z=[]
        self.output["layer_0"]=x
        for l in range(self.layers):
            if l<self.layers-1:
                self.Z.append(np.dot(self.output[f"layer_{l}"],self.w[l]) + self.b[l])
                self.output[f"layer_{l+1}"]=F.relu(self.Z[l])
            else:
                self.Z.append(np.dot(self.output[f"layer_{l}"],self.w[l]) + self.b[l])
                self.output[f"layer_{l+1}"] = F.softmax(self.Z[l])[0]

        # guess =  np.argmax(self.output[f"layer_{self.layers}"])
        pred=np.zeros(len(self.output[f"layer_{self.layers}"]), dtype="int")
        pred[np.argmax(self.output[f"layer_{self.layers}"])]=1
        return {"guess": pred, "output": self.output[f"layer_{self.layers}"]}

    
    def backprop(self, y, out):
        delta_loss_w={}
        delta_loss_b={}
        for l in range(self.layers-1, -1,-1): 
            if l==self.layers-1:
                delta=F.Loss(out, y, loss="entropy", der=True)*F.sigmoid(self.Z[-1], der=True) # delta: 1 x n_{N}
            else:
                # delta, w.T, relu(Z): [(1 x n_{l+1}) (matrix mult) (n_{l+1} x n_{l})] (element mult) (1 x n_{l}) = 1 x n_{l}
                delta = np.dot(delta, self.w[l+1].T) * F.relu(self.Z[l],der=True) 
            
            # delta, output.T: (1 x n_{l}) (element mult) (1 x n_{l}) = 1 x n_{l}    
            delta_loss_w[f"layer{l}"]=np.dot(delta.T, self.output[f"layer_{l}"]).T

            # delta: 1 x n_{l} (elment mult) 1 x n_{l+1} (matrix mult) n_{l+1} x n_{l} = 1 x n_{l}
            delta_loss_b[f"layer{l}"]=delta 
                
        return delta_loss_w, delta_loss_b  
            
    def gradient_update(self, y, y_hat, gd="SGD"):
        dL_dw, dL_db = self.backprop(y, y_hat)
        for l in range(self.layers-1,-1,-1):
            self.w[l]=self.w[l] - self.lr*dL_dw[f"layer{l}"]
            self.b[l]=self.b[l] - self.lr*dL_db[f"layer{l}"]

    def train(self):
        
        trainX, trainY, validX, validY = self.split(self.X, self.y)
        # for t in tqdm(range(self.epoch), ncols=50, leave=False):
        for t in range(self.epoch):
            train_error=0
            valid_error=0
            trainX, trainY = shuffle(trainX, trainY)
            for (x, y) in self.iter_data(trainX, trainY):
                out = self.forward(x)
                loss = F.Loss(y, out["output"], loss="entropy")
                train_error+=loss
                self.gradient_update(y, out["output"])
            train_error=train_error/trainX.shape[0]

            for (x, y) in self.iter_data(validX, validY):
                out=self.forward(x)
                loss = F.Loss(y, out["output"], loss="entropy")
                valid_error+=loss
            valid_error=valid_error/validX.shape[0]
            
            if t%1==0:
                print(" ", t, "train error:", round(train_error, 5), "validation error:", round(valid_error, 5))
        if self.param_path!=None:
            DataBase(self.param_path).save_par(parameters={"weights":self.w, "bias":self.b})
        else:
            DataBase("my_network/my_params/my_prm.txt").save_par(parameters={"weights":self.w, "bias":self.b})

    def predict(self, X):
        out=np.array([])
        for x in X:
            x = np.expand_dims(x, axis=0)
            if len(out)==0:
                out = self.forward(x)["output"] 
            else:
                out = np.vstack([out, self.forward(x)["output"]])
        return out

