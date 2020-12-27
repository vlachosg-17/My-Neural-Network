import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dct = pickle.load(fo, encoding='bytes')
    return dct

def shuffle(X, Y):
    xs=[]
    ys=[]
    for x, y in zip(X,Y):
        xs.append(x.tolist())
        ys.append(y.tolist())
    xs=np.array(xs)
    ys=np.array(ys)
    #print(xs.shape,ys.shape)
    new_raws = np.random.choice(np.arange(xs.shape[0]), size=xs.shape[0], replace=False)
    return xs[new_raws], ys[new_raws]

def one_hot(classes):
        u=[[1 if h==c else 0 for h in range(len(classes))] for c in classes]
        return u
    
def numerize_class(y):
    classes = np.sort(np.unique(y))
    num_classes=np.array([i for i in range(len(classes))])
    ys=np.array([num_classes[i] for el in y for i in range(len(classes)) if el==classes[i]])
    return ys, num_classes

def one_hot_classes(y):
    ys=np.copy(y)
    if ys.ndim<2:
        ys=np.expand_dims(ys, axis=1)
    ys, classes = numerize_class(ys)
    one_hot_classes = one_hot(classes)
    return np.array([one_hot_classes[c] for el in ys for c in range(len(classes)) if el==classes[c]], dtype="int")    

def Loss(y_hat, y, criterion="entropy"):
    if criterion=="entropy":   
        return -(y*np.log(y_hat)).sum(axis=1).mean()
    elif criterion=="mse":
        return ((y-y_hat)**2).sum(axis=1).mean()
        
def print_dims(Z, output, W):
    print("dim(Z) =",[z.shape for z in Z])
    print("dim(output) =",[v.shape for v in output.values()])
    print("dim(w) =",[w.shape for w in W])
    
class DataBase:
    def __init__(self, filename):
        self.file=filename
        self.num_par={}

    def save_drag(self, data, labels):
        self.dims=data.shape
        with open(self.file, "w") as f:
            for d, l in zip(data, labels):
                f.write(str(d.item())+","+str(l)+"\n")
    
    def load_drag(self):
        with open(self.file, "r") as f:
            data_observations=[]
            labels_obserations=[]
            for line in f:
                line=line.strip().split(",")
                data_observations.append([line[0]])
                labels_obserations.append([line[1]])
        data=np.array(data_observations, dtype="float32")
        labels=np.array(labels_obserations, dtype="int32")
        return data, labels
    
    def load_iris(self):
        with open(self.file, "r") as f:
            labels=[]
            data=[]
            for line in f:
                line=line.strip()
                if len(line)==0:
                    break
                line=line.split(",")
                data.append(line[:-1])
                labels.append(line[-1])
        data, labels = np.array(data, dtype="float32"), np.array(labels)
        # np.random.seed(0)
        return shuffle(data, labels)

    def save_par(self, parameters):
        with open(self.file, "w") as f:
            for key, parameter in parameters.items():
                self.num_par[key]=len(parameter)
                for l, par in enumerate(parameter):
                    f.write(f"layer_{l}"+":"+key+"\n")
                    for raw in par:
                        for r in raw:
                            if raw[-1]==r:
                                f.write(str(r)+"\n")
                            else:
                                f.write(str(r)+",")                    
                        
    def load_par(self, num_w):    
        parameters={"weights": [None]*num_w, "bias": [None]*num_w}
        with open(self.file, "r") as f:
            l1=-1
            l2=-1
            for line in f:
                line=line.strip()
                if line[:5]=="layer":
                    if line[-7:]=="weights":
                        l1+=1
                        key="weights"
                        parameters[key][l1]=[]
                    elif line[-4:]=="bias":
                        l2+=1
                        key="bias"
                        parameters[key][l2]=[]
                else:
                    line=line.split(",")
                    line=[float(el) for el in line]
                    if key=="weights":
                        parameters[key][l1].append(line)
                    elif key=="bias":
                        parameters[key][l2].append(line)
                    
        for key, par in parameters.items():
            for l in range(len(par)):
                parameters[key][l]=np.array(parameters[key][l], dtype="float32")
        return parameters["weights"], parameters["bias"]

if __name__ == "__main__":
    # w=DataBase("parameters/weights.txt").load_par(num_w=4-1)
    # for i in range(len(w)):
    #     print(w[i])
    # b=DataBase("parameters/bias.txt").load_par(num_w=4-1)
    # for i in range(len(b)):
    #     print(b[i])
    
    # d, l = DataBase("data/iris.data").load_iris()
    # # one_hot_classes(l)
    # print(one_hot_classes(l))


    X, Y = DataBase("data/iris.data").load_iris()
    # print(X.shape, Y.shape)
    # X, Y = shuffle(X, Y)
    print(numerize_class(Y))
    print(Y)
    # print(X.shape, Y.shape, X.shape[0]*0.2)
    # trainX, teX=np.split(X,[X.shape[0]-int(X.shape[0]*0.2)])
    # print(trainX.shape, teX.shape)
    # trainY, teY=np.split(Y,[Y.shape[0]-int(Y.shape[0]*0.2)])
    # print(trainY.shape, teY.shape)
    

