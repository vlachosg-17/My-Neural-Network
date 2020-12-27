# from helpers import DataBase, one_hot_classes, shuffle
import argparse
from help_functions import functions as F
from help_functions import helpers as H
from my_network.my_neural_net import MyNetwork
import numpy as np

def main(args):
    X, Y = H.DataBase("data/drag.txt").load_drag()
    trainX, teX=np.split(X, [X.shape[0]-int(X.shape[0]*args.testset_percent)])
    trainY, teY=np.split(Y, [X.shape[0]-int(X.shape[0]*args.testset_percent)])
    teY=H.one_hot_classes(teY)
    net = MyNetwork(
        trainX, trainY, param_path=args.param_dir, \
        neurons=[trainX.shape[1]]+ args.hidden_layers+[len(np.unique(trainY))], 
        lr=args.lr, epoch=args.epochs, train=args.allow_train, retrain=args.retrain
        )
    print(F.accuracy(net, teX, teY))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--lr",              type=float, default=0.01)
    parser.add_argument("--allow_train",     type=bool,  default=True)
    parser.add_argument("--retrain",         type=bool,  default=True)
    parser.add_argument("--param_dir",       type=str,   default="my_network/my_params/drag_prm.txt")
    parser.add_argument("--hidden_layers",   type=list,  default=[20, 10, 10])
    parser.add_argument("--testset_percent", type=float, default=0.2)

    args=parser.parse_args()

    main(args)
    
