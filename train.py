import torch
from torch import embedding
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F 
import time
from tqdm import tqdm
import torchtext.vocab as Vocab
import argparse
import os
import pandas as pd
import numpy as np
import collections
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import datetime
from Network import Rnn_simple

dictionary=np.load("dictionary.npy",allow_pickle=True).item()

def labels_encode():
    y=[int(line) for line in open("labels.txt", encoding="utf-8").readlines()]
    y=np.expand_dims(np.array(y),1)
    encoder=OneHotEncoder()
    return encoder.fit_transform(y).toarray()


def texts_encode(X):
    for i in range(len(X)):
        X[i]=[np.int64(dictionary[j]) for j in X[i]]
    return X


def cut(X,length):
    for key,value in enumerate(X):
        if len(value)<=length:
            X[key]=value+[0 for i in range (length-len(value))]
        else:
            X[key]=X[key][:length]
    return X
def evaluate(valdata,vallabel,network):
    ans=0.;
    with torch.no_grad():
        prelabel=network(valdata)
        for i in range(len(prelabel)):
            if vallabel[i][torch.argmax(prelabel[i])]==1:
                ans+=1
    print("the accuracy is",ans/float(len(prelabel)))
    return ans/float(len(prelabel))


def trainloop( n_epochs,dataloader,network,optim,loss_fn,X,y):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        evaluate(X,y,network)
        for input, realout in dataloader:
            predictout = network(input)

            loss = loss_fn(predictout, realout)

            optim.zero_grad()

            loss.backward()
            optim.step()
            loss_train += loss.item()
            #if epoch == 1 or epoch % 100 == 0:
        print(
                f'{datetime.datetime.now()} epoch {epoch} training loss {loss_train/len(dataloader)}')  


if __name__=="__main__":
    Vocab=len(dictionary)+1
    embed_size=100
    num_hiddens=100
    num_layers=2
    network=Rnn_simple(Vocab,embed_size,num_hiddens,num_layers)

    y=torch.Tensor(labels_encode())


    X=[line.split() for line in open("texts.txt",encoding="utf-8").readlines()]
    X=texts_encode(X)
    X=np.array(cut(X,10))
    X=torch.LongTensor(X)


    n_epochs=50
    optim=torch.optim.Adam(network.parameters(),lr=0.001)
    dataset=TensorDataset(X,torch.Tensor(y))
    dataloader=DataLoader(dataset,batch_size=300,shuffle=True)
    loss_fn=nn.BCEWithLogitsLoss()

    if "Rnn-simple.pkl" in os.listdir():
        network.load_state_dict(torch.load("Rnn-simple.pkl"))
    count=0
    while evaluate(X,y,network)<=0.998 and count<n_epochs:
        trainloop(10,dataloader,network,optim,loss_fn,X,y)
        count+=10
        #print("the accuracy is",evaluate(X,y,network))
    
    torch.save(network.state_dict(),"Rnn-simple.pkl")




