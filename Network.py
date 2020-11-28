import torch
import torch.nn as nn
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


class Rnn_simple(nn.Module):
    def __init__(self,vocab,embed_size,num_hiddens,num_layers):
        super(Rnn_simple,self).__init__()
        self.embedding=nn.Embedding((vocab),embed_size)
        self.encoder=nn.LSTM(input_size=embed_size,hidden_size=num_hiddens,num_layers=num_layers,bidirectional=True)
        self.decoder=nn.Linear(4*num_hiddens,2)
    def forward(self,X):
        X=self.embedding(X.permute(1,0))
        X,_=self.encoder(X)
        X=torch.cat((X[0],X[-1]),-1)
        X=self.decoder(X)
        return X

