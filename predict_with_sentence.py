import torch
from torch import embedding
import torch.nn as nn
from torch.nn.modules import loss
from torch.utils.data import DataLoader, TensorDataset
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
import jieba
from aip import AipSpeech
import pyaudio
import wave
from Baidu_api import get_audio,get_file_content
import logging
jieba.setLogLevel(logging.INFO)


dictionary = np.load("dictionary.npy", allow_pickle=True).item()
stopwords = set([line.strip() for line in open("中文停用词表.txt", encoding="utf-8").readlines()] +
                [line.strip() for line in open("哈工大停用词表.txt", encoding="utf-8").readlines()])
emotions = ["negative", "positive"]


def get_input(sentence, network):

    X = list(jieba.cut(sentence))
    t = X[:]
    X.clear()
    for word in t:
        if word not in stopwords:
            X.append(dictionary.get(word, 0))
    if (len(X) < 10):
        X = X+[0 for i in range(10-len(X))]
    else:
        X = X[:10]
    X = np.expand_dims(X, 0)

    y = network(torch.LongTensor(X))
        # print(y)
    print("your emotion is :", emotions[torch.argmax(y)])


def run_emotion_detection():

    APP_ID= "23038327"

    API_KEY= "7RwjGAYYTHXBKWo4dcP3Aaga"

    SECRET_KEY="bonCBakFEuCTL4SjQkbtdBIF38XG3dEB"

    Clinet= AipSpeech(APP_ID,API_KEY,SECRET_KEY)
    in_path="./audios/input.wav"
    get_audio(in_path)
    '''
    1537 普通话
    1737 英语
    '''
    result=Clinet.asr(get_file_content(in_path),'wav',16000,{'dev_pid':1537,})


    Vocab = len(dictionary)+1
    embed_size = 100
    num_hiddens = 100
    num_layers = 2
    network = Rnn_simple(Vocab, embed_size, num_hiddens, num_layers)
    network.load_state_dict(torch.load("Rnn-simple.pkl"))

    sentence = result['result'][0]
    print("you said:",sentence)
    get_input(sentence, network)

if __name__ == "__main__":
    run_emotion_detection()
    raw_input("Press <enter>")
