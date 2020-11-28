import jieba
from tqdm import tqdm
import pandas as pd
import numpy as np

stopwords = set([line.strip() for line in open("中文停用词表.txt", encoding="utf-8").readlines()]+
[line.strip() for line in open("哈工大停用词表.txt", encoding="utf-8").readlines()])

data=pd.read_csv('simplifyweibo_4_moods.csv',encoding="utf-8")



X=data.values[::100,1:]
ans=[]
for text in tqdm(X):
    temp=[]
    for j in list(jieba.cut(text[0])):
        if j not in stopwords:
            temp.append(j)
    ans.append(" ".join(temp))
ans="\n".join(ans)
with open("texts.txt","w",encoding="utf-8") as fp:
    fp.write(ans)

words=list(set(ans.split()))
dictionary=dict(zip(words,range(1,len(words)+1)))


y=data.values[::100,0]

res=[]
for label in tqdm(y):
    res.append(str(label))
res="\n".join(res)

with open("labels.txt","w",encoding="utf-8") as fp:
    fp.write(res)

np.save("dictionary.npy",dictionary)






