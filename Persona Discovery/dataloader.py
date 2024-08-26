from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import os
import numpy as np
import pandas as pd
import tqdm
from nltk import word_tokenize
from string import punctuation
from collections import Counter
import operator

from torchtext import datasets
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F

import hiddenlayer as hl
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import string
import re
import nltk
import gensim

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

filename = './../data/glove.6B.200d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
print("Model Loaded")

#############################################
punct = set(string.punctuation)
punct.remove("!")
punct.remove("?")

numbers = {
    "0":"zero",
    "1":"one",
    "2":"two",
    "3":"three",
    "4":"four",
    "5":"five",
    "6":"six",
    "7":"seven",
    "8":"eight",
    "9":"nine"
}

def remove_puntuations(txt):
    txt = " ".join(txt.split("."))
    txt = " ".join(txt.split(":"))
    txt = " ".join(txt.split(";"))
    txt = " ".join(txt.split("-"))
    
    txt = "".join(ch for ch in txt if ch not in punct)
    return txt

def number_to_words(txt):
    for k in numbers.keys():
        txt = txt.replace(k,numbers[k]+" ")
    return txt

def preprocess_sentence(text):
    text = text.lower()
    text = re.sub(r'_',' ',text)
    text = number_to_words(text)
    text = remove_puntuations(text)
    text = ''.join([i if ord(i) < 128 else '' for i in text])
    text_tokens = nltk.word_tokenize(text)
    text = ' '.join(text_tokens)
    return text 
#############################################

personas2consider = ['trait','likes','relationship','occupation','misc','dislikes','education']

data_path = "./../data/meld4sp - Data - new.csv"
df_dev = pd.read_csv("./../data/meld4sp-dev - meld4sp-dev - new.csv")
df_test = pd.read_csv("./../data/meld4sp-test - meld4sp-test - new.csv")

df = pd.read_csv(data_path)
all_utterances = df["Utterance"].append(df_dev["Utterance"]).append(df_test["Utterance"])
all_persona = df["Persona?"].append(df_dev["Persona?"]).append(df_test["Persona?"])
all_da = df["eda"].append(df_dev["eda"]).append(df_test["eda"])

for i,p in enumerate(all_persona):
    if type(p) != str:
        all_persona[i] = 'no'

idx2persona = ['yes','no','<pad>']
print("Personas -> ",idx2persona)

persona2idx = {v:k for k,v in enumerate(idx2persona)}
total_persona = len(idx2persona)

idx2word = ['<pad>']
for utt in all_utterances:
    if type(utt) == str:
        cln_utt = preprocess_sentence(utt)
        words = cln_utt.split()
        idx2word.extend(words)

idx2word = list(set(idx2word))
word2idx = {v:k for k,v in enumerate(idx2word)}

idx2da = list(set(all_da))
da2idx = {v:k for k,v in enumerate(idx2da)}

persona_count = Counter(all_persona)
max_cnt = sorted(persona_count.items(),key=operator.itemgetter(1),reverse=True)[0][1]
weights = [0]*len(persona2idx)
for key,value in persona_count.items():
    if type(key) == str:
        weights[persona2idx[key]] = max_cnt/value

print('weights- ',weights)
batch_size = 8
seq_len = 70
seq_len2 = 33
emb_size = 200
hidden_size = 200
batch_first = True
model_name = "LSTMs"

weight_matrix = np.zeros((len(idx2word),emb_size))
for i,word in enumerate(idx2word):
    try: 
        weight_matrix[i] = model.wv.get_vector(word)
    except KeyError:
        weight_matrix[i] = np.random.normal(scale=0.6, size=(200,))

weight_matrix = torch.from_numpy(weight_matrix).to(device)
#############################################        
#############################################

X_train_id = []
X_train = []
y_train = []
X_train_utt_len = {}
X_train_speaker = []
X_train_da = []

train_utt_len = []
this_dialogue = []
this_dialogue_y = []
this_dialogue_tmp =[]
this_dialogue_speaker = []
this_dialogue_da = []
this_dialogue_speaker_set_list = []

d_id = 0

for utt,persona,slot,speaker,da in tqdm.tqdm(zip(df['Utterance'],df['Persona?'],df['Persona Slot'],df['Speaker'],df['eda'])):
    if type(utt) != str:
        if this_dialogue != []:
            if len(this_dialogue) < seq_len2:
                for k in range(len(this_dialogue),seq_len2):
                    this_dialogue.append([word2idx["<pad>"]]*seq_len)
                    this_dialogue_y.append(persona2idx["<pad>"])
                    
                    sp = [0]*10
                    sp[9] = 1
                    this_dialogue_speaker.append(sp)
                    
                    eda = [0]*34
                    eda[da2idx["xx"]] = 1
                    this_dialogue_da.append(eda)
            else:
                this_dialogue = this_dialogue[:seq_len2]
                this_dialogue_y = this_dialogue_y[:seq_len2]
                this_dialogue_speaker = this_dialogue_speaker[:seq_len2]
                this_dialogue_da = this_dialogue_da[:seq_len2]

            X_train_id.append(d_id)
            X_train.append(this_dialogue)
            X_train_utt_len[d_id] = [len(train_utt_len),train_utt_len]
            X_train_speaker.append(this_dialogue_speaker)
            X_train_da.append(this_dialogue_da)
            y_train.append(this_dialogue_y)

            this_dialogue = []
            this_dialogue_y = []
            train_utt_len = []
            this_dialogue_speaker = []
            this_dialogue_speaker_set_list = []
            this_dialogue_da = []
            d_id += 1
    else:
        this_utt =[]
        cln_utt = preprocess_sentence(utt)
        words = cln_utt.split()
        for word in words:
            this_utt.append(word2idx[word])

        if len(this_utt) < seq_len:
            train_utt_len.append(len(this_utt))
            for k in range(len(this_utt),seq_len):
                this_utt.append(word2idx["<pad>"])
        else:
            train_utt_len.append(seq_len)
            this_utt = this_utt[:seq_len]

        this_dialogue.append(this_utt)
        if type(persona) != str:
            this_dialogue_y.append(persona2idx['no'])
        else:
            slots = slot.split('\n')
            slots = [s.strip() for s in slots]
            found = False
            for s in slots:
                if s in personas2consider:
                    this_dialogue_y.append(persona2idx['yes'])
                    found = True
                    break
            if not found:
                this_dialogue_y.append(persona2idx['no'])
                   
        if speaker not in this_dialogue_speaker_set_list:
            this_dialogue_speaker_set_list.append(speaker)
        
        idx = this_dialogue_speaker_set_list.index(speaker)
        sp = [0]*10
        sp[idx] = 1
        this_dialogue_speaker.append(sp)
        
        eda = [0]*34
        eda[da2idx[da]] = 1
        this_dialogue_da.append(eda)
        
print("Number of training samples = ",len(X_train))
W = torch.LongTensor(X_train_id)
X = torch.LongTensor(X_train)
Y = torch.LongTensor(y_train)
Z = torch.LongTensor(X_train_speaker)
V = torch.LongTensor(X_train_da)

print("W size -> ",W.size())
print("X size -> ",X.size())
print("Y size -> ",Y.size())
print("Z size -> ",Z.size())
print("V size -> ",V.size())

my_dataset = data.TensorDataset(W,X,Y,Z,V)
data_iter_train = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)

#############################################
X_dev_id = []
X_dev = []
y_dev = []
X_dev_utt_len = {}
X_dev_speaker = []
X_dev_da = []

dev_utt_len = []
this_dialogue = []
this_dialogue_y = []
this_dialogue_tmp =[]
this_dialogue_speaker = []
this_dialogue_da = []
this_dialogue_speaker_set_list = []

d_id = 0

for utt,persona,slot,speaker,da in tqdm.tqdm(zip(df_dev['Utterance'],df_dev['Persona?'],df_dev['Persona Slot'],df_dev['Speaker'],df_dev['eda'])):
    if type(utt) != str:
        if this_dialogue != []:
            if len(this_dialogue) < seq_len2:
                for k in range(len(this_dialogue),seq_len2):
                    this_dialogue.append([word2idx["<pad>"]]*seq_len)
                    this_dialogue_y.append(persona2idx["<pad>"])
                    
                    sp = [0]*10
                    sp[9] = 1
                    this_dialogue_speaker.append(sp)
                    
                    eda = [0]*34
                    eda[da2idx["xx"]] = 1
                    this_dialogue_da.append(eda)
            else:
                this_dialogue = this_dialogue[:seq_len2]
                this_dialogue_y = this_dialogue_y[:seq_len2]
                this_dialogue_speaker = this_dialogue_speaker[:seq_len2]
                this_dialogue_da = this_dialogue_da[:seq_len2]

            X_dev_id.append(d_id)
            X_dev.append(this_dialogue)
            X_dev_utt_len[d_id] = [len(dev_utt_len),dev_utt_len]
            X_dev_speaker.append(this_dialogue_speaker)
            X_dev_da.append(this_dialogue_da)
            y_dev.append(this_dialogue_y)

            this_dialogue = []
            this_dialogue_y = []
            dev_utt_len = []
            this_dialogue_speaker = []
            this_dialogue_da = []
            this_dialogue_speaker_set_list = []
            d_id += 1
    else:
        this_utt =[]
        cln_utt = preprocess_sentence(utt)
        words = cln_utt.split()
        for word in words:
            this_utt.append(word2idx[word])

        if len(this_utt) < seq_len:
            dev_utt_len.append(len(this_utt))
            for k in range(len(this_utt),seq_len):
                this_utt.append(word2idx["<pad>"])
        else:
            dev_utt_len.append(seq_len)
            this_utt = this_utt[:seq_len]
        this_dialogue.append(this_utt)
        if type(persona) != str:
            this_dialogue_y.append(persona2idx['no'])

        else:
            slots = slot.split('\n')
            slots = [s.strip() for s in slots]
            found = False
            for s in slots:
                if s in personas2consider:
                    this_dialogue_y.append(persona2idx['yes'])
                    found = True
                    break
            if not found:
                this_dialogue_y.append(persona2idx['no'])
                   
        if speaker not in this_dialogue_speaker_set_list:
            this_dialogue_speaker_set_list.append(speaker)
        
        idx = this_dialogue_speaker_set_list.index(speaker)
        sp = [0]*10
        sp[idx] = 1
        this_dialogue_speaker.append(sp)
        
        eda = [0]*34
        eda[da2idx[da]] = 1
        this_dialogue_da.append(eda)
        
print("Number of deving samples = ",len(X_dev))
W = torch.LongTensor(X_dev_id)
X = torch.LongTensor(X_dev)
Y = torch.LongTensor(y_dev)
Z = torch.LongTensor(X_dev_speaker)
V = torch.LongTensor(X_dev_da)

print("W size -> ",W.size())
print("X size -> ",X.size())
print("Y size -> ",Y.size())
print("Z size -> ",Z.size())
print("V size -> ",V.size())

my_dataset = data.TensorDataset(W,X,Y,Z,V)
data_iter_dev = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)

#############################################
X_test_id = []
X_test = []
y_test = []
X_test_utt_len = {}
X_test_speaker = []
X_test_da = []

test_utt_len = []
this_dialogue = []
this_dialogue_y = []
this_dialogue_tmp =[]
this_dialogue_speaker = []
this_dialogue_da = []
this_dialogue_speaker_set_list = []

d_id = 0

for utt,persona,slot,speaker,da in tqdm.tqdm(zip(df_test['Utterance'],df_test['Persona?'],df_test['Persona Slot'],df_test['Speaker'],df_test['eda'])):
    if type(utt) != str:
        if this_dialogue != []:
            if len(this_dialogue) < seq_len2:
                for k in range(len(this_dialogue),seq_len2):
                    this_dialogue.append([word2idx["<pad>"]]*seq_len)
                    this_dialogue_y.append(persona2idx["<pad>"])
                    
                    sp = [0]*10
                    sp[9] = 1
                    this_dialogue_speaker.append(sp)
                    
                    eda = [0]*34
                    eda[da2idx["xx"]] = 1
                    this_dialogue_da.append(eda)
            else:
                this_dialogue = this_dialogue[:seq_len2]
                this_dialogue_y = this_dialogue_y[:seq_len2]
                this_dialogue_speaker = this_dialogue_speaker[:seq_len2]
                this_dialogue_da = this_dialogue_da[:seq_len2]

            X_test_id.append(d_id)
            X_test.append(this_dialogue)
            X_test_utt_len[d_id] = [len(test_utt_len),test_utt_len]
            X_test_speaker.append(this_dialogue_speaker)
            X_test_da.append(this_dialogue_da)
            y_test.append(this_dialogue_y)

            this_dialogue = []
            this_dialogue_y = []
            test_utt_len = []
            this_dialogue_speaker = []
            this_dialogue_da = []
            this_dialogue_speaker_set_list = []
            d_id += 1
    else:
        this_utt =[]
        cln_utt = preprocess_sentence(utt)
        words = cln_utt.split()
        for word in words:
            this_utt.append(word2idx[word])

        if len(this_utt) < seq_len:
            test_utt_len.append(len(this_utt))
            for k in range(len(this_utt),seq_len):
                this_utt.append(word2idx["<pad>"])
        else:
            test_utt_len.append(seq_len)
            this_utt = this_utt[:seq_len]

        this_dialogue.append(this_utt)
        if type(persona) != str:
            this_dialogue_y.append(persona2idx['no'])
        else:
            slots = slot.split('\n')
            slots = [s.strip() for s in slots]
            found = False
            for s in slots:
                if s in personas2consider:
                    this_dialogue_y.append(persona2idx['yes'])
                    found = True
                    break
            if not found:
                this_dialogue_y.append(persona2idx['no'])
                   
        if speaker not in this_dialogue_speaker_set_list:
            this_dialogue_speaker_set_list.append(speaker)
        
        idx = this_dialogue_speaker_set_list.index(speaker)
        sp = [0]*10
        sp[idx] = 1
        this_dialogue_speaker.append(sp)
        
        eda = [0]*34
        eda[da2idx[da]] = 1
        this_dialogue_da.append(eda)

print("Number of testing samples = ",len(X_test))
W = torch.LongTensor(X_test_id)
X = torch.LongTensor(X_test)
Y = torch.LongTensor(y_test)
Z = torch.LongTensor(X_test_speaker)
V = torch.LongTensor(X_test_da)

print("W size -> ",W.size())
print("X size -> ",X.size())
print("Y size -> ",Y.size())
print("Z size -> ",Z.size())
print("V size -> ",V.size())

my_dataset = data.TensorDataset(W,X,Y,Z,V)
data_iter_test = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)
       
#############################################

with open("data/idx2persona.p",'wb') as f:
    pickle.dump(idx2persona,f)
with open("data/persona2idx.p",'wb') as f:
    pickle.dump(persona2idx,f)
    
with open("data/idx2word.p",'wb') as f:
    pickle.dump(idx2word,f)
with open("data/word2idx.p",'wb') as f:
    pickle.dump(word2idx,f)
    
with open("data/idx2da.p",'wb') as f:
    pickle.dump(idx2da,f)
with open("data/da2idx.p",'wb') as f:
    pickle.dump(da2idx,f)
    
with open("data/weight_matrix.p",'wb') as f:
    pickle.dump(weight_matrix,f)
    
with open("data/data_iter_train.p",'wb') as f:
    pickle.dump(data_iter_train,f)
    
with open("data/data_iter_dev.p",'wb') as f:
    pickle.dump(data_iter_dev,f)
    
with open("data/data_iter_test.p",'wb') as f:
    pickle.dump(data_iter_test,f)
    
with open("data/X_train_utt_len.p",'wb') as f:
    pickle.dump(X_train_utt_len,f)
    
with open("data/X_dev_utt_len.p",'wb') as f:
    pickle.dump(X_dev_utt_len,f)
    
with open("data/X_test_utt_len.p",'wb') as f:
    pickle.dump(X_test_utt_len,f)