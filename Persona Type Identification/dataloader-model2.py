from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
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

seq_len = 33
seq_len2 = 70
emb_size = 200

#############################################
model = KeyedVectors.load_word2vec_format("./../data/glove.6B.200d.txt.word2vec", binary=False)
print("Model Loaded")

df = pd.read_csv("./../data/meld4sp - Data - new.csv")
df_dev = pd.read_csv("./../data/meld4sp-dev - meld4sp-dev - new.csv")
df_test = pd.read_csv("./../data/meld4sp-test - meld4sp-test - new.csv")

all_utterances = df["Utterance"].append(df_dev["Utterance"]).append(df_test["Utterance"])
all_persona = df["Persona Slot"].append(df_dev["Persona Slot"]).append(df_test["Persona Slot"])

idx2persona = list(set([ap for ap in all_persona if type(ap) != float]))
# idx2persona.insert(0,"<pad>")
persona2idx = {v:k for k,v in enumerate(idx2persona)}
total_persona = len(idx2persona)

idx2word = []
for utt in all_utterances:
    if type(utt) == str:
        cln_utt = preprocess_sentence(utt)
        words = cln_utt.split()
        idx2word.extend(words)
idx2word = list(set(idx2word))
idx2word.insert(0,"<pad>")
word2idx = {v:k for k,v in enumerate(idx2word)}

weight_matrix = np.zeros((len(idx2word),emb_size))
f = 0
for i,word in enumerate(idx2word):
    try: 
        weight_matrix[i] = model.wv.get_vector(word)
        f += 1
    except KeyError:
        weight_matrix[i] = np.random.normal(scale=0.6, size=(emb_size,))

print(f"Out of {len(idx2word)}, {f} were found, i.e. {(f*100)/len(idx2word)}% were found")
weight_matrix = torch.from_numpy(weight_matrix)#.to(device)

with open("data/idx2persona.pt","wb") as f:
    pickle.dump(idx2persona,f)
with open("data/persona2idx.pt","wb") as f:
    pickle.dump(persona2idx,f)
    
with open("data/idx2word.pt","wb") as f:
    pickle.dump(idx2word,f)
with open("data/word2idx.pt","wb") as f:
    pickle.dump(word2idx,f)
    
with open("data/weight_matrix.pt","wb") as f:
    pickle.dump(weight_matrix,f)
#############################################

d_id = 0
this_dialog = []
this_dialog_speaker = []
this_dialog_speaker_set = []
this_utt_len = []

X1, X2, X3, Y1, Y2 = [], [], [], [], []

speaker_info = {}
dialog_len = {}
utt_len = {}

for d,utt,slot,speaker,target in tqdm.tqdm(zip(df["Dialogue_Id"], df["Utterance"], df["Persona Slot"], df["Speaker"], df["Target"])):
    if type(utt) == float:
        this_dialog = []
        this_dialog_speaker = []
        this_dialog_speaker_set = []
        this_utt_len = []
        continue
        
    utt = preprocess_sentence(utt)
    words = utt.split()
    this_utt = []
    for word in words:
        this_utt.append(word2idx[word])
        
    this_utt_len.append(len(this_utt))
    
    if len(this_utt) < seq_len2:
        for k in range(len(this_utt),seq_len2):
            this_utt.append(word2idx["<pad>"])
    else:
        this_utt = this_utt[len(this_utt)-seq_len2:seq_len2]
    this_dialog.append(this_utt)

    speaker = preprocess_sentence(speaker)
    if speaker not in this_dialog_speaker_set:
        this_dialog_speaker_set.append(speaker)
    idx = this_dialog_speaker_set.index(speaker)
    this_dialog_speaker.append(idx)
    
    if type(slot) != float:
        this_dialog_padded = this_dialog[:]
        this_dialog_speaker_padded = this_dialog_speaker[:]
        
        if len(this_dialog_padded) < seq_len:
            for k in range(len(this_dialog),seq_len):
                this_dialog_padded.append([word2idx["<pad>"]]*seq_len2)
                this_dialog_speaker_padded.append(9)
        else:
            this_dialog_padded = this_dialog_padded[len(this_dialog_padded)-seq_len:seq_len]
            this_dialog_speaker_padded = this_dialog_speaker_padded[len(this_dialog_speaker_padded)-seq_len:seq_len]
            
        X1.append(d_id)
        X2.append(this_dialog_padded)
        X3.append(this_dialog_speaker_padded)
        Y1.append(persona2idx[slot])
        
        target = preprocess_sentence(target)
        if target not in this_dialog_speaker_set:
            this_dialog_speaker_set.append(target)
        idx = this_dialog_speaker_set.index(target)
        Y2.append(idx)
        
        speaker_info[d_id] = this_dialog_speaker[:]
        dialog_len[d_id] = len(this_dialog)
        utt_len[d_id] = this_utt_len[:]
        
        d_id += 1

print("Number of training samples = ",len(X1))
X1 = torch.LongTensor(X1)
X2 = torch.LongTensor(X2)
X3 = torch.LongTensor(X3)
Y1 = torch.LongTensor(Y1)
Y2 = torch.LongTensor(Y2)

print("X1 size -> ",X1.size())
print("X2 size -> ",X2.size())
print("X3 size -> ",X3.size())
print("Y1 size -> ",Y1.size())
print("Y2 size -> ",Y2.size())

my_dataset = data.TensorDataset(X1,X2,X3,Y1,Y2)

with open("data/train_dataset.pt","wb") as f:
    pickle.dump(my_dataset,f)
with open("data/train_dialog_len.pt","wb") as f:
    pickle.dump(dialog_len,f)
with open("data/train_speaker_info.pt","wb") as f:
    pickle.dump(speaker_info,f)
with open("data/train_utt_len.pt","wb") as f:
    pickle.dump(utt_len,f)
#############################################

d_id = 0
this_dialog = []
this_dialog_speaker = []
this_dialog_speaker_set = []
this_utt_len = []

X1, X2, X3, Y1, Y2 = [], [], [], [], []

speaker_info = {}
dialog_len = {}
utt_len = {}

for d,utt,slot,speaker,target in tqdm.tqdm(zip(df_dev["Dialogue_Id"], df_dev["Utterance"], df_dev["Persona Slot"], df_dev["Speaker"], df_dev["Target"])):
    if type(utt) == float:
        this_dialog = []
        this_dialog_speaker = []
        this_dialog_speaker_set = []
        this_utt_len = []
        continue
        
    utt = preprocess_sentence(utt)
    words = utt.split()
    this_utt = []
    for word in words:
        this_utt.append(word2idx[word])
        
    this_utt_len.append(len(this_utt))

    if len(this_utt) < seq_len2:
        for k in range(len(this_utt),seq_len2):
            this_utt.append(word2idx["<pad>"])
    else:
        this_utt = this_utt[len(this_utt)-seq_len2:seq_len2]
    this_dialog.append(this_utt)

    speaker = preprocess_sentence(speaker)
    if speaker not in this_dialog_speaker_set:
        this_dialog_speaker_set.append(speaker)
    idx = this_dialog_speaker_set.index(speaker)
    this_dialog_speaker.append(idx)
    
    if type(slot) != float:
        this_dialog_padded = this_dialog[:]
        this_dialog_speaker_padded = this_dialog_speaker[:]
        
        if len(this_dialog_padded) < seq_len:
            for k in range(len(this_dialog),seq_len):
                this_dialog_padded.append([word2idx["<pad>"]]*seq_len2)
                this_dialog_speaker_padded.append(9)
        else:
            this_dialog_padded = this_dialog_padded[len(this_dialog_padded)-seq_len:seq_len]
            this_dialog_speaker_padded = this_dialog_speaker_padded[len(this_dialog_speaker_padded)-seq_len:seq_len]
            
        X1.append(d_id)
        X2.append(this_dialog_padded)
        X3.append(this_dialog_speaker_padded)
        Y1.append(persona2idx[slot])
        
        target = preprocess_sentence(target)
        if target not in this_dialog_speaker_set:
            this_dialog_speaker_set.append(target)
        idx = this_dialog_speaker_set.index(target)
        Y2.append(idx)
        
        speaker_info[d_id] = this_dialog_speaker[:]
        dialog_len[d_id] = len(this_dialog)
        utt_len[d_id] = this_utt_len[:]
        
        d_id += 1

print("Number of training samples = ",len(X1))
X1 = torch.LongTensor(X1)
X2 = torch.LongTensor(X2)
X3 = torch.LongTensor(X3)
Y1 = torch.LongTensor(Y1)
Y2 = torch.LongTensor(Y2)

print("X1 size -> ",X1.size())
print("X2 size -> ",X2.size())
print("X3 size -> ",X3.size())
print("Y1 size -> ",Y1.size())
print("Y2 size -> ",Y2.size())

my_dataset = data.TensorDataset(X1,X2,X3,Y1,Y2)

with open("data/dev_dataset.pt","wb") as f:
    pickle.dump(my_dataset,f)
with open("data/dev_dialog_len.pt","wb") as f:
    pickle.dump(dialog_len,f)
with open("data/dev_speaker_info.pt","wb") as f:
    pickle.dump(speaker_info,f)
with open("data/dev_utt_len.pt","wb") as f:
    pickle.dump(utt_len,f)
#############################################

d_id = 0
this_dialog = []
this_dialog_speaker = []
this_dialog_speaker_set = []
this_utt_len = []

X1, X2, X3, Y1, Y2 = [], [], [], [], []

speaker_info = {}
dialog_len = {}
utt_len = {}

for d,utt,slot,speaker,target in tqdm.tqdm(zip(df_test["Dialogue_Id"], df_test["Utterance"], df_test["Persona Slot"], df_test["Speaker"], df_test["Target"])):
    if type(utt) == float:
        this_dialog = []
        this_dialog_speaker = []
        this_dialog_speaker_set = []
        this_utt_len = []
        continue
        
    utt = preprocess_sentence(utt)
    words = utt.split()
    this_utt = []
    for word in words:
        this_utt.append(word2idx[word])
        
    this_utt_len.append(len(this_utt))
    if len(this_utt) < seq_len2:
        for k in range(len(this_utt),seq_len2):
            this_utt.append(word2idx["<pad>"])
    else:
        this_utt = this_utt[len(this_utt)-seq_len2:seq_len2]
    this_dialog.append(this_utt)

    speaker = preprocess_sentence(speaker)
    if speaker not in this_dialog_speaker_set:
        this_dialog_speaker_set.append(speaker)
    idx = this_dialog_speaker_set.index(speaker)
    this_dialog_speaker.append(idx)
    
    if type(slot) != float:
        this_dialog_padded = this_dialog[:]
        this_dialog_speaker_padded = this_dialog_speaker[:]
        
        if len(this_dialog_padded) < seq_len:
            for k in range(len(this_dialog),seq_len):
                this_dialog_padded.append([word2idx["<pad>"]]*seq_len2)
                this_dialog_speaker_padded.append(9)
        else:
            this_dialog_padded = this_dialog_padded[len(this_dialog_padded)-seq_len:seq_len]
            this_dialog_speaker_padded = this_dialog_speaker_padded[len(this_dialog_speaker_padded)-seq_len:seq_len]
            
        X1.append(d_id)
        X2.append(this_dialog_padded)
        X3.append(this_dialog_speaker_padded)
        Y1.append(persona2idx[slot])
        
        target = preprocess_sentence(target)
        if target not in this_dialog_speaker_set:
            this_dialog_speaker_set.append(target)
        idx = this_dialog_speaker_set.index(target)
        Y2.append(idx)
        
        speaker_info[d_id] = this_dialog_speaker[:]
        dialog_len[d_id] = len(this_dialog)
        utt_len[d_id] = this_utt_len[:]
        
        d_id += 1

print("Number of training samples = ",len(X1))
X1 = torch.LongTensor(X1)
X2 = torch.LongTensor(X2)
X3 = torch.LongTensor(X3)
Y1 = torch.LongTensor(Y1)
Y2 = torch.LongTensor(Y2)

print("X1 size -> ",X1.size())
print("X2 size -> ",X2.size())
print("X3 size -> ",X3.size())
print("Y1 size -> ",Y1.size())
print("Y2 size -> ",Y2.size())

my_dataset = data.TensorDataset(X1,X2,X3,Y1,Y2)

with open("data/test_dataset.pt","wb") as f:
    pickle.dump(my_dataset,f)
with open("data/test_dialog_len.pt","wb") as f:
    pickle.dump(dialog_len,f)
with open("data/test_speaker_info.pt","wb") as f:
    pickle.dump(speaker_info,f)
with open("data/test_utt_len.pt","wb") as f:
    pickle.dump(utt_len,f)
#############################################