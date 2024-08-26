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
import math

from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = [9.294501397949674, 1.0, 0]
print('weights- ',weights)
batch_size = 8
emb_size = 200
hidden_size = 200
batch_first = True
model_name = "phase1"

#############################################

with open("./data/idx2persona.p",'rb') as f:
    idx2persona = pickle.load(f)
with open("./data/persona2idx.p",'rb') as f:
    persona2idx = pickle.load(f)
    
with open("./data/idx2word.p",'rb') as f:
    idx2word = pickle.load(f)
with open("./data/word2idx.p",'rb') as f:
    word2idx = pickle.load(f)
    
with open("./data/weight_matrix.p",'rb') as f:
    weight_matrix = pickle.load(f)
    
with open("./data/data_iter_train.p",'rb') as f:
    data_iter_train = pickle.load(f)
    
with open("./data/data_iter_dev.p",'rb') as f:
    data_iter_dev = pickle.load(f)
    
with open("./data/data_iter_test.p",'rb') as f:
    data_iter_test = pickle.load(f)
    
with open("./data/X_train_utt_len.p",'rb') as f:
    X_train_utt_len = pickle.load(f)
    
with open("./data/X_dev_utt_len.p",'rb') as f:
    X_dev_utt_len = pickle.load(f)
    
with open("./data/X_test_utt_len.p",'rb') as f:
    X_test_utt_len = pickle.load(f)
    
total_persona = len(idx2persona)
#############################################

##Source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=word2idx["<pad>"])
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, device, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

        self.init_weights()
        self.device = device

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        
    def forward(self, src, src_emb):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = src_emb * math.sqrt(self.ninp)        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        
        return output
    
class utterance_emb(nn.Module):
    def __init__(self):
        super(utterance_emb,self).__init__()
        self.emb, num_embeddings, emb_dim = create_emb_layer(weight_matrix)
        
        emsize = emb_dim
        nhid = 768
        nlayers = 6
        nhead = 2
        dropout = 0.2

        self.tr = TransformerModel(emsize, nhead, nhid, nlayers, device, dropout)
        
        self.decoder = nn.Linear(14000,768)
        
    def forward(self,x):        
        emb = self.emb(x)        
        op = torch.zeros((emb.size()[0],emb.size()[1],768)).to(device)
        for b in range(op.size()[0]):
            for s in range(op.size()[1]):
                ip = torch.unsqueeze(emb[b][s],0)
                tr_op = self.tr(ip,ip)
                tr_op = torch.flatten(tr_op)
                op[b][s] = self.decoder(tr_op)

        return op
    
class persona_classifier(nn.Module):
    def __init__(self):
        super(persona_classifier,self).__init__()
        self.utt_emb = utterance_emb()
        
        emsize = 768
        nhid = 768 
        nlayers = 6 
        nhead = 2 
        dropout = 0.2 

        self.tr = TransformerModel(emsize, nhead, nhid, nlayers, device, dropout)        
        self.classifier = nn.Linear(768,total_persona)
        
    def forward(self,x,sp):
        op1 = self.utt_emb(x)
        op2 = self.tr(op1,op1)
        op4 = self.classifier(op2)
        
        return op4
    
def train(model,train_data_loader,epochs=100,log_step=2):
    history1 = hl.History()
    canvas1 = hl.Canvas()
    
    history2 = hl.History()
    canvas2 = hl.Canvas()
    
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    max_f1 = 0.0
    
    write_file = open("logs/confusion_matrix_{}.txt".format(model_name),"w")
    write_file2 = open("logs/confusion_matrix_validate_{}.txt".format(model_name),"w")
    
    for epoch in tqdm.tqdm(range(epochs)):
        print("\n\n-------Epoch {}-------\n\n".format(epoch+1))
        model.train()
        
        y_true = []
        y_pred = []
        
        avg_loss = 0
        for i_batch, sample_batched in tqdm.tqdm(enumerate(train_data_loader)):
            loss = 0
            d_ids = sample_batched[0].to(device)
            inputs = sample_batched[1].to(device)
            targets = sample_batched[2].to(device)
            speaker = sample_batched[3].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs,speaker).to(device)
            pred_intents = torch.argmax(F.softmax(outputs,-1),-1)

            for b in range(outputs.size()[0]):
                d_id = d_ids[b].item()
                for s in range(X_train_utt_len[d_id][0]):
                    y1 = targets[b][s]
                    y2 = pred_intents[b][s]
                    y_true.append(y1.item())
                    y_pred.append(y2.item())

            for b in range(outputs.size()[0]):
                d_id = d_ids[b].item()
                loss += criterion(outputs[b][:X_train_utt_len[d_id][0]],targets[b][:X_train_utt_len[d_id][0]])
            loss /= b
            
            loss.backward()            
            optimizer.step()
            avg_loss += loss
        avg_loss /= len(train_data_loader)
            
        pr,re,f1,_ = precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
        
        print(persona2idx,idx2persona)
        print("Precision ==> ",pr)
        print("Recall ==> ",re)
        print("F1-score ==> ",f1)
        
        print("Classification Report ==>\n",classification_report(y_true,y_pred))
        
        cls_pr,cls_re,cls_f1,_ = precision_recall_fscore_support(y_true,y_pred,zero_division=0)
        conf_mat = confusion_matrix(y_true,y_pred)
        
        write_file.write("After epoch {} detailed metrics are:\n".format(epoch+1))
        write_file.write("Class wise metric:-\nIntent\tPrecision\tRecall\tF1 Score\n")
        for index in range(len(cls_f1)):
            write_file.write("{}\t{}\t{}\t{}\n".format(idx2persona[index],cls_pr[index],cls_re[index],cls_f1[index]))

        write_file.write("\n\nConfusion Matrices:-\n{}\n".format(conf_mat))

        f1_1,v_loss,history2,canvas2 = validate(model,data_iter_dev,X_dev_utt_len,write_file2,epoch,history2,canvas2)

        if f1_1 > max_f1:
            print("Best model. Saving it...")
            max_f1 = f1_1
            if os.path.exists("./model/model_{}_weight_state_dict.pth".format(model_name)):
                os.remove("./model/model_{}_weight_state_dict.pth".format(model_name))
            torch.save(model.state_dict(), "./model_{}_weight_state_dict.pth".format(model_name))

        if epoch % log_step == 0:
            history1.log(epoch,loss=avg_loss,accuracy=cls_f1[persona2idx['yes']])
            canvas1.draw_plot([history1["loss"], history1["accuracy"]])
            canvas1.save("logs/training_graph_{}.png".format(model_name))

def validate(model,test_data_loader,test_utt_len,write_file,epoch,history2,canvas2,log_step=2):
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
   
    print("\n\n-------Validation-------\n\n")
    
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        val_loss = 0
        
        for i_batch, sample_batched in tqdm.tqdm(enumerate(test_data_loader)):
            loss = 0
            d_ids = sample_batched[0].to(device)
            inputs = sample_batched[1].to(device)
            targets = sample_batched[2].to(device)
            speaker = sample_batched[3].to(device)

            outputs = model(inputs,speaker).to(device)
            pred_intents = torch.argmax(F.softmax(outputs,-1),-1)

            for b in range(outputs.size()[0]):
                d_id = d_ids[b].item()
                for s in range(test_utt_len[d_id][0]):
                    y1 = targets[b][s]
                    y2 = pred_intents[b][s]
                    y_true.append(y1.item())
                    y_pred.append(y2.item())

            for b in range(outputs.size()[0]):
                d_id = d_ids[b].item()
                loss += criterion(outputs[b][:test_utt_len[d_id][0]],targets[b][:test_utt_len[d_id][0]])
            loss /= b
            val_loss += loss
        val_loss /= len(test_data_loader)

    pr,re,f1,_ = precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    
    print(persona2idx,idx2persona)
    print("Precision ==> ",pr)
    print("Recall ==> ",re)
    print("F1-score ==> ",f1)
    
    print("Classification Report ==>\n",classification_report(y_true,y_pred))
        
    cls_pr,cls_re,cls_f1,_ = precision_recall_fscore_support(y_true,y_pred,zero_division=0)
    conf_mat = confusion_matrix(y_true,y_pred)

    write_file.write("After epoch {} detailed metrics are:\n".format(epoch+1))
    write_file.write("Class wise metric:-\nIntent\tPrecision\tRecall\tF1 Score\n")
    for index in range(len(cls_f1)):
        write_file.write("{}\t{}\t{}\t{}\n".format(idx2persona[index],cls_pr[index],cls_re[index],cls_f1[index]))

    write_file.write("\n\nConfusion Matrices:-\n{}\n".format(conf_mat))

    if epoch % log_step == 0:
        history2.log(epoch,loss=val_loss,accuracy=cls_f1[persona2idx['yes']])
        canvas2.draw_plot([history2["loss"], history2["accuracy"]])
        canvas2.save("logs/validation_graph_{}.png".format(model_name))
                
    return cls_f1[0],val_loss,history2,canvas2

model = persona_classifier().to(device)
print(model)
train(model,data_iter_train,epochs=100)