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

weights = [1.0, 1.0, 1.0]
print('weights- ',weights)
batch_size = 8
seq_len = 30
seq_len2 = 15
seq2_len = 10
emb_size = 200
hidden_size = 200
batch_first = True
model_name = "phase1"

#############################################

with open("./data/idx2persona.p",'rb') as f:
    idx2persona = pickle.load(f)
with open("./data/persona2idx.p",'rb') as f:
    persona2idx = pickle.load(f)
    
data_path = './data/data_iter_train_{}_up.csv'.format(model_name)
dev_path = './data/pretrainedREP-dev-{}.csv'.format(model_name)
test_path = './data/pretrainedREP-test-{}.csv'.format(model_name)

total_persona = len(idx2persona)

train_df = pd.read_csv(data_path)
dev_df = pd.read_csv(dev_path)
test_df = pd.read_csv(test_path)
    
Y = torch.tensor(train_df['768'].values.astype(np.float32))
X = torch.tensor(train_df.drop('768', axis = 1).values.astype(np.float32))
print("X size -> ",X.size())
print("Y size -> ",Y.size())

my_dataset = data.TensorDataset(X,Y)
data_iter_train = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)

Y = torch.tensor(dev_df['768'].values.astype(np.float32))
X = torch.tensor(dev_df.drop('768', axis = 1).values.astype(np.float32))
print("X size -> ",X.size())
print("Y size -> ",Y.size())

my_dataset = data.TensorDataset(X,Y)
data_iter_dev = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)

Y = torch.tensor(test_df['768'].values.astype(np.float32))
X = torch.tensor(test_df.drop('768', axis = 1).values.astype(np.float32))
print("X size -> ",X.size())
print("Y size -> ",Y.size())

my_dataset = data.TensorDataset(X,Y)
data_iter_test = data.DataLoader(my_dataset,batch_size=batch_size, shuffle=True)

#############################################
    
class persona_classifier(nn.Module):
    def __init__(self):
        super(persona_classifier,self).__init__()
        
        self.linear1 = nn.Linear(768,768)
        self.dropout1 = nn.Dropout()
        
        self.linear2 = nn.Linear(768,768)
        self.dropout2 = nn.Dropout()
        
        self.linear3 = nn.Linear(768,768)
        self.dropout3 = nn.Dropout()
        
        self.linear4 = nn.Linear(768,total_persona)
        self.dropout4 = nn.Dropout()
        
    def forward(self,x):
        op1 = self.linear1(x)
        op1 = self.dropout1(op1)
        
        op2 = self.linear2(op1)
        op2 = self.dropout2(op2)
        
        op3 = self.linear3(op2)
        op3 = self.dropout3(op3)
        
        op4 = self.linear4(op3)
        op4 = self.dropout4(op4)
        
        return op4
    
def train(model,train_data_loader,epochs=100,log_step=2):
    history1 = hl.History()
    canvas1 = hl.Canvas()
    
    history2 = hl.History()
    canvas2 = hl.Canvas()
    
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    max_f1 = 0
    
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
            inputs = sample_batched[0].to(device)
            targets = sample_batched[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs).to(device)
            
            pred_intents = torch.argmax(F.softmax(outputs,-1),-1)

            for b in range(outputs.size()[0]):
                y1 = targets[b]
                y2 = pred_intents[b]
                y_true.append(y1.item())
                y_pred.append(y2.item())

            loss = criterion(outputs,targets.long())
            
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

        f1_1,v_loss,history2,canvas2 = validate(model,data_iter_dev,write_file2,epoch,history2,canvas2)

        if f1_1 > max_f1:
            max_f1_1 = f1_1
            if os.path.exists("./model/model/model_{}_weight_state_dict.pth".format(model_name)):
                os.remove("./model/model/model_{}_weight_state_dict.pth".format(model_name))
            torch.save(model.state_dict(), "./model/model_{}_weight_state_dict.pth".format(model_name))

        if epoch % log_step == 0:
            history1.log(epoch,loss=avg_loss,accuracy=cls_f1[persona2idx['yes']])
            canvas1.draw_plot([history1["loss"], history1["accuracy"]])
            canvas1.save("logs/training_graph_{}.png".format(model_name))

def validate(model,test_data_loader,write_file,epoch,history2,canvas2,log_step=2):
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
            inputs = sample_batched[0].to(device)
            targets = sample_batched[1].to(device)
            
            outputs = model(inputs).to(device)
            
            pred_intents = torch.argmax(F.softmax(outputs,-1),-1)

            for b in range(outputs.size()[0]):
                y1 = targets[b]
                y2 = pred_intents[b]
                y_true.append(y1.item())
                y_pred.append(y2.item())

            loss += criterion(outputs,targets.long())
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
        
    return cls_f1[1],val_loss,history2,canvas2

model = persona_classifier().to(device)
print(model)
train(model,data_iter_train,epochs=100)