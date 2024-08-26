import pickle
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import math
import os

with open("data/word2idx.pt","rb") as f:
    word2idx = pickle.load(f)
with open("data/weight_matrix.pt","rb") as f:
    weight_matrix = pickle.load(f)
with open("data/idx2persona.pt","rb") as f:
    idx2personaSlot = pickle.load(f)

batch_size = 8
seq_len = 33
emb_size = 200
hidden_size = 300
batch_first = True
model_name = "phase2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
class attention(nn.Module):
    def __init__(self,in_size,hid_size,out_size):
        super(attention,self).__init__()
        
        self.make_query = nn.Linear(in_size,hid_size)
        self.make_key = nn.Linear(in_size,hid_size)
        self.make_value = nn.Linear(in_size,hid_size)
        
        self.make_output = nn.Linear(hid_size,out_size)
        
        self.normalise_factor = hid_size**(1/2)
        
    def forward(self,query,key,value=None):
        if value is None:
            value = key
            
        key = self.make_key(key)
        query = self.make_query(query)
        value = self.make_value(value)
        
        score = torch.mm(query,key.permute(1,0))/self.normalise_factor
        score = F.softmax(score,-1)
        output = self.make_output(torch.mm(score,value))
        
        return output,score
        
class utterance_embedding(nn.Module):
    def __init__(self,embed_size):
        super(utterance_embedding,self).__init__()
        self.emb_size = embed_size
        self.emb, num_embeddings, emb_dim = create_emb_layer(weight_matrix)
        self.u_gru = nn.GRU(emb_dim,int(emb_dim/2),bidirectional=True,batch_first=True)
        self.attn = attention(emb_dim,emb_dim,emb_dim)
        self.linear = nn.Linear(emb_dim,embed_size)
        
    def forward(self,d_ids,x,utt_len):
        emb = self.emb(x)
        output = torch.zeros(emb.size()[0],emb.size()[1],self.emb_size)
        
        for i,d in enumerate(emb):
            d_id = d_ids[i].item()
            u_len = utt_len[d_id]
            for j,ul in enumerate(u_len):
                u = torch.unsqueeze(d[j][:ul],0)
                op,_ = self.u_gru(u)
                op = torch.squeeze(op,0)
                op,_ = self.attn(op,op)
                op = torch.unsqueeze(op,0)
                op = self.linear(torch.squeeze(op,0)[-1])
                output[i][j] = op

        return output
        
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.emb = utterance_embedding(emb_size)

        emsize = emb_size
        nhid = 200
        nlayers = 6
        nhead = 2
        dropout = 0.5

        self.d_gru = TransformerModel(emsize, nhead, nhid, nlayers, device, dropout)        
        self.fc1 = nn.Linear(emb_size,1)        
        self.sigmoid = nn.Sigmoid()

        self.s_gru_list = []
        for i in range(10):
            self.s_gru_list.append(TransformerModel(emsize, nhead, nhid, nlayers, device, dropout))
        self.s_grus = nn.ModuleList(self.s_gru_list)
        
        self.speaker_attn = attention(emb_size,emb_size,emb_size)
        self.context_attn = attention(emb_size,emb_size,emb_size)
            
        self.final_attn = attention(emb_size,emb_size,emb_size)
        
        self.fcgru = nn.GRU(emb_size,int(emb_size/2),bidirectional=True,batch_first=True)
        self.fc3 = nn.Linear(2*emb_size,100)
        
        self.classifier = nn.Linear(100,len(idx2personaSlot))
        
    def forward(self,d_ids,dialog_len,utt_len,dialog,speakers,speaker_info):
        emb = F.relu(self.emb(d_ids,dialog,utt_len)).to(device)
        op = torch.empty(emb.size()[0],100).to(device)
        
        for d in range(emb.size()[0]):
            d_id = d_ids[d].item()
            d_len = dialog_len[d_id]

            context_rep = emb[d][:d_len-1]
            target_rep = emb[d][d_len-1]

            num_sp = len(set(speaker_info[d_id]))
            speaker_ip = [[] for _ in range(num_sp)]
            
            speaker_op = torch.zeros(num_sp,emb_size).to(device)
            for i,u in enumerate(speaker_info[d_id]):
                speaker_ip[u].append(torch.unsqueeze(emb[d][i],0))
                            
            for i,si in enumerate(speaker_ip):
                si = torch.cat(si,0).to(device)
                si = torch.unsqueeze(si,0)
                speaker_op[i] = torch.squeeze(self.s_grus[i](si,si)[0],0)[-1]
            
            dialog_op = self.d_gru(torch.unsqueeze(emb[d][:d_len],0),torch.unsqueeze(emb[d][:d_len],0))[0]
            tmp_op = self.fc1(dialog_op)
            sigmoid_op = self.sigmoid(tmp_op).repeat(1,1,emb_size)
            sigmoid_op = sigmoid_op * dialog_op
            tmp_op = sigmoid_op
            
            context_rep = dialog_op[:-1]
            target_rep = torch.unsqueeze(dialog_op[-1],0)

            sar,_ = self.speaker_attn(target_rep,speaker_op)            
            car,_ = self.context_attn(target_rep,context_rep)
            sarcar = torch.cat([sar,car],0)
            
            fop,_ = self.final_attn(target_rep,sarcar)
            
            tmp_op = torch.unsqueeze(self.fcgru(tmp_op)[0][0][-1],0)
            tmp_op = self.fc3(torch.cat([tmp_op,fop],-1))
            op[d] = tmp_op[0]
            
        output = self.classifier(op)
            
        return op,output