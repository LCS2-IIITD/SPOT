from model import *
from dataloader import *
from util import *
import pickle
import my_model as mm
from torch.utils import data as tu_data
from collections import Counter

batch_size = 8
seq_len = 33
emb_size = 200
hidden_size = 300
batch_first = True
model_name = "phase2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/idx2persona.pt","rb") as f:
    idx2personaSlot = pickle.load(f)
with open("data/persona2idx.pt","rb") as f:
    personaSlot2idx = pickle.load(f)
        
with open("data/idx2word.pt","rb") as f:
    idx2word = pickle.load(f)
with open("data/word2idx.pt","rb") as f:
    word2idx = pickle.load(f)
    
with open("data/weight_matrix.pt","rb") as f:
    weight_matrix = pickle.load(f)

with open("data/train_dataset.pt","rb") as f:
    train_dataset = pickle.load(f)
with open("data/train_dialog_len.pt","rb") as f:
    train_dialog_len = pickle.load(f)
with open("data/train_speaker_info.pt","rb") as f:
    train_speaker_info = pickle.load(f)
with open("data/train_utt_len.pt","rb") as f:
    train_utt_len = pickle.load(f)
    
with open("data/dev_dataset.pt","rb") as f:
    dev_dataset = pickle.load(f)
with open("data/dev_dialog_len.pt","rb") as f:
    dev_dialog_len = pickle.load(f)
with open("data/dev_speaker_info.pt","rb") as f:
    dev_speaker_info = pickle.load(f)
with open("data/dev_utt_len.pt","rb") as f:
    dev_utt_len = pickle.load(f)
    
with open("data/test_dataset.pt","rb") as f:
    test_dataset = pickle.load(f)
with open("data/test_dialog_len.pt","rb") as f:
    test_dialog_len = pickle.load(f)
with open("data/test_speaker_info.pt","rb") as f:
    test_speaker_info = pickle.load(f)
with open("data/test_utt_len.pt","rb") as f:
    test_utt_len = pickle.load(f)

data_iter_train = tu_data.DataLoader(train_dataset,batch_size=batch_size, shuffle=False)
data_iter_dev = tu_data.DataLoader(dev_dataset,batch_size=batch_size, shuffle=False)
data_iter_test = tu_data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

class PretrainModelManager:
    
    def __init__(self, args, data):

        self.model = BertForModel.from_pretrained(args.bert_model, cache_dir = "", num_labels = data.num_labels)
        if args.freeze_bert_parameters:
            for name, param in self.model.bert.named_parameters():  
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
                    
        self.model2 = mm.model()
        self.combine1 = nn.Linear(768+100, 768+100)
        self.combine2 = nn.Linear(768+100,5)
                    
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id           
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model2.to(self.device)
        self.combine1.to(self.device)
        self.combine2.to(self.device)
        
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.num_train_optimization_steps = int(len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        
        self.optimizer = self.get_optimizer(args)
        
        self.best_eval_score = 0

    def eval(self, args, data, this_mode='dev'):
        
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)
        
        if this_mode == 'dev':
            data_iter = iter(data_iter_dev)
        if this_mode == 'test':
            data_iter = iter(data_iter_test)
        
        ## To calculate weights
        all_labels = []
        for batch in data.train_dataloader:
            _, _, _, label_ids = batch
            all_labels.extend(label_ids.numpy().tolist())
        label_dict = Counter(all_labels)
        weights = [0]*len(label_dict.keys())
        max_value = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)[0][1]
        for k,v in label_dict.items():
            weights[k] = max_value/v
        weights = torch.tensor(weights).to(device)
        print("Eval weights -> ",weights)
            
        for batch in tqdm(data.eval_dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            batch2 = tuple(t.to(self.device) for t in data_iter.next())
            d_ids, dialog, speakers, y, _ = batch2
                
            with torch.set_grad_enabled(False):
                pooled_op, logits = self.model(input_ids, segment_ids, input_mask, mode = 'eval')
                
                if this_mode == 'dev':
                    outputs,_ = self.model2(d_ids, dev_dialog_len, dev_utt_len, dialog, speakers, dev_speaker_info)
                if this_mode == 'test':
                    outputs,_ = self.model2(d_ids, test_dialog_len, test_utt_len, dialog, speakers, test_speaker_info)
                    
                outputs = outputs[:pooled_op.size()[0]]
                ip = torch.cat([pooled_op,outputs],-1)
                op = self.combine2(self.combine1(ip))

                loss = nn.CrossEntropyLoss(weight = weights)(op,label_ids)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_logits = torch.cat((total_logits, logits))
        
        total_probs, total_preds = F.softmax(total_logits.detach(), dim=1).max(dim = 1)
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        acc = round(accuracy_score(y_true, y_pred) * 100, 2)

        return acc


    def train(self, args, data):    

        wait = 0
        best_model = None
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            data_iter = iter(data_iter_train)
            
            ## To calculate weights
            all_labels = []
            for batch in data.train_dataloader:
                _, _, _, label_ids = batch
                all_labels.extend(label_ids.numpy().tolist())
            label_dict = Counter(all_labels)
            weights = [0]*len(label_dict.keys())
            max_value = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)[0][1]
            for k,v in label_dict.items():
                weights[k] = max_value/v
            weights = torch.tensor(weights).to(device)
            print("Train weights -> ",weights)
                
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                batch2 = tuple(t.to(self.device) for t in data_iter.next())
                d_ids, dialog, speakers, y, _ = batch2
                
                with torch.set_grad_enabled(True):
                    pooled_op, logits = self.model(input_ids, segment_ids, input_mask, label_ids, mode = "train")
                    outputs,_ = self.model2(d_ids, train_dialog_len, train_utt_len, dialog, speakers, train_speaker_info)
                    outputs = outputs[:pooled_op.size()[0]]
                    
                    ip = torch.cat([pooled_op,outputs],-1)
                    op = self.combine2(self.combine1(ip))
                    
                    loss = nn.CrossEntropyLoss(weight = weights)(op,label_ids)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
            
            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.eval(args, data)
            print('eval_score',eval_score)
            
            if eval_score > self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
                
        self.model = best_model
        if args.save_model:
            self.save_model(args)

    def get_optimizer(self, args):
        
        print("self -> ",self)

        param_optimizer = list(self.model.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        other_param_optimizer = list(self.model2.parameters())
        other_param_optimizer.extend(list(self.combine1.parameters()))
        other_param_optimizer.extend(list(self.combine2.parameters()))
        
        optimizer_grouped_parameters.extend([
            {'params': [p for p in other_param_optimizer], 'weight_decay': 0.01}
        ])
        
        optimizer = BertAdam(optimizer_grouped_parameters,
                         lr = args.lr,
                         warmup = args.warmup_proportion,
                         t_total = self.num_train_optimization_steps)   
        return optimizer
    
    def save_model(self, args):

        if not os.path.exists(args.pretrain_dir):
            os.makedirs(args.pretrain_dir)
        self.save_model = self.model.module if hasattr(self.model, 'module') else self.model  

        model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        model_config_file = os.path.join(args.pretrain_dir, CONFIG_NAME)
        torch.save(self.save_model.state_dict(), model_file)
        with open(model_config_file, "w") as f:
            f.write(self.save_model.config.to_json_string())
            
        torch.save(self.model2.state_dict(), "./models/model2_{}_weight_state_dict.pth".format(model_name))
        torch.save(self.combine1.state_dict(), "./models/combine1_{}_weight_state_dict.pth".format(model_name))
        torch.save(self.combine2.state_dict(), "./models/combine2_{}_weight_state_dict.pth".format(model_name))

