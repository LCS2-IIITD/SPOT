from model import *
import my_model as mm
from init_parameter import *
from dataloader import *
from pretrain import *
from util import *
from loss import *
from torch.utils import data as tu_data

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = 'logs/train/' + TIMESTAMP
test_log_dir = 'logs/test/'   + TIMESTAMP

batch_size = 8
seq_len = 33
emb_size = 300
hidden_size = 300
batch_first = True
model_name = "phase2"
    
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

class ModelManager:
    
    def __init__(self, args, data, pretrained_model, model2, combine1, combine2):
        
        self.model = pretrained_model
        self.model2 = model2
        self.combine1 = combine1
        self.combine2 = combine2

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id     
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model2.to(self.device)
        self.combine1.to(self.device)
        self.combine2.to(self.device)
        
        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def open_classify(self, features):

        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id

        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        if mode == 'eval':
            dataloader = data.eval_dataloader
            data_iter = iter(data_iter_dev)
        elif mode == 'test':
            dataloader = data.test_dataloader
            data_iter = iter(data_iter_test)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            
            batch2 = tuple(t.to(self.device) for t in data_iter.next())
            d_ids, dialog, speakers, y, _ = batch2
                
            with torch.set_grad_enabled(False):
                pooled_output, _ = self.model(input_ids, segment_ids, input_mask)
                
                if mode == 'eval':
                    outputs,_ = self.model2(d_ids, dev_dialog_len, dev_utt_len, dialog, speakers, dev_speaker_info)
                if mode == 'test':
                    outputs,_ = self.model2(d_ids, test_dialog_len, test_utt_len, dialog, speakers, test_speaker_info)
                
                outputs = outputs[:pooled_output.size()[0]]
                ip = torch.cat([pooled_output,outputs],-1)
                op = self.combine1(ip)
                op2 = self.combine2(op)
                
                preds = self.open_classify(op)

                total_labels = torch.cat((total_labels,label_ids))
                total_preds = torch.cat((total_preds, preds))
        
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        self.predictions = list([data.label_list[idx] for idx in y_pred])
        self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            return eval_score

        elif mode == 'test':
            
            cm = confusion_matrix(y_true,y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)
            
            print("Confusion Matrix-\n",cm)


    def train(self, args, data):     
        
        criterion_boundary = BoundaryLoss(num_labels = data.num_labels, feat_dim = 768+100)
        self.delta = F.softplus(criterion_boundary.delta)
        delta_last = copy.deepcopy(self.delta.detach())
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr = args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        best_model = None
        wait = 0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            data_iter = iter(data_iter_train)
            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                batch2 = tuple(t.to(self.device) for t in data_iter.next())
                d_ids, dialog, speakers, y, _ = batch2
                
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    outputs,_ = self.model2(d_ids, train_dialog_len, train_utt_len, dialog, speakers, train_speaker_info)
                    
                    outputs = outputs[:features.size()[0]]
                    ip = torch.cat([features,outputs],-1)
                    op = self.combine1(ip)
                    op2 = self.combine2(op)
                    
                    loss, self.delta = criterion_boundary(op, self.centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    tr_loss += loss.item()
                    
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            print('train_loss',loss)
            
            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score',eval_score)
            
            if eval_score >= self.best_eval_score:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_eval_score = eval_score
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break
        
        self.model = best_model 

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, 768+100).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            data_iter = iter(data_iter_train)
            for batch in data.train_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                
                batch2 = tuple(t.to(self.device) for t in data_iter.next())
                d_ids, dialog, speakers, y, _ = batch2
                
                features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                outputs,_ = self.model2(d_ids, train_dialog_len, train_utt_len, dialog, speakers, train_speaker_info)
                
                outputs = outputs[:features.size()[0]]
                ip = torch.cat([features,outputs],-1)
                op = self.combine1(ip)
                op2 = self.combine2(op)                
                
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += op[i]
                
        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()
        
        return centroids

    def restore_model(self, args):
        output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        self.model.load_state_dict(torch.load(output_model_file))
    
    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
        vars_dict = {k:v for k,v in zip(names, var) }
        results = dict(self.test_results,**vars_dict)
        keys = list(results.keys())
        values = list(results.values())
        
        np.save(os.path.join(args.save_results_path, 'centroids.npy'), self.centroids.detach().cpu().numpy())
        np.save(os.path.join(args.save_results_path, 'deltas.npy'), self.delta.detach().cpu().numpy())

        file_name = 'results'  + '.csv'
        results_path = os.path.join(args.save_results_path, file_name)
        
        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori,columns = keys)
            df1.to_csv(results_path,index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results,index=[1])
            df1 = df1.append(new,ignore_index=True)
            df1.to_csv(results_path,index=False)
        data_diagram = pd.read_csv(results_path)
        
        print('test_results', data_diagram)



if __name__ == '__main__':
    
    print('Data and Parameters Initialization...')
    parser = init_model()
    args = parser.parse_args()
    data = Data(args)

    print('Pre-training begin...')
    manager_p = PretrainModelManager(args, data)
    manager_p.train(args, data)
    print('Pre-training finished!')
    
    manager = ModelManager(args, data, manager_p.model, manager_p.model2, manager_p.combine1, manager_p.combine2)
    print('Training begin...')
    manager.train(args, data)
    print('Training finished!')
    
    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")  
    print('Evaluation finished!')