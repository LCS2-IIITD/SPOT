get_ipython().system('pip install -q --upgrade transformers datasets rouge_score accelerate')
get_ipython().system('pip install -q wandb loguru')


import transformers
import torch
from torch import nn
import numpy as np

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import time
import json
import pandas as pd
from IPython.display import display, HTML
import torch.nn.functional as F
from loguru import logger

import transformers
from huggingface_hub import Repository
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
import math
import random
import os
nltk.download('punkt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = Accelerator()

MODEL_PATH = "facebook/bart-base"
TOKENIZER_PATH = "facebook/bart-base"


BASE_PATH = "/models/"
SAVE_MODEL_PATH = BASE_PATH + "models/model.pt"
SPECIAL_SEPARATOR = "SEP"

TRAIN_FILE_PATH = "data/train.csv"
VAL_FILE_PATH = "data/dev.csv"

max_input_length = 512
max_target_length = 32
batch_size = 16

data_files = {}
data_files["train"] = TRAIN_FILE_PATH
data_files["validation"] = VAL_FILE_PATH

raw_datasets = load_dataset("csv", data_files=data_files)
metric = load_metric("rouge")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def preprocess_function(examples):
    global_enc = [prompt + f" {SPECIAL_SEPARATOR} " + speaker + f" {SPECIAL_SEPARATOR} " + dialog for prompt, speaker, slot, dialog in zip(examples['prompt'], examples['speaker'], examples['slot'], examples['dialog'])]
    model_inputs = tokenizer(global_enc, max_length=max_input_length, padding='max_length', truncation=True)
    model_inputs["global_encoding_input_ids"] = model_inputs["input_ids"]
    model_inputs["global_encoding_attention_mask"] = model_inputs["attention_mask"]

    utterance_enc = tokenizer(examples['utterances'], padding='max_length', max_length=max_input_length, truncation=True)
    model_inputs['utterance_encoding_input_ids'] = utterance_enc['input_ids']
    model_inputs['utterance_encoding_attention_mask'] = utterance_enc['attention_mask']

    target_enc = tokenizer(examples['target_utterances'], padding='max_length', max_length=max_input_length, truncation=True)
    model_inputs['target_encoding_input_ids'] = target_enc['input_ids']
    model_inputs['target_encoding_attention_mask'] = target_enc['attention_mask']

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        outputs = examples['output']
        labels = tokenizer(outputs, padding='max_length', max_length=max_target_length, truncation=True)

    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

with accelerator.main_process_first():
    processed_dataset = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)

train_dataset = processed_dataset["train"]
eval_dataset = processed_dataset["validation"]

bart = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
config = AutoConfig.from_pretrained(MODEL_PATH)

class BART_SP(nn.Module):
    def __init__(self, bart, config, alpha=0.1, q_dim=64, upscale_dim=256):
        super(BART_SP, self).__init__()
        self.bart = bart
        self.config = config

        self.encoder = self.bart.model.encoder
        self.decoder = self.bart.model.decoder
        self.lm_head = self.bart.lm_head

        self.output_attentions = self.config.output_attentions
        self.output_hidden_states = self.config.output_hidden_states
        self.use_cache = False
        self.return_dict = True

        self.alpha = alpha
        self.q_dim = 64
        self.hidden_dim = config.hidden_size

        self.query = nn.Linear(self.hidden_dim, self.q_dim)
        self.key = nn.Linear(self.hidden_dim, self.q_dim)
        self.value = nn.Linear(self.hidden_dim, self.q_dim)

        self.target_downscale = nn.Linear(self.hidden_dim, self.q_dim)
        self.attention_upscale = nn.Linear(self.q_dim, self.hidden_dim)

        self.transform = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, batch):
        global_encoding_input_ids = batch["global_encoding_input_ids"]
        global_encoding_attention_mask = batch["global_encoding_attention_mask"]

        target_encoding_input_ids = batch["target_encoding_input_ids"]
        target_encoding_attention_mask = batch["target_encoding_attention_mask"]

        utterance_encoding_input_ids = batch["utterance_encoding_input_ids"]
        utterance_encoding_attention_mask = batch["utterance_encoding_attention_mask"]

        decoder_input_ids = batch['decoder_input_ids']


        global_encoding = self.encoder(input_ids=global_encoding_input_ids, attention_mask=global_encoding_attention_mask,
                                output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states,
                                return_dict=self.return_dict)[0]
        utterance_encoding = self.encoder(input_ids=utterance_encoding_input_ids, attention_mask=utterance_encoding_attention_mask,
                                output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states,
                                return_dict=self.return_dict)[0]
        target_encoding = self.encoder(input_ids=target_encoding_input_ids, attention_mask=target_encoding_attention_mask,
                                output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states,
                                return_dict=self.return_dict)[0]

        query = self.query(target_encoding)
        key = self.key(utterance_encoding)
        value = self.value(utterance_encoding)

        attention_scores = torch.bmm(query, key.transpose(-2, -1)) / np.sqrt(self.q_dim)
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention = torch.bmm(attention_scores, value)

        down_target = self.target_downscale(target_encoding)
        attention = attention + down_target
        attention = self.attention_upscale(attention)

        output = attention + self.transform(utterance_encoding)
        encoded = self.alpha*output + global_encoding

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoded,
            encoder_attention_mask=None,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            return_dict=self.return_dict,
        )
        logits = self.lm_head(decoder_outputs[0])
        return logits

model = BART_SP(bart, config, alpha=0.4)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=bart, label_pad_token_id=-100, pad_to_multiple_of=None)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
)

eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

weight_decay = 0.01
learning_rate = 8e-5

epochs = 60

gradient_accumulation_steps = 1
lr_scheduler_type = 'linear'
num_warmup_steps = 256
num_beams = None

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)

model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = epochs * num_update_steps_per_epoch

total_batch_size = batch_size * accelerator.num_processes * gradient_accumulation_steps

criterion = nn.CrossEntropyLoss()

sns.set_style("darkgrid")

def plot_metrics():
    clear_output(wait=True)
    plt.figure()
    plt.plot(tracker["steps"], tracker["step_loss"])
    plt.title("Loss vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")

    print()
    plt.figure()
    plt.plot(tracker["epochs"], tracker["train_loss"], label="train")
    plt.plot(tracker["epochs"], tracker["val_loss_epochs"], label="validation")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    print()
    plt.figure()
    # plt.plot(tracker["epochs"], tracker["train_metrics"], label="train")
    plt.plot(tracker["epochs"], tracker["val_metrics_epochs"], label="validation")
    plt.title("Metrics vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()

    plt.show()

def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, accelerator, epochs=3, batch_size=16, gradient_accumulation_steps=1, max_train_steps=None):
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {epochs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        completed_steps = train_epoch(model, train_dataloader, eval_dataloader, optimizer, 
                                      lr_scheduler, config, completed_steps, progress_bar, 
                                      epoch, start_time)
        
        evaluate(model, eval_dataloader, type_="val", suffix="_epochs")

        
        print(f"[INFO] EPOCH {epoch+1} COMPLETE! LOSS={tracker['train_loss'][-1]:.2f}\n")

        if completed_steps >= max_train_steps:
            break

    torch.save(model, SAVE_MODEL_PATH)

def train_epoch(model, train_dataloader, eval_dataloader, optimizer, 
                lr_scheduler, config, completed_steps, progress_bar, 
                epoch, start_time):
    train_loss = 0
    num_steps = 0

    for step, batch in enumerate(train_dataloader):
        lm_logits = model(batch)
        labels = batch["labels"]

        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        train_loss += loss.detach().float()
        num_steps += 1

        tracker["step_loss"].append(loss.item())
        tracker["steps"].append(completed_steps)

        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

        if completed_steps % print_every == 0:
            prev_val_rouge = tracker["val_metrics"][-1] if tracker["val_metrics"] else 0
            evaluate(model, eval_dataloader, type_="val")


    train_loss /= num_steps
    tracker["epochs"].append(epoch)
    tracker["train_loss"].append(train_loss.item() if type(train_loss)!=int else train_loss)

    return completed_steps 

def evaluate(model, dloader, type_="val", compute_loss=True, suffix=""):
    model.eval()

    gen_kwargs = {
        "max_length": max_target_length,
        "num_beams": num_beams}
    
    eval_loss = 0
    num_steps = 0
    exact_matches = []
    partial_matches = []

    for step, batch in enumerate(dloader):
        with torch.no_grad():

            if compute_loss:
                lm_logits = model(batch)
                labels = batch['labels']

                loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                eval_loss += loss.detach().item()

            num_steps += 1

            outputs = model(batch)
            outputs = outputs.softmax(dim=-1)
            generated_tokens = torch.argmax(outputs, dim=-1)
            labels = batch["labels"]

            generated_tokens = generated_tokens.cpu().numpy()
            labels = labels.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]

            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            exact_match = sum([1 if pred.lower()==true.lower() else 0 for pred, true in zip(decoded_preds, decoded_labels)])/len(decoded_preds)
            exact_matches.append(exact_match)

            partial_match = sum([1 if (pred.lower() in true.lower() or true.lower() in pred.lower()) else 0 for pred, true in zip(decoded_preds, decoded_labels)])/len(decoded_preds)
            partial_matches.append(partial_match)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    result = metric.compute(use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {k: round(v, 4) for k, v in result.items()}
    result["exact_match"] = np.mean(exact_matches)
    result["partial_match"] = np.mean(partial_matches)

    if compute_loss:
        eval_loss /= num_steps
        tracker[f"{type_}_loss{suffix}"].append(eval_loss)
    
    tracker[f"{type_}_metrics{suffix}"].append(result["rougeLsum"])
    tracker[f"{type_}_em{suffix}"].append(result['exact_match'])
    tracker[f"{type_}_pm{suffix}"].append(result['partial_match'])

    model.train()

tracker = {
    "step_loss": [],
    "steps": [],
    "train_loss": [],
    "train_metrics": [],
    "epochs": [],
    "val_loss": [],
    "val_metrics": [], # {'rouge1': 70.7396, 'rouge2': 28.9791, 'rougeL': 51.875, 'rougeLsum': 58.2712}
    "val_loss_epochs": [],
    "val_metrics_epochs": [],
    "val_em": [],
    "val_pm": [],
    "val_em_epochs": [], 
    "val_pm_epochs": []
}

print_every = 200
save_every = 400

train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, accelerator, epochs=epochs, batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps, max_train_steps=max_train_steps)

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
    torch.save(model, SAVE_MODEL_PATH+"model.pt")