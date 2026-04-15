#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 15:08:13 2026

@author: archiethomas

THIS REQUIRES A DIFFERENT ENVIRONEMNT TO THE REST OF THE PROJECT

"""


import numpy as np

import random

from transformers import BertTokenizer, BertForSequenceClassification

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

from transformers import get_linear_schedule_with_warmup

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from src.my_project.data_loader import load_data

import os
import pandas as pd 
import gensim.downloader as api
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""-------------------- Loading data and tokenizer ------------------------"""


tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')

device = torch.device("cpu")

data = load_data(50)
df = data['data']

sentences = df.iloc[:, 0].values
y = df.iloc[:, 2].values


"""------------------------- Functions -----------------------------------"""


def load_data(level = 100, verbose = True):
    
    base_path = "data/FinancialPhraseBank"
    
    FPB_map = {
        50: "Sentences_50Agree.txt",
        66: "Sentences_66Agree.txt",
        75: "Sentences_75Agree.txt",
        100: "Sentences_AllAgree.txt",
        }

    file_path = os.path.join(base_path, FPB_map[level])
    
    sentences, labels = [], []

    with open(file_path, "r", encoding="latin-1") as f:
        for line in f:
            if "@" in line:
                text, lbl = line.strip().rsplit("@", 1)
                sentences.append(text.strip())
                labels.append(lbl.strip())
    
    

    df = pd.DataFrame({"sentence": sentences, "label": labels})
    neutrals = df.query('label == "neutral"')
    positives = df.query('label == "positive"')
    negatives = df.query('label == "negative"')
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df["label"])
    
    df["y"] = y_encoded
    
    if verbose:
        print("---------------------\n", level, " Agree","\nTotal:", df.shape[0], 
              "\nPositives:", positives.shape[0],
              "\nNegatives:", negatives.shape[0],
              "\nNeutrals:", neutrals.shape[0],
              "\n---------------------")
        
    data = {
        "data": df,
        "Neu": neutrals,
        "Pos": positives,
        "Neg": negatives
        }
    
    return data
    

def tokenize_data(sentences, tokenizer, max_length=81):
    encodings = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encodings

def make_dataloader(encodings, labels, batch_size=16, shuffle=False):
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, optimiser, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        optimiser.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
        optimiser.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)



"""------------------------- Train and Evaluate ----------------------------"""


scores = []
f1_scores = []
all_preds = []
all_true = []
#seeds = [354, 67, 42, 6, 93]
seeds = [354]
j = 0

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for train_idx, test_idx in outer_cv.split(sentences, y):
        j += 1
        print(f"{j} out of 5")

        sentences_train = sentences[train_idx]
        sentences_test = sentences[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        
        train_encodings = tokenize_data(sentences_train, tokenizer)
        test_encodings = tokenize_data(sentences_test,  tokenizer)

        
        train_loader = make_dataloader(train_encodings, y_train, batch_size=16, shuffle=True)
        test_loader = make_dataloader(test_encodings,  y_test,  batch_size=16, shuffle=False)


        model = BertForSequenceClassification.from_pretrained(
            'ProsusAI/finbert',
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        model.to(device)


        optimiser = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        total_steps = len(train_loader) * 3   
        scheduler = get_linear_schedule_with_warmup(
            optimiser,
            num_warmup_steps=int(0.1 * total_steps),  
            num_training_steps=total_steps
        )

        
        for epoch in range(3):
            loss = train_epoch(model, train_loader, optimiser, scheduler, device)
    


        y_pred, _ = evaluate(model, test_loader, device)

        acc = np.mean(y_pred == y_test)
        scores.append(acc)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
        all_preds.extend(y_pred)
        all_true.extend(y_test)




# ── Summary metrics ───────────────────────────────────────────────────────────
mean_acc = np.mean(scores)
std_acc  = np.std(scores)
mean_f1  = np.mean(f1_scores)
std_f1   = np.std(f1_scores)

print("Test Accuracy:", [round(mean_acc*100, 2), round(std_acc*100, 2)])
print("F1 Score:",      [round(mean_f1*100, 2), round(std_f1*100, 2)])

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Negative', 'Neutral', 'Positive']
)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_finbert.png', dpi=300, bbox_inches='tight')
plt.show()


model = BertForSequenceClassification.from_pretrained(
    'ProsusAI/finbert',
    num_labels=3,
    ignore_mismatched_sizes=True
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable:        {total_params - trainable_params:,}")
