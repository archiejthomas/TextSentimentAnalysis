#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 14:47:49 2026

@author: archiethomas

Contains all the functions required to load the Financial Phrasebank Datasets 
and parse them as desired. 

Also loads in the word2vec word embeddings
"""

import os
import pandas as pd 
import gensim.downloader as api
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    

def load_embeddings():
    print("Loading Word2Vec Embeddings")
    model = api.load("word2vec-google-news-300")
    return model




def split_data(split, df):
    length = df.shape[0]
    n = int(np.floor(length*split))
    train_index = np.random.choice(length, size=n, replace=False)
    test_index = np.setdiff1d(np.arange(length), train_index)
    
    Z_train = df.iloc[train_index,]
    Z_test = df.iloc[test_index,]
    
    return {
        'Train': Z_train,
        'Test': Z_test
        }

