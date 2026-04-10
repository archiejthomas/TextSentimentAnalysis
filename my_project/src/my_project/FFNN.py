# -*- coding: utf-8 -*-

from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings


import numpy as np

from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import random

"""----------------------------  Data Loading  -----------------------------"""


embeddings = load_embeddings()

dataset = load_data(100)
df = dataset['data']


"""---------------------  Data Preparation Functions  ----------------------"""


def sentence_vector(tokens):
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    return np.mean(vectors,axis = 0) if vectors else np.zeros(300)


def multi_sentence_vector(sentences):
    n = len(sentences)
    X = []
    for i in range(n):
        X.append(sentence_vector(sentences[i]))
    return np.array(X)


def FFNNprepData(df):
    length = df.shape[0]
    
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:,0]]
    
    
    y = df.iloc[:,2].values
    X = multi_sentence_vector([tokenized_sentences[i] for i in range(length)])
    
    return[X,y]


"""---------------------------  FFNN Evaluation ----------------------------"""


Z = FFNNprepData(df)
X = Z[0]
y = Z[1]    

train_scores = []
scores = []
f1_scores = []

seeds = [354,67,42,6,93]

   
    
for seed in seeds:
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    for train_idx, test_idx in cv.split(X, y):
        
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        
        # ---- Model Goes Here ----
    
        NNmodel = Sequential([
            Dense(85, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(85, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(85, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(85, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(85, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(3, activation='softmax')                 
        ])
        
        # ------------------------
        
        
        NNmodel.compile(optimizer = 'adam',
                        loss = 'sparse_categorical_crossentropy',
                        metrics = ['accuracy'])
          
        NNmodel.fit(X_train, y_train, 
                    epochs=50, batch_size=16, 
                    validation_split = 0.1, 
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5, 
                                                                  restore_best_weights = True)],
                    verbose=0)
        train_acc = NNmodel.evaluate(X_train, y_train)[1]
        loss, acc = NNmodel.evaluate(X_test, y_test, verbose=0)
        
        y_pred = np.argmax(NNmodel.predict(X_test), axis=1)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
        
        scores.append(acc)
        train_scores.append(train_acc)

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores) 
mean_train = np.mean(train_scores)
std_train = np.std(train_scores) 

print([mean_acc, std_acc])
print([round(mean_acc*100,2), round(std_acc*100,2)])
print([round(mean_f1*100,2), round(std_f1*100,2)]) 
print([round(mean_train*100,2), round(std_train*100,2)]) 




"""--------------------------------------------------------------------"""




