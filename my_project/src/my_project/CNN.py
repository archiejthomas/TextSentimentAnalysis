# -*- coding: utf-8 -*-
from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings

from nltk.tokenize import word_tokenize

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Dense, GlobalMaxPooling1D, Dropout, Bidirectional, LSTM
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

import random

le = LabelEncoder()


embeddings = load_embeddings()
maxlen = 81

def CNNprepData(df):
    
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:,0]]
    
    vectorized = []
    for i in range(len(tokenized_sentences)):
        vectors = [embeddings[w] for w in tokenized_sentences[i] if w in embeddings]
        vectorized.append(vectors)
   
    y = df.iloc[:,2].values
    X = vectorized

    X = tf.keras.preprocessing.sequence.pad_sequences(
        X, 
        maxlen=maxlen, 
        dtype='float32', 
        padding='post', 
        truncating='post', 
        value=np.zeros(300)
    )
    
    return[X,y]






dataset = load_data(50)
df = dataset["data"]


Z = CNNprepData(df)
X = Z[0]
y = Z[1]    

train_scores = []
scores = []
f1_scores =[]
all_preds = []
all_true = []
all_histories = []

seeds = [354,67,42,6,93]
seedsmini = [354,67]

# ------ Model Training and Testing ------------------------------------------
           
for seed in seeds:
    
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    for train_idx, test_idx in cv.split(X, y):
        
        
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    
        # ------ Sequential Model Here --------------------------------------
    
        input_layer = Input(shape=(maxlen, 300))
        conv1 = Conv1D(filters = 128, kernel_size = 1,
        activation = 'relu', padding = 'valid',
        kernel_regularizer = l2(0.01))(input_layer)
        conv2 = Conv1D(filters = 128, kernel_size = 2,
        activation = 'relu', padding = 'valid',
        kernel_regularizer = l2(0.01))(input_layer)
        conv3 = Conv1D(filters = 128, kernel_size = 3,
        activation = 'relu', padding = 'valid',
        kernel_regularizer = l2(0.01))(input_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        pool2 = GlobalMaxPooling1D()(conv2)
        pool3 = GlobalMaxPooling1D()(conv3)
        concat = Concatenate()([pool1, pool2, pool3])
        drop = Dropout(0.5)(concat)
        dense = Dense(256, activation='relu')(drop)
        drop1 = Dropout(0.5)(dense)
        output = Dense(3, activation='softmax')(drop1)
        CNNmodel = Model(inputs=input_layer, outputs=output)
        
        
        # -------------------------------------------------------------------
        
        
        CNNmodel.compile(optimizer = 'adam',
                        loss = 'sparse_categorical_crossentropy',
                        metrics = ['accuracy'])
          
        
        history = CNNmodel.fit(X_train, y_train, 
                        epochs=20, batch_size=16, 
                        validation_split = 0.1, 
                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5, 
                                                                      restore_best_weights = True)],
                        verbose=0)
        
        
        all_histories.append(history.history)
        
        trainacc = CNNmodel.evaluate(X_train,y_train)[1]
        loss, acc = CNNmodel.evaluate(X_test, y_test, verbose=1)
        scores.append(acc)
        train_scores.append(trainacc)
        
        y_pred = np.argmax(CNNmodel.predict(X_test), axis=1)
        all_preds.extend(y_pred)   
        all_true.extend(y_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores) 
mean_train = np.mean(train_scores)
std_train = np.std(train_scores) 

print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 
print("Train Accuracy: ",[round(mean_train*100,2), round(std_train*100,2)]) 


# ------ Aggregated Confusion Matrix ------------------------------------------


cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
#ax.set_title('Aggregated Confusion Matrix — CNN 3 Kernel (100% Agreement)')
plt.tight_layout()
plt.savefig('confusion_matrix_hybrid3 - 81.png', dpi=300, bbox_inches='tight')
plt.show()


