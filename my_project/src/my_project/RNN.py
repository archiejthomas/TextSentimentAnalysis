# -*- coding: utf-8 -*-

from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings


from nltk.tokenize import word_tokenize
import numpy as np


import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Dense, GlobalMaxPooling1D, Dropout, LSTM, Embedding, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.regularizers import l2

from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import random



"""----------------------------  Data Loading  -----------------------------"""


embeddings = load_embeddings()

dataset = load_data(100)
df = dataset['data']


posneg = df[df['label'] != 'neutral']
posneg['y'] = posneg['y']//2
posneg

"""----------------------------  Data Preparation  -----------------------------"""

def RNNprepData(df):
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:,0]]
    
    vectorized = []
    for i in range(len(tokenized_sentences)):
        vectors = [embeddings[w] for w in tokenized_sentences[i] if w in embeddings]
        vectorized.append(vectors)
   
    y = df.iloc[:,2].values
    X = vectorized

    X = tf.keras.preprocessing.sequence.pad_sequences(
        X, 
        maxlen=43, 
        dtype='float32', 
        padding='post', 
        truncating='post', 
        value=np.zeros(300)
    )
    
    return[X,y]



"""----------------------------  Data Exploration -----------------------------"""


tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:,0]]
lengths = [len(tokens) for tokens in tokenized_sentences]
print("Mean:", np.mean(lengths))
print("Median:", np.median(lengths))
print("95th percentile:", np.percentile(lengths, 95))
print("Max:", np.max(lengths))



plt.figure(figsize=(8, 5))
plt.hist(
    lengths,
    bins=30,
    edgecolor='black',
    alpha=0.75
)

plt.xlabel("Sentence Length (tokens)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
#plt.title("Distribution of Sentence Lengths", fontsize=14, pad=12)

plt.grid(axis='y', linestyle='--', alpha=0.6)

mean_val = np.mean(lengths)
median_val = np.median(lengths)

plt.axvline(mean_val, linestyle='--', linewidth=1.5, label=f"Mean ({mean_val:.1f})")
plt.axvline(median_val, linestyle='-', linewidth=1.5, label=f"Median ({median_val:.1f})")


plt.legend(fontsize=16)
plt.tight_layout()
plt.show()

"""-----------------------  Neural Network Evaluation ----------------------"""

Z = RNNprepData(posneg)
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

        
for seed in seeds:

    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    for train_idx, test_idx in cv.split(X, y):
        
        random.seed(seed)
        tf.random.set_seed(seed)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        
        # ------  Model Here --------------------------------------
    
        RNNmodel = Sequential([
            

            SimpleRNN(64, dropout = 0.1, input_shape = ( 43,300)),
            
        
            Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
            
        
            Dense(2, activation = 'softmax')
        ])

        # ---------------------------------------------------------

        RNNmodel.compile(optimizer = 'adam',
                        loss = 'sparse_categorical_crossentropy',
                        metrics = ['accuracy'])
          
        
        
        history = RNNmodel.fit(X_train, y_train, 
                        epochs=50, batch_size=32, 
                        validation_split = 0.1, 
                        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5, 
                                                                      restore_best_weights = True)],
                        verbose=0)
        
        all_histories.append(history.history)
        
        
        train_acc = RNNmodel.evaluate(X_train, y_train)[1]
        loss, acc = RNNmodel.evaluate(X_test, y_test, verbose=1)
        scores.append(acc)
        train_scores.append(train_acc)
        
        y_pred = np.argmax(RNNmodel.predict(X_test), axis=1)
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

plt.style.use('default')

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
#ax.set_title('Aggregated Confusion Matrix — CNN 3 Kernel (100% Agreement)')
plt.tight_layout()
ax.grid(False)
plt.savefig('confusion_matrix_VanillaRNN2.png', dpi=300, bbox_inches='tight')
plt.show()






