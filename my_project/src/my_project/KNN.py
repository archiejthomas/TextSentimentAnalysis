#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

This file provides evaluation of the K-Nearest Neighbours classification method
on the Financial Phrasebank Dataset. 

It involves:
 - Loading of the embeddings and the specific degree of agreement dataset
 - Data preparation functions
 - 4 different blocks in which the classifier is run and evaluated
 - The 4 different approaches are:
     - Full w/ PCA
     - Full w/ TSNE
     - Averaged w/ PCA
     - Averaged w/ TSNE
 - Evaluation includes:
     - Average and standard deviation of testing accuracy 
     - Average and standard deviation of F1 Score
     - Aggregated confusion matrix
     
"""


import numpy as np
import random
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE


from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings


from nltk.tokenize import word_tokenize  
import matplotlib.pyplot as plt




"""----------------------------  Data Loading  -----------------------------"""


embeddings = load_embeddings()

dataset = load_data(100)
df = dataset['data']


"""---------------------  Data Preparation Functions  ----------------------"""


def sentence_vector_long(tokens, max_words=81):
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    vectors = vectors[:max_words]
    while len(vectors) < max_words:
        vectors.append(np.zeros(300))
    return np.concatenate(vectors, axis=0)


def multi_sentence_vector_long(sentences):
    n = len(sentences)
    X = []
    for i in range(n):
        X.append(sentence_vector_long(sentences[i]))
    return np.array(X)


def KNNprepDataLong(df):
    length = df.shape[0]
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:, 0]]
    y = df.iloc[:, 2].values
    X = multi_sentence_vector_long([tokenized_sentences[i] for i in range(length)])
    return [X, y]




def sentence_vector(tokens):
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)


def multi_sentence_vector(sentences):
    n = len(sentences)
    X = []
    for i in range(n):
        X.append(sentence_vector(sentences[i]))
    return np.array(X)


def KNNprepData(df):
    length = df.shape[0]
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:, 0]]
    y = df.iloc[:, 2].values
    X = multi_sentence_vector([tokenized_sentences[i] for i in range(length)])
    return [X, y]




"""----------------------  Full KNN Evaluation w/PCA -----------------------"""

Z = KNNprepDataLong(df)
X = Z[0]
y = Z[1]    


train_scores = []
scores = []
f1_scores =[]
all_preds = []
all_true = []
all_histories = []


seeds = [354, 67, 42, 6, 93]
j= 0 
for seed in seeds:
    
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for train_idx, test_idx in outer_cv.split(X, y):
        j+=1
        print(j)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.9)),
            ('knn', KNeighborsClassifier())
        ])


        param_grid = {
            'knn__n_neighbors': [3, 7, 11, 15, 20],
            'knn__weights': ['distance']
        }

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed
            
        )

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        acc = best_model.score(X_test, y_test)
        scores.append(acc)
        
        y_pred = best_model.predict(X_test)
        all_preds.extend(y_pred)   
        all_true.extend(y_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)


mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
#ax.set_title('Aggregated Confusion Matrix — CNN 3 Kernel (100% Agreement)')
plt.tight_layout()
plt.savefig('confusion_matrix_knn8.png', dpi=300, bbox_inches='tight')
plt.show()






"""------------------  Full KNN Evaluation w/TSNE  ----------------------"""


Z = KNNprepDataLong(df)
X_raw = Z[0]  
y = Z[1]

# --- TSNE fit on full dataset before any splitting ---
# Note: this introduces mild data leakage 

tsne = TSNE(
    n_components=3,       
    perplexity=30,        
    max_iter=1000,
    random_state=42
)
X = tsne.fit_transform(X_raw)  
# ----------------------------------------------------

train_scores = []
scores = []
f1_scores = []
all_preds = []
all_true = []
all_histories = []
seeds = [354, 67, 42, 6, 93]

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'knn__weights': ['distance']
        }

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed
        )

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
        scores.append(acc)

        y_pred = best_model.predict(X_test)
        all_preds.extend(y_pred)
        all_true.extend(y_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("Test Accuracy: ", [round(mean_acc * 100, 2), round(std_acc * 100, 2)])
print("F1 Score: ", [round(mean_f1 * 100, 2), round(std_f1 * 100, 2)])

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_knn_tsne1.png', dpi=300, bbox_inches='tight')
plt.show()





"""----------------------  Average KNN Evaluation w/PCA --------------------"""


Z = KNNprepData(df)
X = Z[0]
y = Z[1]    


train_scores = []
scores = []
f1_scores =[]
all_preds = []
all_true = []
all_histories = []


seeds = [354, 67, 42, 6, 93]

for seed in seeds:

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for train_idx, test_idx in outer_cv.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('knn', KNeighborsClassifier(metric='cosine'))
        ])


        param_grid = {
            'pca__n_components': [3],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'knn__weights': ['distance']
        }

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed
        )

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        acc = best_model.score(X_test, y_test)
        scores.append(acc)
        
        y_pred = best_model.predict(X_test)
        all_preds.extend(y_pred)   
        all_true.extend(y_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)


mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
#ax.set_title('Aggregated Confusion Matrix — CNN 3 Kernel (100% Agreement)')
plt.tight_layout()
plt.savefig('confusion_matrix_knncosine.png', dpi=300, bbox_inches='tight')
plt.show()





"""------------------  Average KNN Evaluation w/TSNE  ----------------------"""


Z = KNNprepData(df)
X_raw = Z[0]  
y = Z[1]

# --- TSNE fit on full dataset before any splitting ---
# Note: this introduces mild data leakage 

tsne = TSNE(
    n_components=3,       
    perplexity=30,        
    max_iter=1000,
    random_state=42
)
X = tsne.fit_transform(X_raw)  
# ----------------------------------------------------

train_scores = []
scores = []
f1_scores = []
all_preds = []
all_true = []
all_histories = []
seeds = [354, 67, 42, 6, 93]

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier())
        ])

        param_grid = {
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'knn__weights': ['distance']
        }

        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=seed
        )

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
        scores.append(acc)

        y_pred = best_model.predict(X_test)
        all_preds.extend(y_pred)
        all_true.extend(y_test)

        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("Test Accuracy: ", [round(mean_acc * 100, 2), round(std_acc * 100, 2)])
print("F1 Score: ", [round(mean_f1 * 100, 2), round(std_f1 * 100, 2)])

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_knn_tsne.png', dpi=300, bbox_inches='tight')
plt.show()



"""------------------------------------------------------------------"""


