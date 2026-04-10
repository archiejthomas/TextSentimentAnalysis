#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 19:29:25 2026

@author: archiethomas
"""



import numpy as np
import random
import tensorflow as tf

from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
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



"""----------------------  Average SVM Evaluation w/PCA --------------------"""

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

j = 0 

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
        
        j += 1
        print(j, " out of 25 fits")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),          
            ('svm', LinearSVC(max_iter=5000))  
        ])


        param_grid = {
            'pca__n_components': [3],  # you can adjust
            'svm__C': [0.1, 1, 10, 100]
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


# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


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
plt.savefig('confusion_matrix_SVM4.png', dpi=300, bbox_inches='tight')
plt.show()




"""------------------  Average SVM Evaluation w/TSNE  ----------------------"""


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

j = 0

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

        
        j += 1
        print(j, " out of 25 fits")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),       
            ('svm', LinearSVC(max_iter=5000))  
        ])


        param_grid = {
            'svm__C': [0.1, 1, 10, 100]
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


# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("Test Accuracy: ", [round(mean_acc * 100, 2), round(std_acc * 100, 2)])
print("F1 Score: ", [round(mean_f1 * 100, 2), round(std_f1 * 100, 2)])


# ------ Aggregated Confusion Matrix ------------------------------------------

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_svm_tsne.png', dpi=300, bbox_inches='tight')
plt.show()



"""----------------------  Full SVM Evaluation w/PCA -----------------------"""

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
            ('pca', PCA()),          
            ('svm', LinearSVC(max_iter=5000))  
        ])


        param_grid = {
            'pca__n_components': [300],  # you can adjust
            'svm__C': [0.1, 1, 10, 100]
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

# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


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
plt.savefig('confusion_matrix_svm5.png', dpi=300, bbox_inches='tight')
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
            ('svm', LinearSVC(max_iter=5000))  
        ])


        param_grid = {
 
            'svm__C': [0.1, 1, 10, 100]
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
        
# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("Test Accuracy: ", [round(mean_acc * 100, 2), round(std_acc * 100, 2)])
print("F1 Score: ", [round(mean_f1 * 100, 2), round(std_f1 * 100, 2)])


# ------ Aggregated Confusion Matrix ------------------------------------------

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_svm_tsne1.png', dpi=300, bbox_inches='tight')
plt.show()








"""----------------------  Average SVM RBF Evaluation w/PCA --------------------"""

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

j = 0 

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
        
        j += 1
        print(j, " out of 25 fits")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]


        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('svm', SVC(kernel='rbf'))
        ])
        
        param_grid = {
            'pca__n_components': [300],
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto']  # key parameter for RBF kernel
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


# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


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
plt.savefig('confusion_matrix_SVMRBF2.png', dpi=300, bbox_inches='tight')
plt.show()




"""------------------  Average SVM RBF Evaluation w/TSNE  ----------------------"""


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

j = 0

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

        
        j += 1
        print(j, " out of 25 fits")

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf'))
        ])
        
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto']  # key parameter for RBF kernel
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


# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print("Test Accuracy: ", [round(mean_acc * 100, 2), round(std_acc * 100, 2)])
print("F1 Score: ", [round(mean_f1 * 100, 2), round(std_f1 * 100, 2)])


# ------ Aggregated Confusion Matrix ------------------------------------------

cm = confusion_matrix(all_true, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Negative', 'Neutral', 'Positive'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_fontsize(20)
plt.tight_layout()
plt.savefig('confusion_matrix_svm_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

"""----------------------  Full SVM RBF Evaluation w/PCA -----------------------"""

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
            ('pca', PCA()),
            ('svm', SVC(kernel='rbf'))
        ])
        
        param_grid = {
            'pca__n_components': [300],
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto']  
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

# ------ Summary Metrics -----------------------------------------------------

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)


print("Test Accuracy: ",[round(mean_acc*100,2), round(std_acc*100,2)])
print("F1 Score: ",[round(mean_f1*100,2), round(std_f1*100,2)]) 


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
plt.savefig('confusion_matrix_svm10.png', dpi=300, bbox_inches='tight')
plt.show()

