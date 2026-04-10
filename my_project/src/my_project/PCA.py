#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 12:21:04 2026

@author: archiethomas

This file implements PCA and TSNE to generate some of the graphs and images 
seen in my report

"""

from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings


import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.sequence import pad_sequences


"""----------------------------  Data Loading  -----------------------------"""

embeddings = load_embeddings()

dataset = load_data(100)
df = dataset['data']

"""-------------------------------------------------------------------------"""



def plot_decision_boundary(X, y, model):
    h = 0.02  # Step size for mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict on mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X_train, y_train, svm_classifier)
    






def pairedPCA(list1, list2, title = "PCA of paired word embeddings"):
    data = list(list1) + list(list2)
    scaler = StandardScaler()
    dataEmbeddings = np.array([embeddings[_] for _ in data])
    dataScaled = scaler.fit_transform(dataEmbeddings)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(dataScaled)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    list1_idx = [data.index(c) for c in list1]
    list2_idx = [data.index(c) for c in list2]
    
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=principal_df, x="PC1", y="PC2", ax=ax, size = 500)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(title)
    
    offsets = np.random.uniform(-0.05,0.05, size = (len(data),2))
    for label, x, y, (dx,dy) in zip(data, principal_df['PC1'], principal_df['PC2'], offsets):
        ax.annotate(label, (x, y), textcoords="offset points", 
                                 xytext=(0,10), ha='center', size = 15)
        

    for i, j in zip(list1_idx, list2_idx):
        x1, y1 = principal_df.iloc[i]
        x2, y2 = principal_df.iloc[j]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, 
                                  color='red', alpha=0.3))
    
    return {"Scaler": scaler, "Fit": pca, "Ax": ax, "Plot": fig}



countries = [
    'Austria', 'Belgium', 'Denmark', 'Finland', 'France', 
    'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
    'Norway',  
    'Sweden'
]

capitals = [
    'Vienna', 'Brussels', 'Copenhagen', 'Helsinki', 'Paris',
    'Berlin', 'Athens', 'Budapest', 'Dublin', 'Rome',
    'Oslo',  
    'Stockholm'
]


present = [
    'run', 'jump', 'eat', 'sleep', 'walk',
    'read', 'write', 'sing', 'dance', 'play'
]

past = [
    'ran', 'jumped', 'ate', 'slept', 'walked',
    'read', 'wrote', 'sang', 'danced', 'played'
]


pairedPCA(countries,capitals, "PCA: Countries vs Capitals")
pairedPCA(present, past, "PCA: Present vs Past")

"""------------------------------------------------------------------------"""


def expand_sentences(split, data):
    length = data.shape[0]
    n = int(np.floor(length*split))
    
    train_index = np.random.choice(length, size=n, replace=False)
    test_index = np.setdiff1d(np.arange(length), train_index)
    
    train_sents = data.iloc[train_index, 0].str.lower().tolist()
    test_sents = data.iloc[test_index, 0].str.lower().tolist()
    
    train_tokens = [word_tokenize(sent) for sent in train_sents]
    X_train = [[embeddings[w] if w in embeddings else np.zeros(300) 
                for w in tokens] for tokens in train_tokens]
   
    
    test_tokens = [word_tokenize(sent) for sent in test_sents]
    X_test = [[embeddings[w] if w in embeddings else np.zeros(300) 
                for w in tokens] for tokens in test_tokens]
    
    y = data.iloc[train_index,1].values
    ytest = data.iloc[test_index,1].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  
    ytest_encoded = le.transform(ytest)

    
    return X_train, X_test, y_encoded, ytest_encoded


X_train, X_test, y_train, y_test = expand_sentences(0.6,dataset)
X_train = pad_sequences(X_train, maxlen = 30, dtype = 'float32')
X_test= pad_sequences(X_test, maxlen = 30, dtype = 'float32')
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)



print(X_test)



X = np.array([embeddings[_] for _ in present])

pca = PCA().fit(X_train_flat)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of  Principle Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()



pca = PCA(n_components=300)
#scaler = StandardScaler()
#X_train_flat = scaler.fit_transform(X_train_flat)
#X_test_flat = scaler.transform(X_test_flat)
principal_components = pca.fit_transform(X_train_flat)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

cmap = ['red','blue','green']

plt.scatter(principal_df['PC1'], principal_df['PC2'],s=2, c=y_train, cmap=matplotlib.colors.ListedColormap(cmap))



kpca = KernelPCA(
    n_components=2,        # Target dims (or None for max possible)
    kernel='linear',           # 'rbf', 'poly', 'sigmoid', 'cosine', 'linear'           # Tune for RBF: 1/n_features or 'scale'/'auto'
    alpha=1e-5             # Ridge regularization (helps stability
)

principal_components = kpca.fit_transform(X_train_flat)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

cmap = ['red','blue','green']

plt.scatter(principal_df['PC1'], principal_df['PC2'],s=2, c=y_train, cmap=matplotlib.colors.ListedColormap(cmap))








def dataPCA(data, n):
    X_train = data[0]
    X_test = data[3]
    scaler = StandardScaler()
    X_trainScaled = scaler.fit_transform(X_train)
    X_testScaled = scaler.fit_transform(X_test)
    
    pca = PCA(n_components=n)
    principal_components_train = pca.fit_transform(X_trainScaled)
    principal_components_test = pca.transform(X_testScaled)
    #principal_df_train = pd.DataFrame(data=principal_components_train, columns=['PC1', 'PC2'])
    #principal_df_test = pd.DataFrame(data=principal_components_test, columns=['PC1', 'PC2'])
    
    #cmap = ['red','blue','green']
    #plt.scatter(principal_df_train['PC1'], principal_df_train['PC2'],s=2, c=data[2], cmap=matplotlib.colors.ListedColormap(cmap))
    
    return{
        "X_train": principal_components_train,
        "X_test": principal_components_test
        }


"""-------------------------------------------------------------------------"""


"""---------------------  Data Preparation Functions  ----------------------"""




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






"""-------------  2D Visualisation of averaged embeddings  ----------------"""


Z = KNNprepData(df)
X = Z[0]
y = Z[1]    


tsne = TSNE(
    n_components=2,      
    perplexity=30,       
    max_iter=1000,         
    random_state=42      
)

X_tsne = tsne.fit_transform(X)


colours = ['#E69F00', '#56B4E9', '#009E73']
labels  = ['Negative', 'Neutral', 'Positive']

fig, ax = plt.subplots(figsize=(8, 6))

for i, (colour, label) in enumerate(zip(colours, labels)):
    mask = y == i
    ax.scatter(
        X_tsne[mask, 0],
        X_tsne[mask, 1],
        c=colour,
        label=label,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.4,
        s=60
    )

ax.set_xlabel('TSNE dimension 1')
ax.set_ylabel('TSNE dimension 2')

ax.legend(title='Sentiment', framealpha=0.9)

plt.tight_layout()
plt.savefig('tsne_visualisation.png', dpi=300, bbox_inches='tight')
plt.show()





from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)


colours = ['#E69F00', '#56B4E9', '#009E73']
labels  = ['Negative', 'Neutral', 'Positive']

fig, ax = plt.subplots(figsize=(8, 6))

for i, (colour, label) in enumerate(zip(colours, labels)):
    mask = y == i
    ax.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        c=colour,
        label=label,
        alpha=0.7,
        edgecolors='white',
        linewidths=0.4,
        s=60
    )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance explained)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance explained)')

ax.legend(title='Sentiment', framealpha=0.9)

plt.tight_layout()
plt.savefig('pca_visualisation.png', dpi=300, bbox_inches='tight')
plt.show()


"""--------------------------------------------------------------------------------------"""






