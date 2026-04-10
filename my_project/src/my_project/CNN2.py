# -*- coding: utf-8 -*-
from src.my_project.data_loader import load_data
from src.my_project.data_loader import load_embeddings


from nltk.tokenize import word_tokenize

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Dense, GlobalMaxPooling1D, Dropout, Bidirectional, LSTM, GRU
from tensorflow.keras.regularizers import l2


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

le = LabelEncoder()


import random



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






dataset = load_data(100)
df = dataset["data"]


Z = CNNprepData(df)
X = Z[0]
y = Z[1]    

i =0
train_scores = []
scores = []
f1_scores = []
seeds = [354,67,42,6,93]
seedsmini = [354,67]

if len(scores) !=0 :
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")    
    
for seed in seeds:
    
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
    
    for train_idx, test_idx in cv.split(X, y):
        
        i+=1
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    
    
    
        input_layer = Input(shape=(maxlen, 300))

        conv1 = Conv1D(filters=128, kernel_size=1, 
                       activation='relu', padding='valid',
                       kernel_regularizer = l2(0.01))(input_layer)
        conv2 = Conv1D(filters=128, kernel_size=2, 
                       activation='relu', padding='valid', 
                       kernel_regularizer = l2(0.01))(input_layer)
        conv3 = Conv1D(filters=128, kernel_size=3, 
                       activation='relu', padding='valid',
                       kernel_regularizer = l2(0.01))(input_layer)
        
        pool1 = GlobalMaxPooling1D()(conv1)
        pool2 = GlobalMaxPooling1D()(conv2)
        pool3 = GlobalMaxPooling1D()(conv3)
        

        concat_cnn = Concatenate()([pool3,   pool2, pool1])
        drop_cnn = Dropout(0.4)(concat_cnn)
        
        lstm = Bidirectional(LSTM(64,dropout=0.3, 
                                  recurrent_dropout=0.3))(input_layer)
        drop_lstm = Dropout(0.4)(lstm)
        
        concat = Concatenate()([drop_cnn,drop_lstm])
        
        dense = Dense(64, activation='relu')(concat)
        drop1 = Dropout(0.3)(dense)
        output = Dense(3, activation='softmax')(drop1)

        CNNmodel = Model(inputs=input_layer, outputs=output)
        
        
        CNNmodel.compile(optimizer = 'adam',
                        loss = 'sparse_categorical_crossentropy',
                        metrics = ['accuracy'])
          
        print('COMPILED!')
        
        CNNmodel.fit(X_train, y_train, 
                    epochs=20, batch_size=32, 
                    validation_split = 0.1, 
                    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 5, 
                                                                  restore_best_weights = True)],
                    verbose=0)
        
        trainacc = CNNmodel.evaluate(X_train,y_train)
        loss, acc = CNNmodel.evaluate(X_test, y_test, verbose=1)
        scores.append(acc)
        train_scores.append(trainacc[1])
        
        y_pred = np.argmax(CNNmodel.predict(X_test), axis=1)
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
        
        print("Batch ", i, " average = ", np.mean(scores))

mean_acc = np.mean(scores)
std_acc = np.std(scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores) 
mean_train = np.mean(train_scores)
std_train = np.std(train_scores) 

print([round(mean_acc*100,2), round(std_acc*100,2)])
print([round(mean_f1*100,2), round(std_f1*100,2)]) 
print([round(mean_train*100,2), round(std_train*100,2)]) 





lst = []
for i in range(len(train_scores)):
    lst.append(train_scores[i][1])

"""----------------------------------------------------"""








def CNNprepData1(split, df):
    length = df.shape[0]
    n = int(np.floor(length*split))
    
    train_index = np.random.choice(length, size=n, replace=False)
    test_index = np.setdiff1d(np.arange(length), train_index)
    
    
    tokenized_sentences = [word_tokenize(sent.lower()) for sent in df.iloc[:,0]]
    
    vectorized = []
    
    for i in range(len(tokenized_sentences)):
        vectors = [embeddings[w] for w in tokenized_sentences[i] if w in embeddings]
        vectorized.append(vectors)
        
        
    y = df.iloc[train_index,1].values
    X = np.array(vectorized)[train_index]
    
    
    Xtest = np.array(vectorized)[test_index]
    ytest = df.iloc[test_index,1].values
    
    
    
    y_encoded = le.fit_transform(y)  
    ytest_encoded = le.transform(ytest)
    
    
    
    
    
    X = tf.keras.preprocessing.sequence.pad_sequences(
        X, 
        maxlen=81, 
        dtype='float32', 
        padding='post', 
        truncating='post', 
        value=np.zeros(300)
    )
    Xtest = tf.keras.preprocessing.sequence.pad_sequences(
        Xtest, 
        maxlen=81, 
        dtype='float32', 
        padding='post', 
        truncating='post', 
        value=np.zeros(300)
    )
        
    
    return [X,y,y_encoded,Xtest,ytest,ytest_encoded, le.classes_]


df = dataset["data"]

data = CNNprepData(0.1, df)
X_train = data[0]
Y_train = data[2]
X_test = data[3]
Y_test = data[5]


maxlen = 81
batch_size = 32
embedding_dims = 300 #Length of the token vectors
filters = 250 #number of filters in your Convnet
kernel_size = 3 # a window size of 3 tokens
hidden_dims = 250 #number of neurons at the normal feedforward NN
epochs = 5
padding = 'valid'



CNNmodel = Sequential([
    Conv1D(filters = 250, kernel_size = 3, 
                           padding = 'valid', activation = 'relu', 
                           strides = 1, input_shape = (81,300)),
    tf.keras.layers.GlobalMaxPooling1D(),
    Dense(hidden_dims),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Activation('relu'),
    Dense(3),
    tf.keras.layers.Activation('softmax')
])


CNNmodel = Sequential([
    Conv1D(filters = 250, kernel_size = 3, 
                           padding = 'valid', activation = 'relu', 
                           strides = 1, input_shape = (81,300)),
    GlobalMaxPooling1D(),
    Dense(units = 250, activation = 'relu'),
    Dropout(0.2),
    Dense(units = 3, activation = 'softmax')
])



def fitCNN(model):
    model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train,Y_train,batch_size = batch_size,epochs = epochs)
    return model


def evalCNN(model):
    CNNmodel.evaluate(X_test,Y_test)
    prediction = CNNmodel.predict(X_test)
    y_pred = tf.argmax(prediction, axis=1)
    y_true = tf.constant(Y_test)
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred)
    return conf_matrix


fitCNN(CNNmodel)
evalCNN(CNNmodel)


def singleTest(sent, model):

    tokens = word_tokenize(sent.lower())
    
    embedded = [embeddings[w] for w in tokens if w in embeddings]
    
    new = tf.keras.preprocessing.sequence.pad_sequences( [embedded], 
             maxlen=81, 
             dtype='float32', 
             padding='post', 
             truncating='post', 
             value=np.zeros(300)
     )
    
    print(model.predict(new))



trial = "For the last quarter of 2010 , Componenta 's net sales doubled to EUR131m from EUR76m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax loss of EUR7m"
trial1 = "For the last quarter of 2010 , Componenta 's net sales halved to EUR76m from EUR131m for the same period a year earlier , while it moved to a zero pre-tax profit from a pre-tax gain of EUR7m"

singleTest(trial,CNNmodel)
singleTest(trial1,CNNmodel)
