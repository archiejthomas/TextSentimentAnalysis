#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:27:30 2026

@author: archiethomas

In this file we have all the models that are quoted in the report - Appear in 
the order in which they appear in the report
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Input, Concatenate, Dense, GlobalMaxPooling1D, Dropout, LSTM, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.regularizers import l2

"""-- FFNN Dense --"""

FFNNDense = Sequential([
            Dense(128, activation='relu', input_shape=(300,)),
            Dropout(0.2),
            Dense(128,activation='relu'),
            Dropout(0.2),
            Dense(3, activation='softmax')                 
        ])

"""-- FFNN Deep --"""

FFNNWide = Sequential([
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


"""-- CNN n Kernels --"""

input_layer = Input(shape=(81, 300))

conv1 = Conv1D(filters = 128, kernel_size = 1, activation = 'relu', 
               padding = 'valid', kernel_regularizer = l2(0.01))(input_layer)

conv2 = Conv1D(filters = 128, kernel_size = 2, activation = 'relu', 
               padding = 'valid', kernel_regularizer = l2(0.01))(input_layer)

conv3 = Conv1D(filters = 128, kernel_size = 3, activation = 'relu', 
               padding = 'valid', kernel_regularizer = l2(0.01))(input_layer)

conv4 = Conv1D(filters = 128, kernel_size = 4, activation = 'relu', 
               padding = 'valid', kernel_regularizer = l2(0.01))(input_layer)


pool1 = GlobalMaxPooling1D()(conv1)
pool2 = GlobalMaxPooling1D()(conv2)
pool3 = GlobalMaxPooling1D()(conv3)
pool4 = GlobalMaxPooling1D()(conv4)

concat = Concatenate()([pool1, pool2, pool3, pool4])

drop = Dropout(0.5)(concat)

dense = Dense(256, activation='relu')(drop)

drop1 = Dropout(0.5)(dense)

output = Dense(3, activation='softmax')(drop1)

CNNmodel = Model(inputs=input_layer, outputs=output)


"""-- Vanilla RNN --"""

VanillaRNN = Sequential([
    

    SimpleRNN(64, dropout = 0.1, input_shape = ( 43,300)),
    

    Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
    

    Dense(3, activation = 'softmax')
])


"""-- LSTM --"""

LSTMModel = Sequential([
    

    LSTM(64, dropout = 0.1, input_shape = ( 43,300)),
    

    Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
    

    Dense(3, activation = 'softmax')
])




"""-- Bidirectional LSTM --"""

BiLSTM = Sequential([
    

    Bidirectional(LSTM(32,dropout = 0.3 ), input_shape = ( 43,300)),
    
    Dropout(0.4),
    

    Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
    

    Dense(3, activation = 'softmax')
])


"""-- Bidirectional GRU --"""

BiGRU = Sequential([
    

    Bidirectional(GRU(32,dropout = 0.3 ), input_shape = ( 43,300)),
    
    Dropout(0.4),
    

    Dense(64, activation = 'relu', kernel_regularizer = l2(0.01)),
    

    Dense(3, activation = 'softmax')
])


"""-- Hybrid 1 --"""


input_layer = Input(shape=(81, 300))

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

Hybrid1 = Model(inputs=input_layer, outputs=output)


"""-- Hybrid 2 --"""


input_layer = Input(shape=(81, 300))
 

conv1 = Conv1D(filters=128, kernel_size=1,
               activation='relu', padding='valid',
               kernel_regularizer=l2(0.01))(input_layer)
conv2 = Conv1D(filters=128, kernel_size=2,
               activation='relu', padding='valid',
               kernel_regularizer=l2(0.01))(input_layer)

drop_conv1 = Dropout(0.2)(conv1)
drop_conv2 = Dropout(0.2)(conv2)

lstm1 = Bidirectional(LSTM(32, return_sequences = True))(drop_conv1)
pool1 = GlobalMaxPooling1D()(lstm1)
drop_lstm1 = Dropout(0.2)(pool1)

lstm2 = Bidirectional(LSTM(32, return_sequences = True))(drop_conv2)
pool2 = GlobalMaxPooling1D()(lstm2)
drop_lstm2 = Dropout(0.2)(pool2)

concat_cnn = Concatenate()([drop_lstm1, drop_lstm2])
   
dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(concat_cnn)
drop1 = Dropout(0.3)(dense)
output = Dense(3, activation='softmax')(drop1)

Hybrid2 = Model(inputs=input_layer, outputs=output)
