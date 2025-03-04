from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
import pickle, sys, os

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from numpy import array

import keras

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
VALIDATION_SPLIT =0.2
BATCH_SIZE = 32

with open('PickledData/data.pkl', 'rb') as f:
    X,y, word2int, int2word, tag2int, int2tag = pickle.load(f)
    
    
def generator(all_X, all_y, n_classes, batch_size = BATCH_SIZE):
    num_samples = len(all_X)
    
    
    while True:
        
        for offset in range(0, num_samples, batch_size):
            
            X = all_X[offset:offset+batch_size]
            y = all_y[offset:offset+batch_size]
            
            y = to_categorical(y, num_classes=n_classes)

            
            yield shuffle(X,y)
            
            
n_tags = len(tag2int)

X = pad_sequences(X, maxlen = MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen= MAX_SEQUENCE_LENGTH)

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))

X,y = shuffle(X,y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SPLIT, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VALIDATION_SPLIT, random_state = 1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' %n_test_samples)


def data():
    with open('PickledData/data.pkl', 'rb') as f:
        X,y, word2int, int2word, tag2int, int2tag = pickle.load(f)
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100
    TEST_SPLIT = 0.2
    VALIDATION_SPLIT =0.2
    BATCH_SIZE = 32
    n_tags = len(tag2int)

    X = pad_sequences(X, maxlen = MAX_SEQUENCE_LENGTH)
    y = pad_sequences(y, maxlen= MAX_SEQUENCE_LENGTH)

    print('TOTAL TAGS', len(tag2int))
    print('TOTAL WORDS', len(word2int))

    X,y = shuffle(X,y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SPLIT, random_state = 42)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = VALIDATION_SPLIT, random_state = 1)

    n_train_samples = X_train.shape[0]
    n_val_samples = X_val.shape[0]
    n_test_samples = X_test.shape[0]

    print(X_train.shape)
    print(y_train.shape)
   

    

    with open("filey.pkl",'rb' ) as f:
        embeddings_index = pickle.load(f)
    
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random ((len(word2int) + 1, EMBEDDING_DIM))

    for word, i in word2int.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        
    print('Embedding matrix shape', embedding_matrix.shape)
    print('X_train shape', X_train.shape)
    return X_train, X_test,X_val, y_train,y_test, y_val, embedding_vector, embedding_matrix, embedding_index



def model(X_train, X_test, y_train, y_test, embedding_matrix, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(len(word2int) +1 , EMBEDDING_DIM , weights = [embedding_matrix], trainable = False))
    model.add(LSTM({{choice([16,32,64,128])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(100))
    model.add(Activation({{choice(['sigmoid','softmax','tanh','relu','swish','selu'])}}))

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['adam', 'sgd','rmsprop'])}},
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=4)
    checkpointer = ModelCheckpoint(filepath='keras_weights.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              batch_size={{choice([32, 64, 128])}},
              epochs=50,
              validation_split=0.2,
              callbacks=[early_stopping, checkpointer])

    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print(best_run)