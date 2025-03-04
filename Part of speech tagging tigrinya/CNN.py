from sklearn import model_selection,preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import numpy as np 
import pickle, sys, os

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import re
import io 
raw_corpus = "NagaokaTigrinyaCorpus1_0_rom_T2.txt"
f = io.open(raw_corpus, mode= "r", encoding = "utf-8")
corpus = f.readlines()
X_train = []
Y_train = []
with_stalsh = False
n_omitted = 0 
words= []
tags = []

n_omitted = 0
for line in corpus:
    tempX = []
    tempY = []
    for word in line.split():
            w,tag = word.split('/')
            words.append(w)
            tags.append(tag)
            tempX.append(w)
            tempY.append(tag)
    X_train.append(tempX)
    Y_train.append(tempY)
BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
VALIDATION_SPLIT =0.2

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

train_generator = generator(all_X = X_train, all_y = y_train, n_classes = n_tags + 1)
validation_generator = generator(all_X = X_val, all_y=y_val, n_classes = n_tags + 1)
with open('filey.pkl','rb') as f:
    embeddings_index = pickle.load(f)
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
VALIDATION_SPLIT =0.2
BATCH_SIZE = 32
with open('data.pkl', 'rb') as f:
    X,y, word2int, int2word, tag2int, int2tag = pickle.load(f)
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net = False):
    classifier.fit(feature_vector_train,label)
    predictions = classifier.predict(feature_vector_valid)
    if is_neural_net:
        predictions = predictions.argmax(axis= -1)
    return metrics.accuracy_score(predictions, valid_y)
def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((100, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word2int) + 1, 100, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.6)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    
    return model

classifier = create_cnn()
y_test = to_categorical(y_test, num_classes=n_tags+1)
test_results = classifier.evaluate(X_test, y_test, verbose=0)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))


#accuracy = train_model(classifier, X_train, y_train, X_val, is_neural_net=True)
#print ("CNN, Word Embeddings",  accuracy)