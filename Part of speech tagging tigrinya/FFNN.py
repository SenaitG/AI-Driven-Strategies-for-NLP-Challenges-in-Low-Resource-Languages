#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nltk
import random
import io, re
from nltk.tokenize import sent_tokenize, word_tokenize
import graphviz


# In[2]:


import pickle
import numpy as np 
import os
import io
import re
 
files = os.listdir('Corpra/')
n_sample_files = 1

print('Total No. of files', len(files), '\n')
print('Running on', n_sample_files, 'FILES\n')

raw_corpus = ''
#f = io.open(raw_corpus, mode ="r", encoding ="utf-8")
for file in files[0:n_sample_files]:
    with open('Corpra/'+ file) as f:
        raw_corpus = raw_corpus + '\n' + f.read()
corpus = raw_corpus.split('\n')

print('CORPUS SIZE', len (corpus), '\n')
#print(corpus)

qalat =[]
pastag =[]
words = []
tags = []
sentence = []
all_sentence= []
for line in corpus:
    arr = line.split()
    all_sentence.append(arr)
for el in all_sentence:
        mels=[tuple(i.split("/")) for i in el]
        sentence.append(mels)
#print(sentence)
for lin in sentence:
    for e in lin:
        words.append(e[0])
        tags.append(e[1])
        
#word = "aImIroawi"
for word in words:
    indexs = words.index(word)

#print(indexs)
#print(words)
#print(tags)
#print(len(words))
#print(len(tags))
#print(len(sentence))

    


# In[3]:


train_test_cutoff = int(.80 * len(sentence)) 
training_sentences = sentence[:train_test_cutoff]
testing_sentences = sentence[train_test_cutoff:]
train_val_cutoff = int(.25 * len(training_sentences))
validation_sentences = training_sentences[:train_val_cutoff]
training_sentences = training_sentences[train_val_cutoff:]
#print(validation_sentences)
print("======")
#print(testing_sentences)
print("=====")
#print(training_sentences)


# In[4]:


def add_basic_features(words, indexs):
    term = words[indexs]
    return{
        'nb_terms': len(words),
        'term': term,
        'is_first': indexs == 0,
        'is_last': indexs == len(words) - 1,
        'prev_word': '' if indexs == 0 else tags[indexs - 1],
        'minus2': '' if indexs == 0 else tags[indexs -2],
        'next_word': '' if indexs == len(tags) - 1 else tags[indexs + 1],
        'plus2': '' if indexs == len(tags) -1 else tags[indexs +2]
        
    }

term =words[indexs]


# In[5]:


def untag(sentences):
    return (words)


# In[6]:


def transform_to_dataset(sentence):
     X,y = [], []
     for words in sentence:
        for indexs, (term, class_) in enumerate(words):
            X.append(add_basic_features(untag(tags), indexs))
            y.append(class_)
     return X,y


# In[7]:


print(term)


# In[8]:


X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(testing_sentences)
X_val, y_val = transform_to_dataset(validation_sentences)


# In[ ]:





# In[9]:


from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing
max_features = 10000
maxlen = 20
(xx_train,yy_train), (xx_test, yy_test) = imdb.load_data(num_words = max_features)

xx_train = preprocessing.sequence.pad_sequences(xx_train, maxlen =maxlen)
yy_train = preprocessing.sequence.pad_sequences(xx_test, maxlen = maxlen)
embedding_layer = Embedding(1000,64)


# In[10]:


print(len(X_train))


# In[11]:


from sklearn.feature_extraction import DictVectorizer
dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train + X_test + X_val)

X_train = dict_vectorizer.transform(X_train)
X_test = dict_vectorizer.transform(X_test)
X_val = dict_vectorizer.transform(X_val)


# In[12]:


print(X_train), print(X_train[1])


# In[ ]:





# In[13]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_val)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)


# In[14]:


from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)


# In[15]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

def build_model(input_dim, hidden_neurons, output_dim):
    model= Sequential([
        Dense(hidden_neurons, input_dim = input_dim),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(hidden_neurons, input_dim = input_dim),
        Activation('sigmoid'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
        
    ])
    model.compile(loss = 'categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])
    return model


# In[16]:


from keras.wrappers.scikit_learn import KerasClassifier

model_params = {
    'build_fn': build_model,
    'input_dim': 142,
    'hidden_neurons': 256,
    'output_dim':20,
    'epochs': 100,
    'batch_size':256,
    'verbose': 1,
    'validation_data': (X_val, y_val),
    'shuffle': True
}

clf = KerasClassifier(**model_params)


# In[17]:


X_train.shape[1]


# In[18]:


print(y_train.shape)


# In[19]:


print(X_train.shape)


# In[20]:


print(X_val.shape)


# In[21]:


print(y_val.shape)


# In[22]:


print(X_test.shape)


# In[23]:


print(y_test.shape)


# In[24]:


hist = clf.fit(X_train, y_train)


# In[25]:


import matplotlib.pyplot as plt
def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    """ Plot model loss and accuracy through epochs. """
    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'
    # plot model loss
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='training')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='validation')
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('loss')
    ax1.tick_params('y')
    ax1.legend(loc='upper right', shadow=False)
    ax1.set_title('Model loss through #epochs', color=orange, fontweight='bold')
    # plot model accuracy
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=10, label='training')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=10, label='validation')
    ax2.set_xlabel('# epoch')
    ax2.set_ylabel('accuracy')
    ax2.tick_params('y')
    ax2.legend(loc='lower right', shadow=False)
    ax2.set_title('Model accuracy through #epochs', color=orange, fontweight='bold')


# In[26]:


plot_model_performance(
    train_loss=hist.history.get('loss', []),
    train_acc=hist.history.get('acc', []),
    train_val_loss=hist.history.get('val_loss', []),
    train_val_acc=hist.history.get('val_acc', [])
)


# In[27]:


import graphviz
from keras.utils import plot_model 
plot_model(clf.model, to_file ='model.png', show_shapes = True)


# In[30]:


score = clf.score(X_test, y_test)
print(score)

loss = clf.loss(X_test, y_test)
print(loss)


# In[33]:


test_results = clf.score( X_test, y_test)
print('TEST LOSS %F \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))


# In[2]:


from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils


# In[36]:


from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from hyperas.distributions import uniform

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np 
import pickle, sys, os
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100
TEST_SPLIT = 0.2
VALIDATION_SPLIT =0.2
BATCH_SIZE = 32
ref = "recentdata.txt"
data1 = open('recentdata.txt').read()
words , tags = [] , []
for i, line in enumerate(data1.split("\n")):
    content = line.split()
    for el in content:
        part= el.split(",")
        w = part[0]
        t = part[1]
    words.append(w)
    tags.append(t)

with open('data.pkl', 'rb') as f:
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
print(X_train)
train_generator = generator(all_X = X_train, all_y = y_train, n_classes = n_tags + 1)
validation_generator = generator(all_X = X_val, all_y=y_val, n_classes = n_tags + 1)


# In[37]:


with open("myfile.pkl",'rb') as f:
        embeddings_index = pickle.load(f)


# convert text to sequence of tokens and pad them to ensure equal length vectors 
#train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
#valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

embedding_matrix = numpy.random.random((len(word2int) +1 , 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
         embedding_matrix[i] = embedding_vector
    #return train_x, valid_x, train_y,valid_y, word_index, embedding_matrix, embedding_vector,train_seq_x, valid_seq_x
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    train_generator = generator(all_X = X_train, all_y = y_train, n_classes = n_tags + 1)
    validation_generator = generator(all_X = X_val, all_y=y_val, n_classes = n_tags + 1)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, validation_generator)


# In[38]:


from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

def create_rnn_lstm():
     # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word2int) + 1, 100, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(64, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="relu")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.rmsprop(), loss='binary_crossentropy')
    
    return model

classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-LSTM, Word Embeddings",  accuracy)


# In[35]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import io,re
import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pickle
ref = "recentdata.txt"
data = open('recentdata.txt').read()
words , tags = [] , []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    for el in content:
        part= el.split(",")
        w = part[0]
        t = part[1]
    words.append(w)
    tags.append(t)

    
trainDF = pandas.DataFrame()
trainDF['word'] = words
trainDF['tag'] = tags

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['word'],trainDF['tag'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y= encoder.fit_transform(valid_y)

with open("myfile.pkl",'rb') as f:
    embeddings_index = pickle.load(f)
token = text.Tokenizer()
token.fit_on_texts(trainDF['word'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

embedding_matrix = numpy.random.random((len(word_index) +1 , 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)


# In[37]:


def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((70, ))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

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
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings",  accuracy)


# In[ ]:




