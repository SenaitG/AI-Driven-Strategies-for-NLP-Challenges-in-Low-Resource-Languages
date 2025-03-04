import re, io
import _pickle as pickle
import numpy as np 
from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
corpus = "NagaokaTigrinyaCorpus1_0_rom_T2.txt"
ref_out = "data123.txt"
f = io.open(corpus, mode ="r", encoding = "utf-8")
#corpus1 = list(f) 
lines = f.readlines()
sentences =[]
for line in lines:
    mqul= line.split()
    #print(mqul)
    sentences.append(mqul)
print('Type of corpus:', type(sentences))
print('Length of corpus:',len(sentences))
print(sentences[0])


model = Word2Vec(sentences = sentences, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)
model.init_sims(replace = True)
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')
model.save('word2vec_model')
model = Word2Vec.load('word2vec_model')
model.save('model.bin')
model = Word2Vec.load('model.bin')
print(model)

my_dict = dict({})
for i, word in enumerate(model.wv.vocab):
    my_dict[word] = model.wv[word]
    output = open('filey.pkl','wb')
    pickle.dump(my_dict,output)
output.close()
        
       

print("end")

    