
# coding: utf-8

# In[42]:


from glob import glob
import re
import string
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import nltk #may need for punkt tokenizer
import pandas as pd
import glob
import os
from random import shuffle


# In[107]:


import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from gensim.models.doc2vec import LabeledSentence


# In[167]:


#read labels for files (filennames in our case)
from os import listdir
from os.path import isfile, join
docLabels = []
docLabels = [f for f in listdir("data/") if f.endswith('.txt')]


# In[168]:


#read content of files - memory heavy oepration
data =[]
for doc in docLabels:
    temp = open("data/"+str(doc)).read()
    data.append(temp)


# In[228]:


#customize LabeledSentence Class from gensim to accomodate line to vec parsin
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(doc.split(),[self.labels_list[idx]])


# In[229]:


#compute array of sentence arrays (word tokens + labels)
it = LabeledLineSentence(data, docLabels)


# In[230]:


#define gensim model
model = gensim.models.Doc2Vec(size=300, window=5, min_count=2, workers=11,alpha=0.025, min_alpha=0.025)


# In[231]:


#build vocabulary
model.build_vocab(it)


# In[232]:


#train the model
for epoch in range(10):
    model.train(it,total_examples=model.corpus_count, epochs = model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(it,total_examples=model.corpus_count, epochs = model.iter)


# In[207]:


#save model to disk
model.save("doc2vec.model")


# In[235]:


#loading the model
d2v_model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")


# In[237]:


#doocument vector by specifying file number
docvec = d2v_model.docvecs[2]
print (docvec)


# In[239]:


#document vecotr by specifying file name
docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
print (docvec)


# In[241]:


#to get most similar document with similarity scores using document-index
similar_doc = d2v_model.docvecs.most_similar(14) 
print (similar_doc)


# In[243]:


#to get most similar document with similarity scores using document- name
sims = d2v_model.docvecs.most_similar('2.txt')
print (sims)

