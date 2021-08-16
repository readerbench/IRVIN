from pathlib import Path
import pandas as pd
import gensim
import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import csv
from operator import itemgetter
from tqdm import tqdm
import tensorflow as tf
from tensorflow.data import Dataset
from tqdm import tqdm
from eval import Performance
from heapq import nlargest
from transformers import TFLongformerModel, LongformerTokenizer


#This script allows to compute a clustering on the output of the Longformer, be It pretrained, finetuned or not.



tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)
def clean_read(filename):
    return open(filename, 'r').read()\
            .replace('\n\n','. ')\
            .replace('..', '.')\
            .replace('\n', '')
def create_data_set(files, labels):
    inputs = tokenizer([clean_read(x) for x in files], return_tensors="tf", padding='max_length', max_length=2048, truncation=True)
    inputs = {k: inputs[k] for k in ['input_ids', 'attention_mask']}
    labels = tf.reshape(tf.constant(labels, dtype=tf.int32), (len(labels), 1))
    d_inputs = Dataset.from_tensor_slices(inputs)
    d_labels = Dataset.from_tensor_slices(labels)
    return inputs, labels




#Collection of filenames and labels, and storage in the right order.  
pca = decomposition.PCA()
PATH = Path("../storage/dataset_2/dataset")
iterd = PATH.iterdir()
filenames = []
labels = []
iterd = PATH.iterdir()
for cat in iterd : 
    filenames.append([str(x) for x in cat.iterdir()])
    labels.append([('IRRELEVANT_TREATED' not in str(x)) for x in cat.iterdir()])
    print(filenames)
files_train, files_test, y_train, y_test = filenames[0], filenames[1], labels[0], labels[1]
files = files_train+files_test
y = y_train+y_test
f = open('/home/pfrod/results/clustering_num_shuffled_articles.txt', 'a')
for i, (file, x) in enumerate(zip(files, y)):
    print(f'{i}'+', '+f'{file}'+', '+f'{x}', file = f)
f.close()

#Opening of files in the right format, compute embedded vectors by batches, and concatenation.  
inputs, labels = create_data_set(files, y)
labels = labels.numpy()
labels = labels.flatten()
not_trained = TFLongformerModel.from_pretrained('pretrained_finetune', from_pt = True, attention_window = 128, trainable = False)
trained = TFLongformerModel.from_pretrained('longformer_128', from_pt = True, attention_window = 128, trainable = False)
sh = tf.shape(inputs['input_ids'])
print(sh)
result = not_trained({'input_ids' : inputs['input_ids'][:8,:], 'attention_mask' :inputs['attention_mask'][:8,:]})
liste_embeddings_not_trained = []
liste_embeddings_trained = []
for i in tqdm(range(int(tf.shape(inputs['input_ids'])[0])//8)): 
    liste_embeddings_not_trained.append(not_trained({'input_ids' : inputs['input_ids'][i*8:(i+1)*8,:], 'attention_mask' :inputs['attention_mask'][:8,:]})['pooler_output'])
    liste_embeddings_trained.append(trained({'input_ids' : inputs['input_ids'][i*8:(i+1)*8,:], 'attention_mask' :inputs['attention_mask'][:8,:]})['pooler_output'])

embeddings_mat_not_trained = tf.concat(liste_embeddings_not_trained, axis = 0)
embeddings_mat_not_trained = embeddings_mat_not_trained.numpy()
embeddings_mat_not_trained_train, embeddings_mat_not_trained_test = embeddings_mat_not_trained[:len(files_train)], embeddings_mat_not_trained[len(files_train):]
embeddings_mat_trained = tf.concat(liste_embeddings_trained, axis = 0)
embeddings_mat_trained = embeddings_mat_trained.numpy()
embeddings_mat_trained_train, embeddings_mat_trained_test = embeddings_mat_trained[:len(files_train)], embeddings_mat_trained[len(files_train):]

#Clustering on resulting matrices. 
k = 2
liste = [[None] for _ in range(k)]
km = KMeans(n_clusters=k, random_state=0).fit(embeddings_mat_not_trained_train)
pred_labels_not_trained = km.predict(embeddings_mat_not_trained_test)
km = KMeans(n_clusters=k, random_state=0).fit(embeddings_mat_trained_train)
pred_labels_trained = km.predict(embeddings_mat_trained_test)

#Don't comment if you want to test affinity propagation on the matrices. 
#ap = AffinityPropagation().fit(embeddings_mat_not_trained_train)
#pred_labels_not_trained = ap.predict(embeddings_mat_not_trained_test)
#ap = AffinityPropagation().fit(embeddings_mat_trained_train)
#pred_labels_trained = ap.predict(embeddings_mat_trained_test)

#Compute metrics
perf = Performance()
print(f'accuracy not trained 1 : {perf.confusion_matrix(pred_labels_not_trained, y_test[:-2])}')
print(f'accuracy not trained 2 : {perf.confusion_matrix(1-pred_labels_not_trained, y_test[:-2])}')
print(f'accuracy trained 1 : {perf.confusion_matrix(pred_labels_trained, y_test[:-2])}')
print(f'accuracy trained 2 : {perf.confusion_matrix(1-pred_labels_trained, y_test[:-2])}')

