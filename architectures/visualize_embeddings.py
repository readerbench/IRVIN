
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
from tensorboard.plugins import projector
from tensorflow.data import Dataset
from tqdm import tqdm
from eval import Performance
from heapq import nlargest
from transformers import TFLongformerModel, LongformerTokenizer
import os
import csv


#Script used to visualize results of clustering on Tensorboard Projector. 


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





pca = decomposition.PCA()
PATH = Path("../storage/dataset_2/dataset")
iterd = PATH.iterdir()
filenames = []
labels = []
iterd = PATH.iterdir()
for cat in iterd : 
    filenames.append([str(x) for x in cat.iterdir()])
    labels.append([('IRRELEVANT_TREATED' not in str(x)) for x in cat.iterdir()])

files_train, files_test, y_train, y_test = filenames[0], filenames[1], labels[0], labels[1]
files = files_train+files_test
y = y_train+y_test
f = open('/home/pfrod/results/clustering_num_shuffled_articles.txt', 'a')
for i, (file, x) in enumerate(zip(files, y)):
    print(f'{i}'+', '+f'{file}'+', '+f'{x}', file = f)
f.close()

inputs, labels = create_data_set(files, y)
labels = labels.numpy()[:1624]
labels = labels.flatten()
#Allows to visualize embedding for the finetuned architecture, just change the input file if you want to visualize embeddings without finetune. 
trained = TFLongformerModel.from_pretrained('longformer_128', from_pt = True, attention_window = 128, trainable = False)
sh = tf.shape(inputs['input_ids'])
result = not_trained({'input_ids' : inputs['input_ids'][:8,:], 'attention_mask' :inputs['attention_mask'][:8,:]})
liste_embeddings_not_trained = []
liste_embeddings_trained = []
for i in tqdm(range(tf.shape(inputs['input_ids'])[0]//8)): 
    liste_embeddings_trained.append(trained({'input_ids' : inputs['input_ids'][i*8:(i+1)*8,:], 'attention_mask' :inputs['attention_mask'][:8,:]})['pooler_output'])
embeddings_mat_trained = tf.concat(liste_embeddings_trained, axis = 0)
embeddings_mat_trained = embeddings_mat_trained.numpy()

# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='/home/pfrod/architectures/logs/viz_embeddings_train_nopretrain/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Save Labels separately on a line-by-line manner.
with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
    tsv_writer = csv.writer(f, delimiter='\t')
    for label in labels:
        tsv_writer.writerow([label])
with open(os.path.join(log_dir, 'vectors.tsv'), 'w') as f :
    tsv_writer = csv.writer(f, delimiter='\t') 
    for vector in embeddings_mat_trained : 
        tsv_writer.writerow(vector)
