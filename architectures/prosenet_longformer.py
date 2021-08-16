from transformers import TFBertForSequenceClassification, BertTokenizer, TFTrainer, TFTrainingArguments, TFRobertaForSequenceClassification, RobertaTokenizer, TFRobertaModel, AdamWeightDecay, TFLongformerModel, LongformerTokenizer
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from pathlib import Path
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import sys
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
sys.path.insert(1, '/home/pfrod/architectures/tf-ProSeNet_adapted/prosenet')
from prototypes_2 import Prototypes
from projection_2 import PrototypeProjection
from tensorflow.raw_ops import RepeatDataset
import tensorflow_addons as tfa
from operator import itemgetter
from bce_weights import weighted_binary_crossentropy

#This script instanciates our main architecture, trains and evaluates It on our dataset. 


tf.keras.backend.clear_session()
gpu_act = True
if gpu_act : 
    GPU = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPU[0], True)


class InterpretableLongformerModel(tf.keras.Model):
    """
    This class correspounds to our main architecture, including Longformer as the encoder, a ProSeNet layer and a classifier downstream. 
    """
    def __init__(self, k, list_dense_params, tokenizer_vocab_len,  trainable = False, **kwargs):
        super(InterpretableLongformerModel, self).__init__(**kwargs)
        self.k = k
        self.list_dense_params = list_dense_params
        self.encoder = TFLongformerModel.from_pretrained('longformer_128_w_finetune', from_pt = True, attention_window = 128, trainable = trainable)
        self.encoder.resize_token_embeddings(tokenizer_vocab_len)
        self.prototypes_layer = Prototypes(k=self.k)
        self.list_dense = [Dense(units=param[0], activation=param[1]) for param in list_dense_params]
        self.sigmoid = Dense(units=1, activation='sigmoid')
        
    def call(self, inputs):
        x = self.similarity_vector(inputs)
        for layer in self.list_dense : 
            x = layer(x)
            x = Dropout(0.1)(x)
        return self.sigmoid(x)
    
    def similarity_vector(self, x):
        """Return the similarity vector(s) of shape (batches, k,)."""
        r_x = self.encoder(x)[1]
        return self.prototypes_layer(r_x)

#Instanciates the architecture with the hyperparameters. 
tokenizer = LongformerTokenizer.from_pretrained('../storage/tokenizer', max_length = 2048)
k = 15
list_dense_params = [(16, 'gelu'), (16, 'gelu')]#, (32, 'gelu')
global_model = InterpretableLongformerModel(k=k, list_dense_params=list_dense_params, tokenizer_vocab_len = len(tokenizer))


#Prepare data
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
    return Dataset.zip((d_inputs, d_labels))


PATH = Path("../storage/dataset_2/dataset")
iterd = PATH.iterdir()
filenames = []
labels = []
iterd = PATH.iterdir()
for cat in iterd : 
    filenames.append([str(x) for x in cat.iterdir()])
    labels.append([('IRRELEVANT_TREATED' not in str(x)) for x in cat.iterdir()])
files_train, files_test, y_train, y_test = filenames[0], filenames[1], labels[0], labels[1]
files_train, y_train = shuffle(files_train, y_train)
files_test, y_test = shuffle(files_test, y_test)
print(np.sum(y_test)/len(y_test), np.sum(y_train)/len(y_train))
f = open('/home/pfrod/results/num_shuffled_articles.txt', 'a')
for i, (file, y) in enumerate(zip(files_train, y_train)):
    print(f'{i}'+', '+f'{file}'+', '+f'{y}', file = f)
f.close()

#define datasets
data_train = create_data_set(files_train, y_train).batch(8).repeat(18)
data_test = create_data_set(files_test, y_test).batch(8).repeat(8)
for x in data_train : 
    print(x)

#define callbacks
projection = PrototypeProjection(data_train, freq = 2)
log_dir = '../results/logging_prosenet_longformer'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#fit
global_model.compile(optimizer = AdamWeightDecay(learning_rate=5e-4), loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])#1e-3
global_model.fit(data_train, epochs = 6, steps_per_epoch = 300, validation_data=data_test, callbacks=[projection, tensorboard_callback], shuffle=False, validation_steps = 101)
global_model.save_weights('interpretablelongformer_w_finetune/model')

#Reload model and evaluate
new_model = InterpretableLongformerModel(k=k, list_dense_params=list_dense_params, tokenizer_vocab_len = len(tokenizer), trainable = True)
new_model.compile(optimizer = AdamWeightDecay(2e-4), loss = weighted_binary_crossentropy(1.5), metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
new_model.load_weights('interpretablelongformer_w_finetune/model')
projection = PrototypeProjection(data_train, freq = 3)
new_model.evaluate(data_test, verbose = 2)

