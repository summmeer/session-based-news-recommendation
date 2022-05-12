#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd

import csv
import random
import numpy as np
from nltk.tokenize import word_tokenize 
import argparse

MAX_TITLE_LENGTH=30
MAX_NEWS=20

def data_partition(fname, foldnum):
    # assume user/item index starting from 1
    train_data = pickle.load(open(fname+'train_session_'+str(foldnum)+'.txt', 'rb'))
    train_id = train_data[0]
    train_session = train_data[1]
    train_timestamp = train_data[2]
    train_predict = train_data[3]

    for i, s in enumerate(train_session):
        train_session[i] += [train_predict[i]]

    test_data = pickle.load(open(fname+'test_session_'+str(foldnum)+'.txt', 'rb'))
    test_id = test_data[0]
    test_session = test_data[1]
    test_timestamp = test_data[2]
    test_predict = test_data[3]

    item_dict = pickle.load(open(fname+'item_dict_'+str(foldnum)+'.txt', 'rb'))
    
    content_weight = pickle.load(open(fname+'content_weight_'+str(foldnum)+'.txt', 'rb'))

    return [train_id, train_session, train_timestamp, test_id, test_session, test_timestamp, test_predict, item_dict, content_weight]

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='../../data/globo/TCAR-mid/Normal-2/')
parser.add_argument('--fold', default=1)
args = parser.parse_args()

dataset = data_partition(args.dataset, args.fold)
[train_id, train_session, train_timestamp, test_id, test_session, test_timestamp, test_predict, item_dict, content_weight] = dataset

df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
category_id = df_data['category_id'].tolist()
reverse_item = {}
for idx, cnt in item_dict.items():
    reverse_item[cnt-1] = idx

train_label = []
train_user_his = []
cand_title_train = []

for session in train_session:
    clickids = session[-MAX_NEWS-1:-1]
    train_label.append(session[-1])
    clickids += [0]*(MAX_NEWS-len(clickids))
    train_user_his.append(clickids)
    # cand_title_train.append([i for i in range(1+len(item_dict))])

print('train len', len(train_label))

test_label=[]
test_user_his=[]
cand_title_test = []

for session in test_session:
    clickids = session[-MAX_NEWS-1:-1]
    test_label.append(session[-1])
    clickids += [0]*(MAX_NEWS-len(clickids))
    test_user_his.append(clickids)
    # cand_title_test.append([i for i in range(1+len(item_dict))])

print('test len', len(test_label))

train_label=np.array(train_label,dtype='int32')
train_user_his=np.array(train_user_his,dtype='int32')
cand_title_train=np.array(cand_title_train,dtype='int32')
test_label=np.array(test_label,dtype='int32')
test_user_his=np.array(test_user_his,dtype='int32')
cand_title_test=np.array(cand_title_test,dtype='int32')
print(train_user_his.shape)

def generate_batch_data(batch_size):
    idx = np.arange(len(train_label))
    np.random.shuffle(idx)
    y=train_label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    # batches = batches[:2000]
    while (True):
        for i in batches:
            user_title=train_user_his[i]
            # cand_title_tra=cand_title_train[i]
            yield (user_title, train_label[i])

def generate_batch_data_test(batch_size):
    idx = np.arange(len(test_label))
    y=test_label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            user_title = test_user_his[i]
            # cand_title_te=cand_title_test[i]
            yield (user_title, test_label[i])
        
def getILD(category_id, y_score, reverse_item):
    order = np.argsort(y_score[1:])[::-1][:20]
    score = 0
    recList = order
    n = len(recList)
    for i in range(0, n):
        for j in range(0, n):
            if j!=i and category_id[reverse_item[recList[i]]]!=category_id[reverse_item[recList[j]]]:
                score += 1
    return score/(n*(n-1))

def ndcg_score(y_true, y_score, k=20):
    order = np.argsort(y_score[1:])[::-1]
    rank = np.where(order==(y_true-1))[0][0]+1
    if rank<=k:
        return 1 / np.log2(rank + 1)
    else:
        return 0

def hr_score(y_true, y_score, k=20):
    order = np.argsort(y_score[1:])[::-1]
    rank = np.where(order==(y_true-1))[0][0]+1
    if rank<=k:
        return 1
    else:
        return 0


from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers 
from sklearn.metrics import accuracy_score, classification_report,roc_auc_score
from keras.optimizers import *
import keras
from keras.layers import *
import tensorflow as tf


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

# candidate_ids = K.variable(list([i for i in range(1+len(item_dict))]))
# candidate_input = Input(tensor=candidate_ids)
# candidate_input = Input((len(item_dict)+1,), dtype='int32')

class SimpleLayer(Layer):

  def __init__(self):
      super(SimpleLayer, self).__init__()

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.embe_dict = tf.Variable(content_weight, dtype=tf.float32, trainable=False)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, tf.transpose(self.embe_dict))
      
results=[]

keras.backend.clear_session()

HEDDEN_SIZE = 250

embedding_layer = Embedding(len(item_dict)+1, HEDDEN_SIZE, trainable=False)


his_title_input =  Input((MAX_NEWS,), dtype='int32')

titlebeh = embedding_layer(his_title_input)

attention_title = Dense(250,activation='tanh')(titlebeh)
attention_title = Flatten()(Dense(1)(attention_title))
attention_weight_title = Activation('softmax')(attention_title)
userrep_title=keras.layers.Dot((1, 1))([titlebeh, attention_weight_title])
userrep = userrep_title
# attentionvecs= Dense(250,activation='tanh')(uservecs)
# attentionvecs = Flatten()(Dense(1)(attentionvecs))
# attention_weightvecs = Activation('softmax')(attentionvecs)
# userrep=keras.layers.Dot((1, 1))([uservecs, attention_weightvecs])

# cand_titlerep = K.variable(embedding_layer.get_weights()[0])
# cand_titlerep = embedding_layer(candidate_input)
# cand_titlerep = K.transpose(cand_titlerep)
# logits = keras.layers.Dot((1,0))([userrep, cand_titlerep])
logits = SimpleLayer()(userrep)
# logits = Reshape((1+len(item_dict)))(logits)
print(logits.shape)
logits = keras.layers.Activation(keras.activations.softmax)(logits)

model = Model(inputs=his_title_input, outputs=logits)
embedding_layer.set_weights([content_weight])
# print(embedding_layer.get_weights()[0].shape)
#    model.compile(loss=['categorical_crossentropy','mae'], optimizer=Adam(lr=0.001), metrics=['acc','mae'],loss_weights=[1.,0.4])
model.compile(loss=['sparse_categorical_crossentropy'], optimizer=Adam(lr=0.001))

batch_size = 512
for ep in range(10):
    traingen=generate_batch_data(batch_size)
    print('batch', len(train_label)//batch_size)
    model.fit_generator(traingen, epochs=10,steps_per_epoch=len(train_label)//batch_size)

print('training finished')

testgen = generate_batch_data_test(batch_size)

hr = []
ndcg = []
ild = []
pred = model.predict_generator(testgen, steps=len(test_label)//batch_size,verbose=1)

for i, score in enumerate(pred):
    hr.append(hr_score(test_label[i], score, k=20))
    ndcg.append(ndcg_score(test_label[i], score, k=20))
    ild.append(getILD(category_id, score, reverse_item))
print('hr', np.mean(hr), np.mean(ndcg), np.mean(ild))