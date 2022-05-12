#!/usr/bin/env python
# coding: utf-8
import pickle
import pandas as pd

import csv
import random
import numpy as np
from nltk.tokenize import word_tokenize 
import argparse

MAX_TITLE_LENGTH=15
MAX_NEWS=10

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
parser.add_argument('--dataset', default='../../data/adressa/TCAR-mid/Normal/')
parser.add_argument('--fold', default=1)
args = parser.parse_args()

dataset = data_partition(args.dataset, args.fold)
[train_id, train_session, train_timestamp, test_id, test_session, test_timestamp, test_predict, item_dict, content_weight] = dataset

category_id = pickle.load(open('/home/sansa/dataset/Adressa/articles_category.pkl', 'rb'))
reverse_item = {}
for idx, cnt in item_dict.items():
    reverse_item[cnt] = idx

articles_content = pickle.load(open('/home/sansa/recsys/TCAR/data/adressa/articles_titles_2.pkl', 'rb'))
news = {} # news dict, news id and title

for c_id, a_id in reverse_item.items():
    title = articles_content[a_id]
    body_tokens = word_tokenize(title.lower())
    news[c_id] = [body_tokens, body_tokens, max(len(body_tokens),1)]

train_label = []
train_user_his = []
train_label_speed=[]
train_user_his_satis=[] 

for idx, session in enumerate(train_session):
    clickids = session[-MAX_NEWS-1:-1]
    train_label.append(session[-1])

    dwelltime=[x['active_t'] for x in train_timestamp[idx]][-MAX_NEWS-1:-1]
    clicknewslen=[news[x][2] for x in session][-MAX_NEWS-1:-1]
    readspeed=[clicknewslen[_]/(dwelltime[_]+1) for _ in range(len(dwelltime))]
    avgreadspeed=np.mean(readspeed)
    readspeednorm=[min(max(np.log2(avgreadspeed/x)+6,0),13) for x in readspeed]
    train_user_his_satis.append(readspeednorm+[0]*(MAX_NEWS-len(clickids)))

    clickids += [0]*(MAX_NEWS-len(clickids))
    train_user_his.append(clickids)

    dwelltimewordvec=[train_timestamp[idx][-1]['active_t']]
    wordvecnewslen=[news[session[-1]][2]]
    readspeedwordvec=[wordvecnewslen[_]/(dwelltimewordvec[_]+1) for _ in range(len(dwelltimewordvec))]
    readspeedwordvecnorm=[min(max(np.log2(avgreadspeed/x)+6,0),13)/13. for x in readspeedwordvec]
    train_label_speed.append(readspeedwordvecnorm)

print('train label len', len(train_label))

test_label=[]
test_user_his=[]
test_user_his_satis=[]

for idx, session in enumerate(test_session):
    clickids = session[-MAX_NEWS:]
    test_label.append(test_predict[idx])

    dwelltime=[x['active_t'] for x in test_timestamp[idx]][-MAX_NEWS-1:-1]
    clicknewslen=[news[x][2] for x in session][-MAX_NEWS:]
    readspeed=[clicknewslen[_]/(dwelltime[_]+1) for _ in range(len(dwelltime))]
    avgreadspeed=np.mean(readspeed)
    readspeednorm=[min(max(np.log2(avgreadspeed/x)+6,0),13) for x in readspeed]
    test_user_his_satis.append(readspeednorm+[0]*(MAX_NEWS-len(clickids)))

    clickids += [0]*(MAX_NEWS-len(clickids))
    test_user_his.append(clickids)

print('test label len', len(test_label))

word_dict_raw={'PADDING': [0,999999]}

for i in news:
    for j in news[i][0]:
        if j in word_dict_raw:
            word_dict_raw[j][1]+=1
        else:
            word_dict_raw[j]=[len(word_dict_raw),1]

word_dict={}
for i in word_dict_raw:
    if word_dict_raw[i][1]>=1:
        word_dict[i]=[len(word_dict),word_dict_raw[i][1]]
print('size word_dict', len(word_dict),len(word_dict_raw))

embdict={} 
import pickle
# with open('/home/sansa/dataset/glove.840B.300d.txt','rb')as f: 
#     while True:
#         line=f.readline()
#         if len(line)==0:
#             break
#         line=line.split() 
#         word=line[0].decode() 
#         if len(word) != 0:
#             vec=[float(x) for x in line[1:]]
#             if word in word_dict:
#                 embdict[word]=vec 

with open('/home/sansa/dataset/no.tsv')as f: 
    read_tsv = csv.reader(f, delimiter="\t")
    word = ""
    vec = ""
    for row in read_tsv:
        if len(row)>1:
            if len(word) != 0:
                vec_f = [float(x) for x in vec[1:-1].split()]
                if word in word_dict:
                    embdict[word]=vec_f
            word = row[1]
            vec = row[2]
        else:
            vec += row[0]
        
print('size embdict', len(embdict))

from numpy.linalg import cholesky
emb_table=[0]*len(word_dict) 
wordvec=[]
for i in embdict.keys():
    emb_table[word_dict[i][0]]=np.array(embdict[i],dtype='float32')
    wordvec.append(emb_table[word_dict[i][0]])
wordvec=np.array(wordvec,dtype='float32')

mu=np.mean(wordvec, axis=0)
Sigma=np.cov(wordvec.T)

norm=np.random.multivariate_normal(mu, Sigma, 1) 

for i in range(len(emb_table)):
    if type(emb_table[i])==int:
        emb_table[i]=np.reshape(norm, 300)
        
emb_table[0]=np.zeros(300,dtype='float32')
emb_table=np.array(emb_table,dtype='float32')

news_words=[[0]*MAX_TITLE_LENGTH]

for i in news:
    line=[]
    for word in news[i][0]:
        if word in word_dict:
            line.append(word_dict[word][0])
    line=line[:MAX_TITLE_LENGTH]
    news_words.append(line+[0]*(MAX_TITLE_LENGTH-len(line)))

news_words=np.array(news_words,dtype='int32') 

train_label=np.array(train_label,dtype='int32')
train_user_his=np.array(train_user_his,dtype='int32')
train_user_his_satis=np.array(train_user_his_satis,dtype='int32')
train_label_speed=np.array(train_label_speed,dtype='float32')
test_label=np.array(test_label,dtype='int32')
test_user_his=np.array(test_user_his,dtype='int32')
test_user_his_satis=np.array(test_user_his_satis,dtype='int32')

def generate_batch_data(batch_size):
    idx = np.arange(len(train_label))
    np.random.shuffle(idx)
    y=train_label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    # batches = batches[:2000]
    while (True):
        for i in batches:
            user_title=news_words[train_user_his[i]]
            user_body = user_title
            pos_body = news_words[train_label[i]]
            user_satis=train_user_his_satis[i]
            yield ([user_title] + [user_body, user_satis, pos_body], [train_label[i], train_label_speed[i]])

def generate_batch_data_test(batch_size):
    idx = np.arange(len(test_label))
    y=test_label
    batches = [idx[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            user_title=news_words[test_user_his[i]]
            user_body = user_title
            user_satis=test_user_his_satis[i]

            yield ([user_title, user_body, user_satis], [test_label[i]])
        
def getILD(category_id, y_score, reverse_item):
    order = np.argsort(y_score[1:])[::-1][:20]
    score = 0
    recList = order
    n = len(recList)
    for i in range(0, n):
        for j in range(0, n):
            if j!=i:
                if reverse_item[recList[i]+1] in category_id and reverse_item[recList[j]+1] in category_id:
                    if category_id[reverse_item[recList[i]+1]]!=category_id[reverse_item[recList[j]+1]]:
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


results=[]

keras.backend.clear_session()
    
    
title_input = Input(shape=(MAX_TITLE_LENGTH,), dtype='int32')

body_input= Input(shape=(MAX_TITLE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(word_dict), 300, weights=[emb_table],trainable=True)

embedded_sequences = embedding_layer(title_input)
drop_emb=Dropout(0.2)(embedded_sequences)

selfatt = Attention(16,16)([drop_emb,drop_emb,drop_emb]) 
drop_selfatt=Dropout(0.2)(selfatt)

attention = Dense(200,activation='tanh')(drop_selfatt)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
title_rep=keras.layers.Dot((1, 1))([drop_selfatt, attention_weight])


title_encoder = Model([title_input], title_rep)

embedded_sequences2 = embedding_layer(body_input)
drop_emb2=Dropout(0.2)(embedded_sequences2)

selfatt2 = Attention(16,16)([drop_emb2,drop_emb2,drop_emb2]) 
drop_selfatt2=Dropout(0.2)(selfatt2)

attention2 = Dense(200,activation='tanh')(drop_selfatt2)
attention2 = Flatten()(Dense(1)(attention2))
attention_weight2 = Activation('softmax')(attention2)
body_rep=keras.layers.Dot((1, 1))([drop_selfatt2, attention_weight2])

bodyEncodert = Model([body_input], body_rep)

his_title_input =  Input((MAX_NEWS, MAX_TITLE_LENGTH,), dtype='int32') 
his_body_input = Input((MAX_NEWS,MAX_TITLE_LENGTH,), dtype='int32')

titlebeh=TimeDistributed(title_encoder)(his_title_input)
bodybeh=TimeDistributed(bodyEncodert)(his_body_input)

attention_title = Dense(200,activation='tanh')(titlebeh)
attention_title = Flatten()(Dense(1)(attention_title))
attention_weight_title = Activation('softmax')(attention_title)
userrep_title=keras.layers.Dot((1, 1))([titlebeh, attention_weight_title])

attention_body = Dense(200,activation='tanh')(bodybeh)
attention_body = Flatten()(Dense(1)(attention_body))
attention_weight_body = Activation('softmax')(attention_body)
userrep_body=keras.layers.Dot((1, 1))([bodybeh, attention_weight_body])


time_input = Input(shape=(MAX_NEWS,), dtype='int32')
timeembedding_layer = Embedding(100, 50, trainable=True)(time_input)
attention_satis = Lambda(lambda x:K.sum(x,axis=-1))(multiply([bodybeh, Dense(256)(timeembedding_layer) ]))
attention_weight_satis = Activation('softmax')(attention_satis)
userrep_satis=keras.layers.Dot((1, 1))([bodybeh, attention_weight_satis])
userrep_read=add([userrep_body,userrep_satis])

uservecs = concatenate([Lambda(lambda x: K.expand_dims(x,axis=1))(vec) for vec in [userrep_title,userrep_read]],axis=1)
attentionvecs= Dense(200,activation='tanh')(uservecs)
attentionvecs = Flatten()(Dense(1)(attentionvecs))
attention_weightvecs = Activation('softmax')(attentionvecs)
userrep=keras.layers.Dot((1, 1))([uservecs, attention_weightvecs])
print('userrep shape', userrep.get_shape())

class SimpleLayer(Layer):

  def __init__(self):
      super(SimpleLayer, self).__init__()

  def build(self, input_shape):  # Create the state of the layer (weights)
    self.content_dict = tf.Variable(news_words, dtype=tf.int32, trainable=False)
    print(self.content_dict.get_shape())
    self.cont_encode = bodyEncodert(self.content_dict)
    print(self.cont_encode.get_shape())

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, tf.transpose(self.cont_encode))

cand_pos_body = keras.Input((MAX_TITLE_LENGTH,)) 
cand_one_bodyrep = bodyEncodert([cand_pos_body])

dense1=Dense(100)
dense2=Dense(100)
predspeed = keras.layers.dot([dense1(userrep), dense2(cand_one_bodyrep)], axes=-1) 
logits = SimpleLayer()(userrep)
logits = keras.layers.Activation(keras.activations.softmax)(logits)

model = Model([his_title_input] + [his_body_input,time_input,cand_pos_body], [logits,predspeed])
model.compile(loss=['sparse_categorical_crossentropy','mae'], optimizer=Adam(lr=0.0001), metrics=['acc','mae'],loss_weights=[1.,0.4])

model_test = keras.Model([his_title_input,his_body_input,time_input], logits)

batch_size = 256

testgen = generate_batch_data_test(batch_size)

for ep in range(1):
    traingen=generate_batch_data(batch_size)
    print('batch', len(train_label)//batch_size)
    model.fit_generator(traingen, epochs=3, steps_per_epoch=len(train_label)//batch_size)
    
    print('training finished')
    hr = []
    ndcg = []
    ild = []
    pred = model_test.predict_generator(testgen, steps=len(test_label)//batch_size,verbose=1)

    for i, score in enumerate(pred):
        hr.append(hr_score(test_label[i], score, k=20))
        ndcg.append(ndcg_score(test_label[i], score, k=20))
        ild.append(getILD(category_id, score, reverse_item))
    print('hr', np.mean(hr), np.mean(ndcg), np.mean(ild))


