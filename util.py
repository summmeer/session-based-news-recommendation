# coding=utf-8
import random
import numpy as np
import pickle
import tensorflow as tf
import time

def cau_metrics(preds, labels, cutoff = 20):
    recall = []
    mrr = []
    ndcg = []

    for batch, b_label in zip(preds, labels):
        ranks = (batch[b_label] < batch).sum() + 1
        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)
        ndcg.append(1/np.log2(ranks + 1) if ranks <= cutoff else 0.0)
    return recall, mrr, ndcg

def data_partition(fname, foldnum):
    # assume user/item index starting from 1
    len_dict_train = pickle.load(open(fname + 'len_dict_train' + str(foldnum) + '.pkl', 'rb'))
    session_dict_train = pickle.load(open(fname + 'session_dict_train_' + str(foldnum) + '.pkl', 'rb'))
    session_time_dict_train = pickle.load(open(fname + 'session_time_dict_train' + str(foldnum) + '.pkl', 'rb'))

    len_dict_test = pickle.load(open(fname + 'len_dict_test' + str(foldnum) + '.pkl', 'rb'))
    session_dict_test = pickle.load(open(fname + 'session_dict_test_' + str(foldnum) + '.pkl', 'rb'))
    session_time_dict_test = pickle.load(open(fname + 'session_time_dict_test' + str(foldnum) + '.pkl', 'rb'))

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
    for i, s in enumerate(test_session):
        test_session[i] += [test_predict[i]]

    item_dict = pickle.load(open(fname + 'item_dict_' + str(foldnum) + '.txt', 'rb'))
    neighbor_dict = pickle.load(open('/home/sansa/recsys/TCAR/data/mind/sess_impressions.mid', 'rb'))
    # neighbor_dict = pickle.load(open(fname + 'neighbor_' + str(foldnum) + '.txt', 'rb'))

    content_emb = pickle.load(open(fname + 'content_weight_' + str(foldnum) + '.txt', 'rb'))
    
    publish_time = pickle.load(open(fname + 'publish_time_' + str(foldnum) + '.txt', 'rb'))
    # candidate_mask = pickle.load(open(fname + 'candidate_masks_' + str(foldnum) + '.txt', 'rb'))
    candidate_mask = None

    return (len_dict_train, session_dict_train, session_time_dict_train), (len_dict_test, session_dict_test, session_time_dict_test), item_dict, neighbor_dict, content_emb, publish_time, candidate_mask
    # return (train_session, train_timestamp), (test_session, test_timestamp), item_dict, neighbor_dict, content_emb, publish_time, candidate_mask

# activer
def activer(inputs, case='tanh'):
    '''
    The active enter. 
    '''
    switch = {
        'tanh': tanh,
        'relu': relu,
        'sigmoid': sigmoid,
    }
    func = switch.get(case, tanh)
    return func(inputs)


def tanh(inputs):
    '''
    The tanh active. 
    '''
    return tf.nn.tanh(inputs)

def sigmoid(inputs):
    '''
    The sigmoid active.
    '''
    return tf.nn.sigmoid(inputs)


def relu(inputs):
    '''
    The relu active. 
    '''
    return tf.nn.relu(inputs)

def normalizer(inputs, axis=1):
    '''
    inputs: the inputs should use to count the softmax. 
    axis: the axis of the softmax on. 
    '''
    # max_nums = tf.reduce_max(inputs, axis, keep_dims=True)
    inputs = tf.exp(inputs)
    _sum = tf.reduce_sum(inputs, axis=axis, keep_dims=True) + 1e-9
    return inputs / _sum

def save_model(self, sess, args, saver=None):
    suf = time.strftime("%Y%m%d%H%M", time.localtime()) + '-' + args['dataset'] + '-' + args['splitway'] + '-' + str(args['foldnum'])
    if saver is not None:
        saver.save(sess, args['modelpath'] + "model.ckpt-" + suf)
    return args['modelpath'] + ".ckpt-" + suf