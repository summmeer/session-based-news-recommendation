# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os, sys
import tensorflow as tf
import numpy as np
import argparse, random
import pandas as pd
import pickle
from model import GRU4Rec
from utils import data_partition

unfold_max = 20
error_during_training = False

class Args():
    is_training = True
    layers = 1
    rnn_size = 100
    n_epochs = 10
    batch_size = 50
    keep_prob=1
    learning_rate = 0.001
    decay = 0.98
    decay_steps = 2*1e3
    sigma = 0.0001
    init_as_normal = False
    grad_cap = 0
    loss = 'cross-entropy'
    final_act = 'softmax'
    hidden_act = 'tanh'
    n_items = -1
    n_users = 1000
    init_from = None
    eval_point = 1*1e2

def parseArgs():
    args = Args()
    parser = argparse.ArgumentParser(description='LSTM4Rec args')
    parser.add_argument('--dataset', default='../../data/adressa/TCAR-mid/Normal-2/')
    parser.add_argument('--fold', default=1)
    parser.add_argument('--layer', default=2, type=int)
    parser.add_argument('--size', default=250, type=int)
    parser.add_argument('--batch', default=256, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--dr', default=0.98, type=float)
    parser.add_argument('--ds', default=400, type=int)
    parser.add_argument('--keep', default='1.0', type=float)
    parser.add_argument('--init_from', default=None, type=str)
    command_line = parser.parse_args()
    
    args.dataset = command_line.dataset
    args.fold = command_line.fold
    args.layers = command_line.layer
    args.batch_size = command_line.batch
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.decay = command_line.dr
    args.decay_steps = command_line.ds
    args.rnn_size = command_line.size
    args.keep_prob = command_line.keep
    return args

def train(args):
    # Read train and test data. 

    [train_x, train_y, test_x, test_y, item_dict, content_weight] = data_partition(args.dataset, args.fold)

    # df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
    # category_id = df_data['category_id'].tolist()
    ## adressa
    category_id = pickle.load(open('/home/sansa/dataset/Adressa/articles_category.pkl', 'rb'))
    reverse_item = {}
    for idx, cnt in item_dict.items():
        reverse_item[cnt-1] = idx

    n_items = len(item_dict)
    args.n_items = n_items
    print('#Items: {}'.format(n_items))
    print('#Training sessions: {}'.format(len(train_x)))
    sys.stdout.flush()
    # set gpu configuations.
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        model = GRU4Rec(args, content_weight)
    
        sess.run(tf.global_variables_initializer())
        print('Randomly initialize model')
        valid_losses = []
        num_batches = len(train_x) // args.batch_size 

        data = list(zip(train_x, train_y))
        random.shuffle(data)
        train_x, train_y = zip(*data)
        maxrecall = 0.0
        for epoch in range(args.n_epochs):
            epoch_cost = []
            for k in range(num_batches+1):
                if not k == num_batches:
                    in_data = train_x[k*args.batch_size: (k+1)*args.batch_size]
                    out_data = train_y[k*args.batch_size: (k+1)*args.batch_size]
                else:
                    in_data = train_x[k*args.batch_size:]
                    out_data = train_y[k*args.batch_size:]
                fetches = [model.nan, model.cost, model.global_step, model.lr, model.train_op]
                feed_dict = {model.X: in_data, model.Y: out_data}
                xnan, cost, step, lr, _ = sess.run(fetches, feed_dict)
                epoch_cost.append(cost)
                if np.isnan(cost):
                    print(str(epoch) + ':Nan error!')
                    return
                if step == 1 or step % args.eval_point == 0:
                    avgc = np.mean(epoch_cost)
                    print('Epoch {}\tProgress {}/{}\tloss: {:.6f}'.format(epoch, k, num_batches, avgc))

            valid_loss, recall = eval_validation(model, sess, test_x, test_y, args.fold, category_id, reverse_item, epoch)
            valid_losses.append(valid_loss)
            maxrecall = max(recall, maxrecall)
            print('######')
            print('max recall:', maxrecall)
            print('######')
            print('Evaluation loss after step {}: {:.6f}'.format(step, valid_loss))

def eval_validation(model, sess, test_x, test_y, fold, category_id, reverse_item, epoch):
    recall_l, ndcg_l, ild = [], [], []
    predictRes = set()
    total_loss = []
    num_batches = len(test_x) // args.batch_size 
    print('#Testing sessions: {}'.format(len(test_x)))
    print('eval....', num_batches)
    for k in range(num_batches+1):
        if not k == num_batches:
            in_data = test_x[k*args.batch_size: (k+1)*args.batch_size]
            out_data = test_y[k*args.batch_size: (k+1)*args.batch_size]
        else:
            in_data = test_x[k*args.batch_size:]
            out_data = test_y[k*args.batch_size:]
        feed_dict = {model.X: in_data, model.Y: out_data}
        predictions, masks, loss = sess.run([model.logits, model.mask, model.cost], feed_dict=feed_dict)
        total_loss.append(loss)
        # print(prediction.shape, batch_x, batch_y)
        hit, ndcg, ild, predictRes = _metric_at_k(predictions, masks, in_data, out_data, epoch, predictRes, fold, reverse_item, category_id)
        recall_l += hit
        ndcg_l += ndcg
    print('recall and ndcg', np.mean(recall_l), np.mean(ndcg_l))
    print('ild@20.......', np.mean(ild))
    print('diversity=======', len(predictRes))

    return np.mean(total_loss), np.mean(recall_l)

def printData(filename, batch_in, batch_out, batch_pred):
    file = open('saved/'+ filename + '.txt','a+')
    file.write('# batch in: {} # batch out: {} # batch pred: {} \n'.format(str(batch_in), str(batch_out), str(batch_pred)))
    file.close()


def getILD(category_id, recList, reverse_item):
    score = 0
    n = len(recList)
    for i in range(0, n):
        for j in range(0, n):
            if j!=i and category_id[reverse_item[recList[i]]]!=category_id[reverse_item[recList[j]]]:
                score += 1
    return score/(n*(n-1))

def _metric_at_k(prediction, mask, inp, label, epoch, predictRes, fold, reverse_item, category_id, k=20):
    lastid = int(np.sum(mask)-1)
    prediction = prediction.reshape((-1, 20, args.n_items+1))
    # print('shape', prediction.shape)
    mask = mask.reshape((-1, 20))
    # print('shape', mask.shape)
    # print('shape', len(inp))
    # print('shape', len(label))
    hit_at_k = []
    ndcg_at_k = []
    ild_k = []

    for index in range(prediction.shape[0]):    
        lastid = int(np.sum(mask[index])-1)
        # lastid = 0
        y = label[index][lastid]
        pred_ = prediction[index, lastid, 1:]
        predict_list = np.argsort(pred_).tolist()[::-1][:k]
        # printData('newAD_Normal2_predict_'+str(fold)+'_' + str(epoch), inp[index][:lastid+1], y-1, predict_list)
        ndcg = 0.0
        hit = 0.0
        ild_k.append(getILD(category_id, predict_list, reverse_item))
        for p in predict_list:
            predictRes.add(p)
        if y-1 in predict_list:
            rank = predict_list.index(y-1) + 1
            ndcg = 1. / (np.log2(1.0 + rank))
            hit = 1.0
        hit_at_k.append(hit)
        ndcg_at_k.append(ndcg)
    # print('in', inp[index], 'out', label[index])
    # print('y', y, 'pred', predict_list)
    return hit_at_k, ndcg_at_k, ild_k, predictRes


if __name__ == '__main__':
    args = parseArgs()
    print('rnn size: {}\tlayer: {}\tbatch: {}\tepoch: {}\tkeep: {}'.format(args.rnn_size, args.layers, args.batch_size, args.n_epochs, args.keep_prob))
    train(args)
