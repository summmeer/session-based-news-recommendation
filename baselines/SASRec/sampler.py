import numpy as np
import random

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def neg_neighbor(neighbor_dict, itemid):
    neighor_set = neighbor_dict[itemid-1]
    t = random.choice(neighor_set)
    while t == itemid-1:
        t = random.choice(neighor_set)
    return t+1

def sample_function(session, session_id, itemnum, maxlen, neg_sample_num, neighbor_dict):
    neg_sample_num = 20
    seq = np.zeros([maxlen], dtype=np.int32)
    pos = np.zeros([maxlen], dtype=np.int32)
    neg = np.zeros([maxlen], dtype=np.int32)
    ts = set(session)
    for i in range(neg_sample_num):
        neg[i] = random_neq(1, itemnum + 1, ts)

    groudTruth = session[-1]
    idx = maxlen - 1
    pos[idx] = groudTruth
    # if random.random()<0.5:
    #     neg[idx] = neg_neighbor(neighbor_dict, groudTruth)

    for i in reversed(session[:-1]):
        seq[idx] = i
        if idx>0:
            pos[idx-1] = i
            # if random.random()<0.5:
            #     neg[idx-1] = neg_neighbor(neighbor_dict, i)
        idx -= 1
        if idx == -1: break
    return (seq, pos, neg)


class WarpSampler(object):
    def __init__(self, train_session, session_num, itemnum, batch_num, neg_sample_num, batch_size=128, maxlen=10):
        self.train_session = []
        self.session_num = session_num
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.maxlen = maxlen
        self.neg_sample_num = neg_sample_num
        self.neighbor_dict = {}
        shaffle_ids = np.random.randint(0, session_num, size=session_num)
        self.shaffle_dict = {}
        for i in range(session_num):
            self.shaffle_dict[i] = shaffle_ids[i] + 1
            self.train_session.append(train_session[shaffle_ids[i]])

    def next_batch(self, batch_i):
        if batch_i==self.batch_num:
            train_batch = self.train_session[batch_i*self.batch_size:]
        else:
            train_batch = self.train_session[batch_i*self.batch_size:(batch_i+1)*self.batch_size]
        seq_batch = []
        neg_batch = []
        pos_batch = []
        for index, session in enumerate(train_batch):
            (seq, pos, neg) = sample_function(session, self.shaffle_dict[batch_i*self.batch_size + index], self.itemnum, self.maxlen, self.neg_sample_num, self.neighbor_dict)
            seq_batch.append(seq)
            pos_batch.append(pos)
            neg_batch.append(neg)
        seq_batch = np.stack(seq_batch, axis=0)
        pos_batch = np.stack(pos_batch, axis=0)
        neg_batch = np.stack(neg_batch, axis=0)
        return (seq_batch, pos_batch, neg_batch)
