import sys
import copy
import random
import numpy as np
import pickle


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

def printData(filename, batch_in, batch_out, batch_pred):
    file = open('saved/'+ filename + '.txt','a+')
    file.write('# batch in: {} # batch out: {} # batch pred: {} \n'.format(str(batch_in), str(batch_out), str(batch_pred)))
    file.close()

def getILD(category_id, recList, reverse_item):
    score = 0
    n = len(recList)
    for i in range(0, n):
        for j in range(0, n):
            if reverse_item[recList[i]] in category_id and reverse_item[recList[j]] in category_id:
                if j!=i and category_id[reverse_item[recList[i]]]!=category_id[reverse_item[recList[j]]]:
                    score += 1
    return score/(n*(n-1))

def getESIR(item_freq_dict_norm, recList):
    score = 0.0
    n = len(recList)
    norm = 0.0
    for i in range(1, n+1):
        if recList[i-1]+1 in item_freq_dict_norm:
            score += item_freq_dict_norm[recList[i-1]+1]/np.log2(i+1)
        else:
            score += 20/np.log2(i+1)
        norm += 1.0/np.log2(i+1)
    return score/norm

def getUnexp(inSeq, recList, category_id, reverse_item):
    score = 0
    n = len(recList)
    cnt = 0
    
    for ini in inSeq:
        if ini == 0:
            continue
        else:
            cnt += 1
        for i in range(0, n):
            if reverse_item[recList[i]] in category_id and reverse_item[ini] in category_id:
                if category_id[reverse_item[recList[i]]]!=category_id[reverse_item[ini]]:
                    score +=1
    if cnt==0 or n==0:
        return 0
    return score/(n*cnt)

def evaluate(model, test_id, test_session, item_dict, itemnum, test_predict, args, sess, foldnum, epoch, category_id, reverse_item, item_freq_dict_norm):
    NDCG_20 = 0.0
    HT_20 = 0.0
    ILD = 0.0
    ESIR = 0.0
    unEXP = 0.0
    valid_session = 0.0

    test_size = len(test_session)

    diversity = set()

    for test_i in range(test_size):
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = test_session[test_i][-1]
        idx -= 1
        for i in reversed(test_session[test_i][:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        
        predictions = model.predict(sess, [seq])
        predictions = predictions[0]
        predictList = list(predictions.argsort()[::-1][:20])
        # printData('newAD_Normal2_predict_'+str(foldnum)+'_'+str(epoch), test_session[test_i], test_predict[test_i], predictList)
        ILD += getILD(category_id, predictList, reverse_item)
        ESIR += getESIR(item_freq_dict_norm, predictList)
        unEXP += getUnexp(seq, predictList, category_id, reverse_item)
        for p in predictList:
            diversity.add(p)

        if test_predict[test_i] in predictList:
            rank = predictList.index(test_predict[test_i]) + 1
            NDCG_20 += 1 / np.log2(rank + 1)
            HT_20 += 1
        valid_session += 1
        if valid_session % 100000 == 0:
            print(valid_session, 'seq', seq[-10:])
            print(predictions.argsort()[:20])
            print(test_predict[test_i])
    print('diversity of predict dict', len(diversity))
    print('ILD@20', ILD/valid_session)
    print('ESIR@20', ESIR/valid_session)
    print('UNEXP@20', unEXP/valid_session)
    return NDCG_20 / valid_session, HT_20 / valid_session
