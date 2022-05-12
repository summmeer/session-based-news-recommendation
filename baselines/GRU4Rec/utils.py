import os
import pandas as pd
import numpy as np
import pickle

PATH_TO_TRAIN = '../../data/globo/GRU-STAMP-mid/Normal-2/train_session_1.txt'
PATH_TO_TEST = '../../data/globo/GRU-STAMP-mid/Normal-2/test_session_1.txt'

def get_item():
    train = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={0:str, 1:str, 2:np.float32})
    test = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={0:str, 1:str, 2:np.float32})
    data = pd.concat([train, test])
    return data.ItemId.unique()


def _load_data(f, max_len):
    """
    Data format in file f:
    SessionId\tItemId\tTimestamp\n
    """
    
    items = get_item()
    item2idmap = dict(zip(items, range(1, 1+items.size))) 
    n_items = len(item2idmap)
    data = pd.read_csv(f, sep='\t', dtype={0:str, 1:str, 2:np.float32})    
    data['ItemId'] = data['ItemId'].map(item2idmap)
    data = data.sort_values(by=['Time']).groupby('SessionId')['ItemId'].apply(list).to_dict()
    new_x = []
    new_y = []
    for k, v in data.items():
        x = v[:-1]
        y = v[1:]
        if len(x) < 1:
            continue
        padded_len = max_len - len(x)
        if padded_len > 0:
            x.extend([0] * padded_len)
            y.extend([0] * padded_len)
        new_x.append(x[:max_len])
        new_y.append(y[:max_len])
    return (new_x, new_y, n_items)

def load_train(max_len):
    return _load_data(PATH_TO_TRAIN, max_len)

def load_test(max_len):
    return _load_data(PATH_TO_TEST, max_len)


def split_data(data, max_len):
    new_x = []
    new_y = []
    for v in data:
        x = v[:-1]
        y = v[1:]
        if len(x) < 1:
            continue
        padded_len = max_len - len(x)
        if padded_len > 0:
            x.extend([0] * padded_len)
            y.extend([0] * padded_len)
        new_x.append(x[:max_len])
        new_y.append(y[:max_len])
    return (new_x, new_y)

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

    for i, s in enumerate(test_session):
        test_session[i] += [test_predict[i]]

    item_dict = pickle.load(open(fname+'item_dict_'+str(foldnum)+'.txt', 'rb'))
    
    content_weight = pickle.load(open(fname+'content_weight_'+str(foldnum)+'.txt', 'rb'))

    train_x, train_y = split_data(train_session, 20)
    test_x, test_y = split_data(test_session, 20)

    return [train_x, train_y, test_x, test_y, item_dict, content_weight]

if __name__ == '__main__':
    x_test, y_test, n_items = load_test(20)
    print(x_test[10000:11000])
    print(y_test[10000:11000])
    print(n_items)
