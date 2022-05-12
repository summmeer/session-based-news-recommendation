import pickle
import pandas as pd
import argparse

def processData(train_data):
    train_id = train_data[0]
    train_session = train_data[1]
    train_timestamp = train_data[2]
    train_predict = train_data[3]
    data_train = {
        'SessionId': [],
        'ItemId': [],
        'Cate': [],
        'Time': []
    }

    for i, s in enumerate(train_session):
        train_session[i] += [train_predict[i]]
        timei = 0
        for itemid in train_session[i]:
            data_train['SessionId'].append(train_id[i])
            data_train['ItemId'].append(itemid)
            data_train['Cate'].append(1)
            data_train['Time'].append(train_timestamp[i][timei]['click_t'])
            timei += 1

    data_train = pd.DataFrame(data_train)
    # data_train.to_csv('./dataTrainLen/new_STAMP_adressa_train_full_minlen_2_'+str(epoch)+'.txt', sep='\t', index=False)
    return data_train

def main(args):
    if args.split_way == 'Normal-2':
        folds = [1,2,3,4,5,6,7]
    else:
        folds = [1,2,3]
    # folds = [1]
    savefile = 'GRU-STAMP-mid'
    for fold in folds:
        train_data = pickle.load(open('../data/adressa/CBCF-mid/' + args.split_way + '/train_session_' + str(fold) + '.txt', 'rb'))
        data_train = processData(train_data)
        data_train.to_csv('../data/adressa/' + savefile + '/' + args.split_way + '/train_session_' + str(fold) + '.txt', sep='\t', index=False)

        test_data = pickle.load(open('../data/adressa/CBCF-mid/' + args.split_way + '/test_session_' + str(fold) + '.txt', 'rb'))
        data_test = processData(test_data)
        data_test.to_csv('../data/adressa/' + savefile + '/' + args.split_way + '/test_session_' + str(fold) + '.txt', sep='\t', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_way', default='Normal-2', type=str, choices=['Normal-2', 'TrainLen-1', 'TestLen-1'], help='Choose different split ways')
    args = parser.parse_args()

    main(args)