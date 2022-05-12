import pickle
import numpy as np
from STAN import STAN
import math
import time
from datetime import datetime
import pandas as pd

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


##  globo
# df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
# category_id = df_data['category_id'].tolist()
## adressa
category_id = pickle.load(open('/home/sansa/dataset/Adressa/articles_category.pkl', 'rb'))

folds = [1, 2, 3, 4, 5, 6, 7]
for fold in folds:
    # load data [id,x,t,y]
    train_data = pickle.load(open('../../data/adressa/TCAR-mid/Normal-2/train_session_'+str(fold)+'.txt', 'rb'))
    train_id = train_data[0]
    train_session = train_data[1]
    train_timestamp = [datetime.timestamp(t[0]['click_t']) for t in train_data[2]]
    train_predict = train_data[3]

    for i, s in enumerate(train_session):
        train_session[i] += [train_predict[i]]

    test_data = pickle.load(open('../../data/adressa/TCAR-mid/Normal-2/test_session_'+str(fold)+'.txt', 'rb'))
    item_dict = pickle.load(open('../../data/adressa/TCAR-mid/Normal-2/item_dict_'+str(fold)+'.txt', 'rb'))
    reverse_item = {}
    for idx, cnt in item_dict.items():
        reverse_item[cnt] = idx

    test_id = test_data[0]
    test_session = test_data[1]
    test_timestamp = [datetime.timestamp(t[0]['click_t']) for t in test_data[2]]
    test_predict = test_data[3]

    print('size of the label', len(set(test_predict)))
    model = STAN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500,
                factor1=True, l1=2, factor2=True, l2=80 * 24 * 3600, factor3=True, l3=22.5)

    testing_size = len(test_session)
    # testing_size = 10

    R_10 = 0
    R_20 = 0
    NDCG_20 = 0
    ILD20 = 0

    resultItemDict = {}
    for i in range(testing_size):
        if i % 10000 == 0:
            print("%d/%d" % (i, testing_size))

        score = model.predict(session_id=test_id[i], session_items=test_session[i], session_timestamp=test_timestamp[i],
                            k=20)
       
        items = [x[0] for x in score]

        if len(items)>1:
            ILD20 += getILD(category_id, items[:20], reverse_item)

        for p in items[:20]:
            resultItemDict[p] = resultItemDict.get(p, 0) + 1
        
        printData('newAD_Normal2_predict_'+str(fold)+'_0', test_session[i], test_predict[i], items)

        if test_predict[i] in items:
            rank = items.index(test_predict[i]) + 1
            # print(rank)
            # MRR_20 += 1 / rank
            R_20 += 1
            NDCG_20 += 1 / math.log(rank + 1, 2)
            if rank <= 10:
                # MRR_10 += 1 / rank
                R_10 += 1
                # NDCG_10 += 1 / math.log(rank + 1, 2)

    R_10 = R_10 / testing_size
    R_20 = R_20 / testing_size
    NDCG_20 = NDCG_20 / testing_size
    ILD20 = ILD20 / testing_size

    print("R@10: %f" % R_10)
    print("R@20: %f" % R_20)
    print("NDCG@20: %f" % NDCG_20)
    print("ILD20: %f" % ILD20)

    print("training size: %d" % len(train_session))
    print("testing size: %d" % testing_size)
    print('len of result dict: ', len(resultItemDict))
