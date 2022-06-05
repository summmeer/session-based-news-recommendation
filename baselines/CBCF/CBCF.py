import pickle
import pandas as pd
import numpy as np

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

############
#      globo
############
# df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
# category_id = df_data['category_id'].tolist()
# total_cate = len(set(category_id))
# print('total_cate', total_cate, category_id[:10])

# with open('/home/sansa/dataset/globo/articles_embeddings.pickle', 'rb') as f:
#     articles_embeddings = pickle.load(f)

############
#      adressa
############
category_id = pickle.load(open('/home/sansa/dataset/Adressa/articles_category.pkl', 'rb'))
total_cate = len(set(list(category_id.values())))
print('total_cate', total_cate)

# MIND
# category_id_ori = pickle.load(open('/home/sansa/dataset/MIND/articles_category.pkl', 'rb'))
# total_cate = len(set(list(category_id_ori.values())))
# print('total_cate', total_cate)
# category_id = {}
# category2id = {}
# cnt = 0
# for a_id, catename in category_id_ori.items():
#     if catename not in category2id:
#         category2id[catename] = cnt
#         cnt += 1
#     category_id[a_id] = category2id[catename]

with open('../../data/adressa/articles_embeddings_4.pkl', 'rb') as f:
    articles_embeddings = pickle.load(f)

# item_freq_dict_norm = pickle.load(open('/home/sansa/recsys/TCAR/data/adressa/TCAR-mid/Normal/item_freq_dict_norm_1.txt', 'rb'))

def construct_articles_embedding(articles_embeddings, item_dict): 
    articles_vector = np.zeros((len(item_dict)+1, 250+total_cate))
    for article_id, index in item_dict.items():
        articles_vector[index] = np.append(articles_embeddings[article_id], np.eye(total_cate)[category_id[article_id]])
    return articles_vector

def construct_articles_sessions(train_session):
    articles_vector = np.zeros((len(item_dict)+1, len(train_session)), np.int8)
    for i, session in enumerate(train_session):
        for index in session:
            articles_vector[index][i] = 1
    return articles_vector

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

def getUnexp(inSeq, recList):
    score = 0
    n = len(recList)
    cnt = 0
    
    for ini in inSeq:
        if ini == 0:
            continue
        else:
            cnt += 1
        for i in range(0, n):
#             if reverse_item[recList[i]] in category_id and reverse_item[ini] in category_id:
            if category_id[reverse_item[recList[i]]]!=category_id[reverse_item[ini]]:
                score +=1
    if cnt==0 or n==0:
        return 0
    return score/(n*cnt)

# def getESIR(item_freq_dict_norm, recList):
#     score = 0.0
#     n = len(recList)
#     norm = 0.0
#     if len(recList)==0:
#         return 0
#     for i in range(1, n+1):
#         if recList[i-1]+1 in item_freq_dict_norm:
#             score += item_freq_dict_norm[recList[i-1]+1]/np.log2(i+1)
#         else:
#             score += 20/np.log2(i+1)
#         norm += 1.0/np.log2(i+1)
#     return score/norm

folds = [2]

for fold in folds: 
    PATH_TO_TRAIN = '../../data/adressa/CBCF-mid/Normal-2/train_session_'+str(fold)+'.txt'
    PATH_TO_TEST = '../../data/adressa/CBCF-mid/Normal-2/test_session_'+str(fold)+'.txt'
    PATH_TO_ITEM = '../../data/adressa/CBCF-mid/Normal-2/item_dict_'+str(fold)+'.txt'
    # PATH_TO_TRAIN = '../../data/mind/CBCF-mid-1/Normal/train_session_'+str(fold)+'.txt'
    # PATH_TO_TEST = '../../data/mind/CBCF-mid-1/Normal/test_session_'+str(fold)+'.txt'
    # PATH_TO_ITEM = '../../data/mind/CBCF-mid-1/Normal/item_dict_'+str(fold)+'.txt'

    # load data [id,x,t,y]
    train_data = pickle.load(open(PATH_TO_TRAIN, 'rb'))
    train_id = train_data[0]
    print(train_id[-1], '# train session size', len(train_id))
    train_session = train_data[1]
    train_predict = train_data[3]

    for i, s in enumerate(train_session):
        train_session[i] += [train_predict[i]]

    test_data = pickle.load(open(PATH_TO_TEST, 'rb'))
    test_id = test_data[0]
    print(test_id[-1], '# test session size', len(test_id))
    test_session = test_data[1]
    test_predict = test_data[3]

    item_dict = pickle.load(open(PATH_TO_ITEM, 'rb'))
    reverse_item = {}
    for idx, cnt in item_dict.items():
        reverse_item[cnt] = idx

    simivector1 = construct_articles_embedding(articles_embeddings, item_dict)
    simivector2 = construct_articles_sessions(train_session)

    A_sparse1 = sparse.csr_matrix(simivector1)
    A_sparse2 = sparse.csr_matrix(simivector2)
    simi1 = cosine_similarity(A_sparse1)
    simi2 = cosine_similarity(A_sparse2)

    print('construct matrix finished..')


    R_20 = 0
    NDCG_20 = 0
    ILD_20 = 0
    UNEXP = 0

    resultItemDict = set()

    for i, session in enumerate(test_session):
        score = np.zeros(len(item_dict)+1)
        for index in session:
            # print(index)
            score += 0.1*simi1[index] + 0.9*simi2[index]
            # score += simi1[index] # only CB contnet
            # score += simi2[index] # only CF session
        sortlist = list(score.argsort())[::-1]
        ILD_20 += getILD(category_id, sortlist[:20], reverse_item)
        UNEXP += getUnexp(session, sortlist[:20])
        for p in sortlist[:20]:
            resultItemDict.add(p)
        # printData('MI_final_CBCF_Normal_predict_'+str(fold)+'_0', session, test_predict[i], sortlist[:20])
        rank = sortlist.index(test_predict[i]) + 1

        if rank <= 20:
            R_20 += 1
            NDCG_20 += 1 / np.log2(rank + 1)

    testing_size = len(test_session)

    R_20 = R_20 / testing_size
    NDCG_20 = NDCG_20 / testing_size
    ILD_20 = ILD_20 / testing_size
    UNEXP = UNEXP/testing_size

    print("R@20: %f" % R_20)
    print("NDCG@20: %f" % NDCG_20)
    print("ILD@20: %f" % ILD_20)
    print("UNEXP: %f" % UNEXP)
    print ('len of result dict: ', len(resultItemDict))
