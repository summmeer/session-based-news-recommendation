import pickle
import operator
import time
import pandas as pd
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import argparse
import json

def get_datetime(click_t, publish_t, active_time):
    t1 = datetime.fromtimestamp(click_t)
    t2 = datetime.fromtimestamp(publish_t)
    delta_hour = (t1-t2).days * 24 + (t1-t2).seconds//3600 # int hour
    if active_time == 0:
        active_time = 1 # at least 1s 
    return {'click_t': t1, 'publish_t': t2, 'delta_h': delta_hour, 'active_t': active_time}

def ISOTimeMap(isotime):
    return [isotime.month, isotime.day, isotime.isoweekday(), isotime.hour+1, isotime.minute+1]

def isSession(timestamp1, timestamp2): 
    '''
    t1 happens after t2
    30 minutes as a session
    '''
    t1 = datetime.fromtimestamp(timestamp1)
    t2 = datetime.fromtimestamp(timestamp2)
    delta_sec = (t1-t2).days * 24 * 3600 + (t1-t2).seconds
    return (delta_sec<30*60)

def preProcess(data_path, use_preprocess, articles, filter_len):
    if use_preprocess:
        print('Loading preprocessed middle file...')
        sess_clicks = pickle.load(open('../data/adressa/sess_clicks.mid', 'rb'))
        sess_date_sorted = pickle.load(open('../data/adressa/sess_date_sorted.mid', 'rb'))
    else:
        sess_clicks = {} # {key: sess_id, value: list of click items}
        sess_date = {} # {key: sess_id, value: timestamp of session}
        sess_userid = {} # {key: sess_id, value: user_id} # the same user may appear in different session, but in session-based task we regard him as different anonymous user

        for line in open(data_path + 'eventUser.pkl'):
            ori_session_data = json.loads(line)
        print('Reloading origin {} users...'.format(len(ori_session_data)))

        sess_id = 1
        countMiss = 0 # number of sessions need to delete
        MissSessionID = set() # bad session id set
        countLong = 0 # number of long sessions need to cut
        start_timestamp = 0

        size_cnt = 1 # count flag to determine whether the session is end
        for user_id, sessions in ori_session_data.items():
            for session in sessions:
                article_click_time = session['utime']
                click_article_id = session['url']
                article_publish_time = articles[click_article_id]['utime']
                time_context = get_datetime(article_click_time, article_publish_time, session['activeTime'])

                if size_cnt>1 and not isSession(article_click_time, start_timestamp): 
                    # this click is within current session
                    # if a session last over 30 min, count it as a new session from start
                    size_cnt = 1
                    sess_id += 1
                    countLong += 1

                if time_context['delta_h']<0: # click before publish, bad case need to delete
                    countMiss += 1
                    MissSessionID.add(sess_id)

                item = (click_article_id, time_context)

                if size_cnt > 1: # if sess_id in sess_clicks: # not the start of session
                    if click_article_id == sess_clicks[sess_id][-1][0]:
                        sess_clicks[sess_id][-1][1]['active_t'] += time_context['active_t'] # merge the successive items
                        size_cnt -= 1
                    else:
                        sess_clicks[sess_id] += [item]
                else: # start of session
                    sess_userid[sess_id] = user_id
                    sess_clicks[sess_id] = [item]
                    sess_date[sess_id] = article_click_time  # timestamp first item timestamp as session timestamp
                    start_timestamp = article_click_time
                
                size_cnt += 1
            # next one is a new session
            size_cnt = 1
            sess_id += 1

        print('Length of session clicks: {}, number of bad case {}, number of cut case {}.'.format(len(sess_clicks), countMiss, countLong))
        # print example
        for sess_id, sess in sess_clicks.items():
            seq = [arti for (arti, t) in sess]
            if len(seq) != len(set(seq)):
                countMiss += 1
                MissSessionID.add(sess_id)

        print('Length of session clicks: {}, number of bad case {}, number of cut case {}.'.format(len(sess_clicks), countMiss, countLong))
        # filter out length 1 sessions
        for s in list(sess_clicks):
            if len(sess_clicks[s]) < filter_len:
                del sess_clicks[s]
                del sess_date[s]
            if s in MissSessionID and s in sess_clicks:
                del sess_clicks[s]
                del sess_date[s]
        print('After filter out sessions whose length is less than {} and remove bad case, length of session clicks: {}.'.format(filter_len, len(sess_clicks)))

        # split out train set and test set based on date
        sess_date_sorted = sorted(sess_date.items(), key=operator.itemgetter(1))
        print(sess_date_sorted[0])
        print(sess_date_sorted[-1])

        print('Saving middle file...')
        pickle.dump(sess_clicks, open('../data/adressa/sess_clicks.mid', 'wb'))
        pickle.dump(sess_date_sorted, open('../data/adressa/sess_date_sorted.mid', 'wb'))
        pickle.dump(sess_userid, open('../data/adressa/sess_userid.mid', 'wb'))

    return sess_clicks, sess_date_sorted

def get_statistic(sess_clicks, sess_date_sorted, ori_articles, start_time, end_time):
    articles = {} # {key: article_id, value: list of date}
    seq_len_list = []
    delta_time_list = [] # record time gap between click and publish time

    for (sid, date) in sess_date_sorted:
        seq = sess_clicks[sid]
        seq_len_list.append(len(seq))
        for (article_id, date) in seq:
            delta_time_list.append(date['delta_h'])
            if article_id in articles:
                articles[article_id] += [date['click_t']]
            else:
                articles[article_id] = [date['click_t']]
    print('articles', len(articles))
    # for ori_id, article_t in ori_articles.items():
    #     article_t = article_t['utime']
    #     if start_time<article_t and article_t<end_time and ori_id not in articles: # add them as candidate
    #         articles[ori_id] = []
    pickle.dump(list(articles.keys()), open('../data/adressa/articles_list.txt', 'wb'))
    print('Total sessions: {}, avg {:.5} clicks per session.'.format(len(sess_clicks), sum(seq_len_list)/len(seq_len_list)))
    print('Total articles: {}, avg {:.5} clicks per article.'.format(len(articles), len(delta_time_list)/len(articles)))


def get_seq(sess_clicks, session_ids, item_dict, item_cnt, ignore_cold=False, padding=0):
    sess_ids = []
    sess_seqs = []
    sess_dates = []
    for sid in session_ids:
        seq = sess_clicks[sid]
        idseq = []
        timeseq = []
        for (item_id, date) in seq:
            timeseq += [date]
            if item_id in item_dict:
                idseq += [item_dict[item_id]]
            else:
                if ignore_cold:
                    idseq += [padding]
                else:
                    item_dict[item_id] = item_cnt
                    idseq += [item_cnt]
                    item_cnt += 1
        sess_ids += [sid]
        sess_dates += [timeseq]
        sess_seqs += [idseq]
    # print('...item_num: %d' % (item_cnt - 1))
    return sess_ids, sess_dates, sess_seqs, item_dict, item_cnt

# split sequence
def split_seq(sess_ids, sess_dates, sess_seqs, filter_len, augment=False):
    s_in = []
    s_t = []
    s_out = []
    s_id = []
    for sid, date, seq in zip(sess_ids, sess_dates, sess_seqs):
        if augment:
            tmpLen = len(seq) - filter_len + 2
        else:
            tmpLen = 2
        for i in range(1, tmpLen):
            s_out += [seq[-i]]
            s_in += [seq[:-i]]
            if i == 1:
                s_t += [date]
            else:
                s_t += [date[:-i+1]]
            s_id += [sid]
    return s_in, s_out, s_t, s_id

def main(args, articles):
    sess_clicks, sess_date_sorted = preProcess(args.data_path, args.use_preprocess, articles, args.filter_len)
    print(len(sess_date_sorted))

    start_time = 1483225207
    one_day = 86400
    end_time = 1485039601

    if args.show_statistic:
        get_statistic(sess_clicks, sess_date_sorted, articles, start_time, end_time)

    timelist = []
    for i in range(21):
        timelist.append(start_time + i * one_day)
    timelist.append(end_time)

    if args.split_way == 'Normal-2':
        folds = [1,2,3,4,5,6,7]
    else:
        folds = [1]

    for fold in folds:
        train_session_ids = []
        test_session_ids = []
        
        if args.split_way == 'Normal-2':
            start_train = 3*(fold-1)
            end_train = 3*fold-1
            start_test = end_train
            end_test = 3*fold
        elif args.split_way == 'TrainLen-7':
            start_train = fold-1
            end_train = 7
            start_test = end_train
            end_test = 8
        elif args.split_way == 'TestLen':
            start_train = 8
            end_train = 13
            start_test = 13 + fold - 1
            end_test = 14 + fold - 1
        for (sid, date) in sess_date_sorted:
            if (timelist[start_train]<date and date<timelist[end_train]):
                train_session_ids.append(sid)
            elif (timelist[start_test]<date and date<timelist[end_test]):
                test_session_ids.append(sid)
            elif (date>timelist[end_test]):
                break

        print('Split way: {}, foldnum: {}, len of train_session: {}, len of test_session: {}.'.format(args.split_way, fold, len(train_session_ids), len(test_session_ids)))

        item_dict = {}
        item_cnt = 1
        # convert training sessions to sequences and renumber the items
        tra_sess_ids, tra_sess_dates, tra_sess_seqs, item_dict, item_cnt = get_seq(sess_clicks, train_session_ids, item_dict, item_cnt)
        print('...training dictlen:', item_cnt-1)
        tes_sess_ids, tes_sess_dates, tes_sess_seqs, item_dict_all, item_cnt_all = get_seq(sess_clicks, test_session_ids, item_dict, item_cnt, ignore_cold=not args.cold_start)
        print('...testing dictlen:', item_cnt_all-1)
        print("...training sess len: %d" % len(tra_sess_ids))
        print("...testing sess len: %d" % len(tes_sess_ids))

        tra_in, tra_out, tra_t, tra_id = split_seq(tra_sess_ids, tra_sess_dates, tra_sess_seqs, args.filter_len, augment=args.train_augment)
        tes_in, tes_out, tes_t, tes_id = split_seq(tes_sess_ids, tes_sess_dates, tes_sess_seqs, args.filter_len)
        print('After split, trainig sequences: {}, testing sequences: {}'.format(len(tra_in), len(tes_in)))
        print('...examples:', tes_in[0], tes_t[0])

        train_data = (tra_id, tra_in, tra_t, tra_out)
        test_data = (tes_id, tes_in, tes_t, tes_out)

        savefile = ''
        if not args.cold_start:
            if args.train_augment:
                savefile = 'SRGNN-mid'
            else:
                savefile = 'GRU-STAMP-mid'
        else:
            if args.train_augment:
                savefile = 'TCAR-mid'
            else:
                savefile = 'CBCF-mid'
        if args.split_way == 'TestLen':
            if fold == 1:
                pickle.dump(item_dict, open('../data/adressa/' + savefile + '/TestLen/item_dict.txt', 'wb'))
                pickle.dump(train_data, open('../data/adressa/' + savefile + '/TestLen/train_session.txt', 'wb'))
            pickle.dump(test_data, open('../data/adressa/' + savefile + '/TestLen/test_session_' + str(fold) + '.txt', 'wb'))
        else:
            pickle.dump(item_dict, open('../data/adressa/' + savefile + '/' + args.split_way + '/item_dict_' + str(fold) + '.txt', 'wb'))
            pickle.dump(train_data, open('../data/adressa/' + savefile + '/' + args.split_way + '/train_session_' + str(fold) + '.txt', 'wb'))
            pickle.dump(test_data, open('../data/adressa/' + savefile + '/' + args.split_way + '/test_session_' + str(fold) + '.txt', 'wb'))
        
        if args.content_info: # add context info
            articles_embeddings = pickle.load(open('../data/adressa/articles_embeddings_4.pkl', 'rb'))
            print('Load article embedding...', len(articles_embeddings))
            # print(list(articles_embeddings.items())[0])
            emb_dim = 250
            idx_list = []

            for ori_id, cur_id in item_dict_all.items():
                idx_list.append(ori_id)
            
            # for ori_id, article_t in articles.items():
            #     article_t = article_t['utime']
            #     if timelist[start_test]<article_t and article_t<timelist[end_test] and ori_id not in item_dict_all: # add them as candidate
            #         idx_list.append(ori_id)

            content_weight = np.expand_dims(np.array([0.0]*emb_dim), 0)
            for ori_id in idx_list:
                content_weight = np.append(content_weight, [articles_embeddings[ori_id]], axis=0)
            print('Create article embedding...', content_weight.shape)
            pickle.dump(content_weight, open('../data/adressa/' + savefile + '/' + args.split_way + '/content_weight_' + str(fold) + '.txt', 'wb'))

            # generate candidate mask (0: not candidate, 1: is candidate)
            # candidate_masks = {}
            # for s_id, s_t in zip(tes_id, tes_t):
            #     candidate_mask = np.array([1]*len(idx_list))
            #     idx = item_cnt-1
            #     for ori_id in idx_list[item_cnt-1:]:
            #         article_t = articles[ori_id]
            #         if article_t<timelist[start_test] or datetime.fromtimestamp(article_t)>s_t[-1]['click_t']:
            #             candidate_mask[idx] = 0
            #         idx += 1
            #     candidate_masks[s_id] = candidate_mask
            # print('...example', np.sum(candidate_masks[s_id]))
            # pickle.dump(candidate_masks, open('../data/adressa/' + savefile + '/' + args.split_way + '/candidate_masks_' + str(fold) + '.txt', 'wb'))

            publishTime_MDWHM = []
            timeList = []
            for ori_id in idx_list:
                article_t = articles[ori_id]['utime']
                t = datetime.fromtimestamp(article_t)
                timeList.append(t)
                publishTime_MDWHM.append(ISOTimeMap(t))
            publishTime_MDWHM = np.array(publishTime_MDWHM)
            pickle.dump((timeList, publishTime_MDWHM), open('../data/adressa/' + savefile + '/' + args.split_way + '/publish_time_' + str(fold) + '.txt', 'wb'))

            if args.build_len_dict:
                print('build len dict')
                len_dict_train = {}
                session_dict_train = {}
                session_time_dict_train = {}

                for (tra_id, tra_in, tra_t, tra_out) in zip(train_data[0], train_data[1], train_data[2], train_data[3]):
                    len_dict_train[len(tra_in)] = len_dict_train.get(len(tra_in), []) + [str(tra_id)+'_'+str(len(tra_in))]
                    session_dict_train[str(tra_id)+'_'+str(len(tra_in))] = tra_in + [tra_out]
                    session_time_dict_train[str(tra_id)+'_'+str(len(tra_in))] = tra_t
                pickle.dump(len_dict_train, open('../data/adressa/' + savefile + '/' + args.split_way + '/len_dict_train' + str(fold) + '.pkl', 'wb'))
                pickle.dump(session_dict_train, open('../data/adressa/' + savefile + '/' + args.split_way + '/session_dict_train_' + str(fold) + '.pkl', 'wb'))
                pickle.dump(session_time_dict_train, open('../data/adressa/' + savefile + '/' + args.split_way + '/session_time_dict_train' + str(fold) + '.pkl', 'wb'))

                len_dict_test = {}
                session_dict_test = {}
                session_time_dict_test = {}
                for (tes_id, tes_in, tes_t, tes_out) in zip(test_data[0], test_data[1], test_data[2], test_data[3]):
                    len_dict_test[len(tes_in)] = len_dict_test.get(len(tes_in), []) + [tes_id]
                    session_dict_test[tes_id] = tes_in + [tes_out]
                    session_time_dict_test[tes_id] = tes_t
                pickle.dump(len_dict_test, open('../data/adressa/' + savefile + '/' + args.split_way + '/len_dict_test' + str(fold) + '.pkl', 'wb'))
                pickle.dump(session_dict_test, open('../data/adressa/' + savefile + '/' + args.split_way + '/session_dict_test_' + str(fold) + '.pkl', 'wb'))
                pickle.dump(session_time_dict_test, open('../data/adressa/' + savefile + '/' + args.split_way + '/session_time_dict_test' + str(fold) + '.pkl', 'wb'))
                
    print("Finish data preprocess...")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/sansa/dataset/Adressa/AdressaDataLSTMRecommender/', type=str, help='Location of origin dataset')
    parser.add_argument('--use_preprocess', action="store_false", help='Use preprocessed middle file or not')
    parser.add_argument('--show_statistic', action="store_true", help='Print data statistic or not')
    parser.add_argument('--filter_len', default=2, type=int, help='The shortest session length in dataset, at least 2')
    parser.add_argument('--split_way', default='Normal-2', type=str, choices=['Normal-2', 'TrainLen-7', 'TestLen'], help='Choose different split ways')
    # details
    parser.add_argument('--cold_start', action="store_false", help='Count for cold-start item or not')
    parser.add_argument('--content_info', action="store_true", help='Generate other information/cold-start item or not')
    parser.add_argument('--train_augment', action="store_false", help='Do training data augmentation or not')
    parser.add_argument('--build_len_dict', action="store_false", help='Build training data length dictionary for batch training (for various length model)')

    args = parser.parse_args()
    print(args)

    for line in open(args.data_path + 'articles_unfiltered_withword.json'):
        articles = json.loads(line)

    if not args.use_preprocess:
        sess_clicks, sess_date_sorted = preProcess(args.data_path, args.use_preprocess, articles, args.filter_len)
        if args.show_statistic:
            start_time = 1483225207
            end_time = 1485039601
            get_statistic(sess_clicks, sess_date_sorted, articles, start_time, end_time)
    else:
        main(args, articles)