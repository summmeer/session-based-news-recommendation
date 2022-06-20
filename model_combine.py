#coding=utf-8
import numpy as np
import tensorflow as tf
import time
import pickle
from util import cau_metrics
from sampler import Sampler
from modules import *

class Seq2SeqAttNN():
    """
    The memory network with context/temproal attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, args):
        '''
        build network
        '''
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.input_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        self.input_publish_month = tf.placeholder(tf.int32, [None, None], name="publish_month")
        self.input_publish_day = tf.placeholder(tf.int32, [None, None], name="publish_day")
        self.input_publish_week = tf.placeholder(tf.int32, [None, None], name="publish_week")
        self.input_publish_hour = tf.placeholder(tf.int32, [None, None], name="publish_hour")
        self.input_publish_minute = tf.placeholder(tf.int32, [None, None], name="publish_minute")

        self.input_click_week = tf.placeholder(tf.int32, [None], name="click_week")
        self.input_click_hour = tf.placeholder(tf.int32, [None], name="click_hour")

        self.label = tf.placeholder(tf.int32, [None], name="lab_input") # index from 0
        self.label_neg = tf.placeholder(tf.int32, [None, None], name="lab_neg")

        self.active_interval = tf.placeholder(tf.int32, [None, None], name="active_time")
        
        self.publish_time_MWDHM = tf.Variable(args['publish_time_MWDHM'], dtype=tf.int32, trainable=False)
        self.itemnum  = args['itemnum']
        self.category_id = args['category_id']
        self.item_freq_dict_norm = args['item_freq_dict_norm']
        self.reverse_item = args['reverse_item']
        self.candidate_n = args['content_emb'].shape[0]
        self.emb_stddev = args['emb_stddev']
        self.stddev = args['stddev']
        self.hidden_size = args['hidden_size']
        self.time_hidden_size = args['time_hidden_size']
        self.l2_emb = args['l2_emb']
        self.batch_size = args['batch_size']
        self.epoch = args['epoch']
        self.neg_num = args['neg_num']

        # INPUT-CONTEXT
        with tf.variable_scope("INPUT-CONTEXT"):
            seq_item, self.item_emb = embedding(self.input_seq, vocab_size=self.candidate_n, num_units=self.hidden_size, zero_pad=True, 
                                                 scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="item_emb", with_t=True)

            t, pos_emb_table = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]), vocab_size=40,
                num_units=self.hidden_size,
                zero_pad=False,
                scale=False,
                l2_reg=self.l2_emb,
                scope="dec_pos",
                with_t=True
            )
            seq_item += t

            self.content_emb = tf.Variable(args['content_emb'], dtype=tf.float32, trainable=False)
            seq_content = tf.nn.embedding_lookup(self.content_emb, self.input_seq, max_norm=1)
            print('size of seq_content', seq_content.get_shape())

        #TEMPROAL-INFO
        with tf.variable_scope("TEMPROAL-INFO"):
            seq_month, self.month_emb = embedding(self.input_publish_month, vocab_size=12+1, num_units=self.time_hidden_size, zero_pad=True, # 12 months, no zero padding
                                          scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="month_embedding", with_t=True)
            seq_day, self.day_emb = embedding(self.input_publish_day, vocab_size=31+1, num_units=self.time_hidden_size, zero_pad=True, # 31 days, no zero padding
                                          scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="day_embedding", with_t=True)
            seq_week, self.week_emb = embedding(self.input_publish_week, vocab_size=7+1, num_units=self.time_hidden_size, zero_pad=True, # 7 weeks, no zero padding
                                          scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="week_embedding", with_t=True)
            seq_hour, self.hour_emb = embedding(self.input_publish_hour, vocab_size=24+1, num_units=self.time_hidden_size, zero_pad=True, # 25 hours, no zero padding
                                          scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="hour_embedding", with_t=True)
            seq_minute, self.minute_emb = embedding(self.input_publish_minute, vocab_size=60+1, num_units=self.time_hidden_size, zero_pad=True, # 60 minutes, no zero padding
                                          scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="minute_embedding", with_t=True)
            
            seq_publish_t = tf.concat([seq_month, seq_day, seq_week, seq_hour, seq_minute], -1)
            print('size of seq_publish_t', seq_publish_t.get_shape())
            candidate_publish_t = tf.concat([
                tf.nn.embedding_lookup(self.month_emb, self.publish_time_MWDHM[:,0], max_norm=1),
                tf.nn.embedding_lookup(self.day_emb, self.publish_time_MWDHM[:,1], max_norm=1),
                tf.nn.embedding_lookup(self.week_emb, self.publish_time_MWDHM[:,2], max_norm=1),
                tf.nn.embedding_lookup(self.hour_emb, self.publish_time_MWDHM[:,3], max_norm=1),
                tf.nn.embedding_lookup(self.minute_emb, self.publish_time_MWDHM[:,4], max_norm=1),
            ], -1)

            click_t = tf.concat([
                tf.nn.embedding_lookup(self.week_emb, self.input_click_week, max_norm=1),
                tf.nn.embedding_lookup(self.hour_emb, self.input_click_hour, max_norm=1)
            ], -1)
            print('size of click_t', click_t.get_shape())


        batch_size = tf.shape(self.input_seq)[0]

        # active_time = tf.expand_dims(self.active_interval, -1) # 1-d vector
        # print('size of active_time', active_time_cate.get_shape())

        seq_active_time, self.duration_emb = embedding(self.active_interval, vocab_size=11, num_units=self.time_hidden_size, zero_pad=False, # 11 cates, no zero padding
                            scale=False, stddev=self.emb_stddev, l2_reg=self.l2_emb, scope="duration_embedding", with_t=True)

        ###### item and cont seq ########

        seq_item_cont = tf.concat([seq_item, seq_content], -1)
        attout_item_cont, alph = multi_attention_layer(seq_item_cont, seq_content, 
                                                       interval=None, 
                                                       click_time=click_t,
                                                       interval=seq_active_time,
                                                       edim1=self.hidden_size*2, edim2=self.hidden_size, edim3=self.time_hidden_size, 
                                                       scope="multi_attention", hidden_size=self.hidden_size, stddev=self.stddev)
        item_cont_edim = self.hidden_size*2
        attout_item_cont = linear_2d(attout_item_cont, item_cont_edim, item_cont_edim, self.stddev, "attout_item_cont_trans")
        print('size of attout_item_cont', attout_item_cont.get_shape())

        ###### publish time seq #########

        attout_publish_t, alph_t = single_attention_layer(seq_publish_t, seq_content, edim1=self.time_hidden_size*5, edim2=self.hidden_size,
                                                          scope="cont_attention", hidden_size=self.hidden_size, stddev=self.stddev)
        time_edim = self.time_hidden_size*5
        attout_publish_t = linear_2d(attout_publish_t, time_edim, time_edim, self.stddev, "attout_pt_trans")
        print('size of attout_publish_t', attout_publish_t.get_shape())

        ###### compute score ########

        attout = tf.concat([attout_item_cont, attout_publish_t], -1) #(batch_size, h+c, 1)
        # print('size of attout', attout.get_shape())

        items_emb_cont = tf.concat([self.item_emb[1:], self.content_emb[1:]], -1)
        items_emb = tf.concat([items_emb_cont, candidate_publish_t], -1)
        
        self.softmax_input = tf.matmul(attout, items_emb, transpose_b=True)
        # self.softmax_input = tf.matmul(attout_item_cont, items_emb_cont, transpose_b=True)
        print('size of softmax_input', self.softmax_input.get_shape())

        neg_logits = tf.reduce_sum(tf.matmul(tf.nn.embedding_lookup(items_emb_cont, self.label_neg), tf.expand_dims(attout_item_cont, -1)), 1)
        neg_feedback = tf.reshape(- tf.log(1 - tf.sigmoid(neg_logits) + 1e-24), [batch_size, 1])
        # neg_feedback = 0
        self.cross_loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.softmax_input, labels=self.label), [batch_size, 1])

        self.loss = self.cross_loss + 0.01*neg_feedback

        ###### optimize ########
    
        params = tf.trainable_variables()
        self.variables_names = [v.name for v in tf.trainable_variables()]
        print(self.variables_names)

        optimizer = tf.train.AdamOptimizer(args['lr'])
        grads_and_vars = optimizer.compute_gradients(self.loss, params)
        if args['max_grad'] != None:
            capped_gvs = [
                (tf.clip_by_norm(gv[0], args['max_grad']), gv[1]) for gv in grads_and_vars
            ]
        else:
            capped_gvs = grads_and_vars
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
    
    def printData(self, filename, batch_in, batch_out, batch_pred):
        file = open('saved/CAR+P_Normal_predict_exa_' + filename + '.txt','a+')
        for index in range(len(batch_in)):
            file.write('# batch in: {} # batch out: {} # batch pred: {} \n'.format(str(batch_in[index]), str(batch_out[index]), str(batch_pred[index])))
        file.close()


    ###### evaluation ########

    def getILD(self, recList):
        score = 0
        n = len(recList)
        for i in range(0, n):
            for j in range(0, n):
                # if self.reverse_item[recList[i]] in self.category_id and self.reverse_item[recList[j]] in self.category_id:  ## this line is needed for Addressa and MIND dataset, but not needed for Globo, or it costs a lot of time, and seems like dead loop
                if j!=i and self.category_id[self.reverse_item[recList[i]]]!=self.category_id[self.reverse_item[recList[j]]]:
                    score += 1
        return score/(n*(n-1))

    def getUnexp(self, inSeq, recList):
        score = 0
        n = len(recList)
        if n==0:
            return 0
        for i in range(0, n):
            for ini in inSeq:
                # if self.reverse_item[recList[i]] in self.category_id and self.reverse_item[ini-1] in self.category_id: ### this line is needed for Addressa and MIND dataset, but not needed for Globo, or it costs a lot of time, and seems like dead loop
                if self.category_id[self.reverse_item[recList[i]]]!=self.category_id[self.reverse_item[ini-1]]:
                    score +=1
        return score/(n*len(inSeq))

    def train(self, sess, item_dict, train_data, neighbor_dict, args, test_data=None, saver=None, threshold_acc=0.99):
        (len_dict_train, session_dict_train, session_time_dict_train) = train_data

        for epoch in range(self.epoch):   # epoch round.
            self.curEpoch = epoch
            print('Epoch {}'.format(epoch))
            batch = 0
            c = []
            sampler = Sampler(len_dict_train, session_dict_train, session_time_dict_train, 
                neighbor_dict, item_dict, args['neg_num'],
                batch_size=self.batch_size
            )
            while sampler.has_next():    # batch round.
                batch += 1
                # print(batch)
                # get this batch data
                batch_in, batch_out, batch_pt, batch_ct, neg, gap = sampler.next_batch()
                # build the feed_dict
                feed_dict = {
                    self.input_seq: batch_in,
                    self.label: batch_out,
                    self.input_publish_month: batch_pt[0],
                    self.input_publish_day: batch_pt[1],
                    self.input_publish_week: batch_pt[2],
                    self.input_publish_hour: batch_pt[3],
                    self.input_publish_minute: batch_pt[4],
                    self.input_click_week: batch_ct[2],
                    self.input_click_hour: batch_ct[3],
                    self.label_neg: neg,
                    self.active_interval: gap,
                    self.is_training: True
                }
                # train
                if batch<3:
                    print(neg[0][:10])
                crt_loss, crt_step, opt = sess.run(
                    [self.loss, self.global_step, self.train_op],
                    feed_dict=feed_dict
                )
                c += list(crt_loss)
            avgc = np.mean(c)
            # self.minuteEmbedding_c = emb_minute
            # self.hourEmbedding_c = emb_hour
            # self.weekEmbedding_c = emb_week
            # self.dayEmbedding_c = emb_day
            # self.monthEmbedding_c = emb_month
            # pickle.dump((emb_day, emb_week, emb_hour, emb_minute), open('saved/'+str(self.foldnum)+'/trained_embe_time_dict_'+str(epoch)+'.pkl', 'wb'))
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('\tloss: {:.6f}'.format(avgc))
            if test_data != None:
                recall = self.test(sess, test_data, args)
                if recall > threshold_acc:
                    modelname = save_model(sess, args, saver)
                    print('Model saved - {}'.format(modelname))

    def test(self, sess, test_data, args):
        print('Measuring...')
        (len_dict_test, session_dict_test, session_time_dict_test) = test_data
        mrr20, recall20, ndcg20 = [], [], []
        ild20 = []
        c_loss =[]
        unexp20 = []
        sampler = Sampler(len_dict_test, session_dict_test, session_time_dict_test, batch_size=self.batch_size)
        batch = 0
        resultItemDict = {}
        while sampler.has_next():    # batch round.
            batch += 1
            # get this batch data
            batch_in, batch_out, batch_pt, batch_ct, neg, gap = sampler.next_batch()
            # build the feed_dict
            feed_dict = {
                self.input_seq: batch_in,
                self.label: batch_out,
                self.input_publish_month: batch_pt[0],
                self.input_publish_day: batch_pt[1],
                self.input_publish_week: batch_pt[2],
                self.input_publish_hour: batch_pt[3],
                self.input_publish_minute: batch_pt[4],
                self.input_click_week: batch_ct[2],
                self.input_click_hour: batch_ct[3],
                self.active_interval: gap,
                self.is_training: False
            }
            # test
            preds, loss = sess.run(
                [self.softmax_input, self.cross_loss],
                feed_dict=feed_dict
            )

            if batch<3:
                print('batch_in:', batch_in[0])
                print('active_interval:', gap[0])
                print('input_click_week:', batch_ct[2][0])
                print('batch_out:', batch_out[0], args['publish_time'][batch_out[0]])
                print('batch pred:', np.argsort(preds[0]).tolist()[::-1][:10])
            if len(preds.shape) == 1:
                preds = np.expand_dims(preds, axis=0)
            t_r, t_m, t_n = cau_metrics(preds, batch_out, 20)
            recall20 += t_r
            mrr20 += t_m
            ndcg20 += t_n
            c_loss += list(loss)
            batch_pred = [np.argsort(pred).tolist()[::-1][:20] for pred in preds]
            for idx, pred in enumerate(batch_pred):
                ild20 += [self.getILD(pred)]
                unexp20 += [self.getUnexp(batch_in[idx], pred)]
                for p in pred:
                    resultItemDict[p] = resultItemDict.get(p, 0) + 1
            if args['is_print']:
                self.printData(str(args['foldnum']) + '_' + str(self.curEpoch), batch_in, batch_out, batch_pred)

        print ('avg loss...', np.mean(c_loss))
        print ('avg ILD...', np.mean(ild20))
        print ('avg unexp...', np.mean(unexp20))
        print ('len of result dict: ', len(resultItemDict))
        print ('MRR@20: {}, Recall@20: {}, nDCG@20: {}'.format(np.mean(mrr20), np.mean(recall20), np.mean(ndcg20)))
        return np.mean(recall20)