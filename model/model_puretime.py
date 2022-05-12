#coding=utf-8
import numpy as np
import tensorflow as tf
import time
import pickle
from layers.NN_adam import NN
from util import cau_recall_mrr_ndcg
from util import pooler
from sampler import Sampler
from FwNn3AttLayer import FwNnAttLayer

class Seq2SeqAttNN(NN):
    """
    The memory network with context attention.
    """
    # ctx_input.shape=[batch_size, mem_size]

    def __init__(self, config):
        super(Seq2SeqAttNN, self).__init__(config)
        self.config = None
        if config != None:
            self.config = config
            # the config.
            self.datas = config['dataset']
            self.nepoch = config['nepoch']  # the train epoches.
            self.batch_size = config['batch_size']  # the max train batch size.
            self.init_lr = config['init_lr']  # the initialize learning rate.
            # the base of the initialization of the parameters.
            self.stddev = config['stddev']
            self.edim = config['edim']  # the dim of the embedding.
            self.max_grad_norm = config['max_grad_norm']   # the L2 norm.
            self.n_items = config["n_items"]
            # the pad id in the embedding dictionary.
            self.pad_idx = config['pad_idx']
            self.neg_sample_num = config['neg_sample_num']

            # the pre-train embedding.
            ## shape = [nwords, edim]
            self.pre_embedding = config['pre_embedding']
            self.content_emb = config['content_embedding']
            self.hour_embedding = config['hour_embedding']
            self.week_embedding = config['week_embedding']
            self.publish_time = config['publish_time']
            self.publish_time_week = config['publish_time_week']
            self.publish_time_hour = config['publish_time_hour']

            # generate the pre_embedding mask.
            self.pre_embedding_mask = np.ones(np.shape(self.pre_embedding))
            self.pre_embedding_mask[self.pad_idx] = 0

            # update the pre-train embedding or not.
            self.emb_up = config['emb_up']

            # the active function.
            self.active = config['active']

            # hidden size
            self.hidden_size = config['hidden_size']

            self.is_print = config['is_print']

            self.cut_off = config["cut_off"]
        self.bpreg = 1.0
        self.is_first = True
        # the input.
        self.inputs = None
        self.aspects = None
        # sequence length
        self.sequence_length = None
        self.reverse_length = None
        self.aspect_length = None
        # the label input. (on-hot, the true label is 1.)
        self.lab_input = None
        self.embe_dict = None  # the embedding dictionary.
        self.article_time = None
        self.hourEmbedding = None
        self.weekEmbedding = None
        self.clickt_week = None
        self.clickt_hour = None
        self.publisht_week = None
        self.publisht_hour = None

        # the optimize set.
        self.global_step = None  # the step counter.
        self.loss = None  # the loss of one batch evaluate.
        self.loss_acc = None
        self.lr = None  # the learning rate.
        self.optimizer = None  # the optimiver.
        self.optimize = None  # the optimize action.
        # the mask of pre_train embedding.
        self.pe_mask = None
        # the predict.
        self.pred = None
        # the params need to be trained.
        self.params = None

    def build_model(self):
        '''
        build the MemNN model
        '''
        
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # the input.
        self.inputs = tf.placeholder(
            tf.int32,
            [None, None],
            name="inputs"
        )

        self.inputs_timecontext1 = tf.placeholder(
            tf.float32,
            [None, None, 1],
            name="absolute_click_hour"
        )

        self.inputs_timecontext2 = tf.placeholder(
            tf.float32,
            [None, None, 1],
            name="absolute_publish_hour"
        )

        self.inputs_timecontext3 = tf.placeholder(
            tf.float32,
            [None, None, 1],
            name="delta_hour"
        )

        # self.item_weight = tf.placeholder(
        #     tf.float32,
        #     [self.n_items],
        #     name="item_weight"
        # )

        self.last_inputs = tf.placeholder(
            tf.int32,
            [None],
            name="last_inputs"
        )

        batch_size = tf.shape(self.inputs)[0]

        self.sequence_length = tf.placeholder(
            tf.int64,
            [None],
            name='sequence_length'
        )

        self.lab_input = tf.placeholder(
            tf.int32,
            [None],
            name="lab_input"
        )

        # self.lab_neg = tf.placeholder(
        #     tf.int32,
        #     [None, self.neg_sample_num],
        #     name="lab_neg"
        # )

        # the lookup dict.
        self.embe_dict = tf.Variable(
            self.content_emb/20,
            dtype=tf.float32,
            trainable=self.emb_up
        )
        print('size of embe_dict', self.embe_dict.get_shape()) 
        ############!!!!!# input is start from 1, label is start from 0
        # self.embe_dict = tf.get_variable("embe_dict", shape=[17001, 100], initializer=tf.contrib.layers.xavier_initializer())
    
        # content emb look up dict.
        self.content_dict = tf.Variable(
            self.content_emb/20,
            dtype=tf.float32,
            trainable=False
        )
        print('size of content_dict', self.content_dict.get_shape())
        
        self.article_time = tf.Variable(
            self.publish_time,
            dtype=tf.float32,
            trainable=False
        )

        self.pe_mask = tf.Variable(
            self.pre_embedding_mask,
            dtype=tf.float32,
            trainable=False
        )
        self.embe_dict *= self.pe_mask

        sent_bitmap = tf.ones_like(tf.cast(self.inputs, tf.float32))

        inputs = tf.nn.embedding_lookup(self.embe_dict, self.inputs, max_norm=1)
        inputs_content_emb = tf.nn.embedding_lookup(self.content_dict, self.inputs, max_norm=1)
        lastinputs = tf.nn.embedding_lookup(self.embe_dict, self.last_inputs, max_norm=1)
        # lab_time = tf.nn.embedding_lookup(self.article_time, self.lab_input, max_norm=1)
        # inputs_item_weight = tf.nn.embedding_lookup(self.item_weight, self.inputs, max_norm=1)
        # inputs_item_weight = tf.expand_dims(inputs_item_weight, -1)

        # time_context = tf.concat([self.inputs_timecontext1, self.inputs_timecontext2, self.inputs_timecontext3, inputs_item_weight], -1)
        # print('size of time_context', time_context.get_shape())

        # org_memory = (inputs + inputs_content_emb)*0.5
        # org_memory = inputs
        org_memory = tf.concat([inputs, inputs_content_emb], -1)

        pool_out = pooler(
            org_memory,
            'mean',
            axis=1,
            sequence_length = tf.cast(tf.reshape(self.sequence_length, [batch_size, 1]), tf.float32)
        )
        pool_out = tf.reshape(pool_out, [-1, self.edim*2])

        attlayer = FwNnAttLayer(
            self.edim,
            active=self.active,
            stddev=self.stddev,
            norm_type='none'
            # flag = True
        )
        attout, alph = attlayer.forward4(org_memory, lastinputs, inputs_content_emb, sent_bitmap)
        attout = tf.reshape(attout, [-1, 500]) + pool_out

        # attlayer_time = FwNnAttLayer(
        #     1,
        #     active=self.active,
        #     stddev=self.stddev,
        #     norm_type='none'
        #     # flag = True
        # )
        # attout_time, alph_t = attlayer_time.forward2(self.inputs_timecontext1, self.inputs_timecontext2, self.inputs_timecontext3, sent_bitmap)
        # attout_time = tf.reshape(attout_time, [-1, 1])

        self.alph = tf.reshape(alph, [batch_size, 1, -1])

        self.w1 = tf.Variable(
            tf.random_normal([self.edim*2, 500], stddev=self.stddev),
            trainable=True
        )

        self.w2 = tf.Variable(
            tf.random_normal([self.edim, 500], stddev=self.stddev),
            trainable=True
        )
        attout = tf.tanh(tf.matmul(attout, self.w1))
        # attout = tf.layers.batch_normalization(attout, training=self.is_training)
        # attout = tf.nn.dropout(attout, 0.7)
        lastinputs = tf.tanh(tf.matmul(lastinputs, self.w2))
        # lastinputs= tf.layers.batch_normalization(lastinputs, training=self.is_training)
        prod = attout * lastinputs
        # item_weight_expand = tf.expand_dims(self.item_weight, -1)
        # items_emb = tf.concat([self.embe_dict[1:], self.article_time, item_weight_expand], -1)
        items_emb = tf.concat([self.embe_dict[1:], self.content_dict[1:]], -1)

        # sco_mat = tf.matmul(prod, self.embe_dict[1:], transpose_b=True)
        self.before_softmax = prod
        sco_mat = tf.matmul(prod, items_emb, transpose_b=True)
        print('size of sco_mat', sco_mat.get_shape())
        # sco_mat= tf.layers.batch_normalization(sco_mat, training=self.is_training)
        self.softmax_input = sco_mat
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sco_mat, labels = self.lab_input)
        
        # softmax_output = tf.nn.softmax(sco_mat)
        # softmax_input_neg = tf.batch_gather(sco_mat, self.lab_neg)
        # print('size of softmax_input_neg', softmax_input_neg.get_shape())
        # softmax_output_neg = tf.batch_gather(softmax_output, self.lab_neg)
        # lab_pos = tf.reshape(
        #     tf.tile(self.lab_input, [self.neg_sample_num]),
        #     [batch_size, self.neg_sample_num]
        # )
        # print('size of lab_pos', lab_pos.get_shape())
        # softmax_input_pos = tf.batch_gather(sco_mat, lab_pos)
        # print('size of softmax_input_pos', softmax_input_pos.get_shape())
        # loss_bpr = -1 * tf.log(tf.reduce_sum(softmax_output_neg * tf.sigmoid(softmax_input_pos-softmax_input_neg), axis=-1) + 1e-24) + self.bpreg*tf.reduce_sum(tf.square(softmax_input_neg)*softmax_output_neg, axis=-1)
        # # loss_bpr = tf.reduce_sum(tf.log(tf.sigmoid(softmax_output_neg) + 1e-24), axis=-1)
        # print('size of loss_bpr', loss_bpr.get_shape())
        print('size of loss', self.loss.get_shape())
        # self.loss_acc = self.loss + 0.0*tf.losses.mean_squared_error(attout_time, lab_time)
        self.loss_acc = self.loss
        # article_time = tf.reshape(
        #     tf.tile(self.article_time, [batch_size, 1]),
        #     [batch_size, tf.shape(self.article_time)[0]]
        # )
        # print('size of article_time', article_time.get_shape())
        # pred_time = tf.exp(-1000 * (tf.abs(attout_time - article_time)))
        # pred_time = tf.reshape(pred_time, [batch_size, tf.shape(self.article_time)[0]])
        # self.pred_test = sco_mat + 4 * pred_time
        # print('size of pred_test', self.pred_test.get_shape())
        # the optimize.
        self.params = tf.trainable_variables()
        self.optimize = super(Seq2SeqAttNN, self).optimize_normal(
            self.loss_acc, self.params)
    
    def printData(self, filename, batch_in, batch_out, batch_pred, batch_timestamp):
        file = open('saved/' + filename + '.txt','a+')
        for index in range(len(batch_in)):
            file.write('# batch in: {} # batch out: {} # batch pred: {} # batch time: {} \n'.format(str(batch_in[index]), str(batch_out[index]), str(batch_pred[index])+str([ self.publish_time[p] for p in batch_pred[index]]), str(batch_timestamp[index])))
        file.close()

    def train(self, sess, train_data, item_dict, neighbor_dict, test_data=None, saver=None, threshold_acc=0.99):
        train_data = list(train_data)
        print('build train len dict..')
        len_dict_train = {}
        session_dict_train = {}
        session_time_dict_train = {}
        session_len_dict = {}
        # cnt = 0
        item_freq = np.ones(len(item_dict))
        # for (session_id, train_session, train_timestamp, train_predict) in train_data:
        #     len_dict_train[len(train_session)] = len_dict_train.get(len(train_session), []) + [str(session_id)+'_'+str(len(train_session))]
        #     session_len_dict[session_id] = len(train_session)
        #     session_dict_train[str(session_id)+'_'+str(len(train_session))] = train_session + [train_predict]
        #     item_freq[train_predict-1]+=1
        #     session_time_dict_train[str(session_id)+'_'+str(len(train_session))] = train_timestamp
        #     # cnt += 1
        #     # if cnt>1200:
        #     #     break
        # pickle.dump(len_dict_train, open('saved/len_dict_train.pkl', 'wb'))
        # pickle.dump(session_len_dict, open('saved/session_len_dict.pkl', 'wb'))
        # pickle.dump(session_dict_train, open('saved/session_dict_train', 'wb'))
        # pickle.dump(item_freq, open('saved/item_freq', 'wb'))
        # pickle.dump(session_time_dict_train, open('saved/session_time_dict_train', 'wb'))

        len_dict_train = pickle.load(open('saved/len_dict_train.pkl', 'rb'))
        session_len_dict = pickle.load(open('saved/session_len_dict.pkl', 'rb'))
        session_dict_train = pickle.load(open('saved/session_dict_train', 'rb'))
        item_freq = pickle.load(open('saved/item_freq', 'rb'))
        session_time_dict_train = pickle.load(open('saved/session_time_dict_train', 'rb'))

        # print(session_dict_train)
        item_freq = 1.0/item_freq
        self.item_freq = item_freq
        print(item_freq[:100])
        if test_data:
            test_data = list(test_data)
            print('build test len dict..')
            len_dict_test = {}
            session_dict_test = {}
            session_time_dict_test = {}
            # for (session_id, test_session, test_timestamp, test_predict) in test_data:
            #     len_dict_test[len(test_session)] = len_dict_test.get(len(test_session), []) + [session_id]
            #     session_dict_test[session_id] = test_session + [test_predict]
            #     session_time_dict_test[session_id] = test_timestamp
            #     # cnt += 1
            #     # if cnt>1600:
            #     #     break
            # pickle.dump(len_dict_test, open('saved/len_dict_test.pkl', 'wb'))
            # pickle.dump(session_dict_test, open('saved/session_dict_test.pkl', 'wb'))
            # pickle.dump(session_time_dict_test, open('saved/session_time_dict_test', 'wb'))

            len_dict_test = pickle.load(open('saved/len_dict_test.pkl', 'rb'))
            session_dict_test = pickle.load(open('saved/session_dict_test.pkl', 'rb'))
            session_time_dict_test = pickle.load(open('saved/session_time_dict_test', 'rb'))

        candidate_sortedByTime = []
        candidate_sortedByTime.append(list(np.argsort(self.publish_time[:,0])))
        candidate_sortedByTime.append(list(np.sort(self.publish_time[:,0])))
        
        for epoch in range(self.nepoch):   # epoch round.
            self.curEpoch = epoch
            print('Epoch {}'.format(epoch))
            batch = 0
            c = []
            sampler = Sampler(
                len_dict_train, session_dict_train, item_dict, neighbor_dict, session_len_dict, neg_sample_num=None, session_time_dict=session_time_dict_train, candidate_sortedByTime=candidate_sortedByTime, batch_size=self.batch_size
            )
            while sampler.has_next():    # batch round.
                batch += 1
                # print(batch)
                # get this batch data
                batch_in, batch_out, batch_last, batch_seq_l, batch_neg, batch_time = sampler.next_batch()
                # build the feed_dict
                feed_dict = {
                    self.inputs: batch_in,
                    self.last_inputs: batch_last,
                    self.lab_input: batch_out,
                    self.sequence_length: batch_seq_l,
                    # self.item_weight: item_freq,
                    # self.lab_neg: batch_neg,
                    # self.clickt_week: batch_time[1],
                    # self.clickt_hour: batch_time[2],
                    # self.publisht_week: batch_time[3],
                    # self.publisht_hour: batch_time[4],
                    # self.inputs_timecontext1: batch_time[5],
                    # self.inputs_timecontext2: batch_time[6],
                    # self.inputs_timecontext3: batch_time[7],
                    self.is_training: True
                }
                # train
                crt_loss, crt_step, opt, embe_dict = sess.run(
                    [self.loss_acc, self.global_step, self.optimize, self.embe_dict],
                    feed_dict=feed_dict
                )
                c += list(crt_loss)
            avgc = np.mean(c)
            pickle.dump(embe_dict, open('saved/trained_embe_dict_'+str(epoch)+'.pkl', 'wb'))
            embe_dict[12566:] = self.content_emb[12566:]/20
            if np.isnan(avgc):
                print('Epoch {}: NaN error!'.format(str(epoch)))
                self.error_during_train = True
                return
            print('\tloss: {:.6f}'.format(avgc))
            print('batch num, ', batch, ', batch size: ', self.batch_size)
            if test_data != None:
                recall = self.test(sess, len_dict_test, session_dict_test, session_time_dict_test, embe_dict)
                if recall > threshold_acc:
                    self.save_model(sess, self.config, saver)
                    print('Model saved - {}'.format(self.config['saved_model']))
        if self.is_print:
            pass

    def test(self, sess, len_dict_test, session_dict_test, session_time_dict_test, embe_dict):

        # calculate the acc
        print('Measuring...')
        mrr20, recall20, ndcg20 = [], [], []
        mrr10, recall10, ndcg10 = [], [], []
        c_loss =[]
        sampler = Sampler(
            len_dict_test, session_dict_test, 
            session_time_dict=session_time_dict_test,
            batch_size=self.config['batch_size']
        )
        batch = 0
        resultItemDict = {}
        while sampler.has_next():    # batch round.
            batch += 1
            # get this batch data
            batch_in, batch_out, batch_last, batch_seq_l, _, batch_time = sampler.next_batch()
            # build the feed_dict
            feed_dict = {
                self.inputs: batch_in,
                self.last_inputs: batch_last,
                self.lab_input: batch_out,
                self.sequence_length: batch_seq_l,
                self.embe_dict: embe_dict,
                # self.item_weight: self.item_freq,
                # self.clickt_week: batch_time[1],
                # self.clickt_hour: batch_time[2],
                # self.publisht_week: batch_time[3],
                # self.publisht_hour: batch_time[4],
                # self.inputs_timecontext1: batch_time[5],
                # self.inputs_timecontext2: batch_time[6],
                # self.inputs_timecontext3: batch_time[7],
                self.is_training: False
            }
            # test
            preds_old, loss, alpha, before_softmax = sess.run(
                [self.softmax_input, self.loss_acc, self.alph, self.before_softmax],
                feed_dict=feed_dict
            )
            if batch<2:
                pickle.dump(before_softmax, open('saved/before_softmax_'+str(self.curEpoch)+str(batch)+'.pkl', 'wb'))
            # index = 0
            preds = preds_old
            # for pred in preds_old:
            #     pred[np.where((self.publish_time<batch_time[5][index][-1][0]-24/10000))[0]] /= 10.0 #publish
            #     pred[np.where((self.publish_time>batch_time[5][index][-1][0]+1/10000))[0]] = -10.0 #click
            #     preds[index] = pred
            #     index += 1
            t_r, t_m, t_n = cau_recall_mrr_ndcg(preds, batch_out, cutoff=self.cut_off)
            recall20 += t_r
            mrr20 += t_m
            ndcg20 += t_n
            t_r, t_m, t_n = cau_recall_mrr_ndcg(preds, batch_out, cutoff=self.cut_off/2)
            recall10 += t_r
            mrr10 += t_m
            ndcg10 += t_n
            c_loss += list(loss)
            batch_pred = [np.argsort(pred).tolist()[::-1][:20] for pred in preds]
            for pred in batch_pred:
                for p in pred:
                    resultItemDict[p] = resultItemDict.get(p, 0) + 1
            if self.is_print:
                self.printData('test_batch_'+str(self.curEpoch), batch_in, batch_out, batch_pred, batch_time[6])

        print ('avg loss...', np.mean(c_loss))
        print ('len of result dict: ', len(resultItemDict))
        print ('MRR@20: {}, Recall@20: {}, nDCG@20: {}'.format(np.mean(mrr20), np.mean(recall20), np.mean(ndcg20)))  
        print ('MRR@10: {}, Recall@10: {}, nDCG@10: {}'.format(np.mean(mrr10), np.mean(recall10), np.mean(ndcg10)))
        return np.mean(recall20)
