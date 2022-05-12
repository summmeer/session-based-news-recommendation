from modules import *

class Model():
    def __init__(self, usernum, itemnum, args, content_emb, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        lstm_layers = 3
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        self.content_emb = tf.Variable(content_emb, dtype=tf.float32, trainable=False)
        seq_content = tf.nn.embedding_lookup(self.content_emb, self.input_seq, max_norm=1)

        print('pos shape', pos.get_shape().as_list())
        print('neg shape', neg.get_shape().as_list())
        print('seq shape', self.input_seq.get_shape().as_list())

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.seq = tf.concat([self.seq, self.seq_content], -1)
            self.seq *= mask

            # Build blocks
        batch_size = tf.shape(self.input_seq)[0]
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(args.hidden_units*2) for i in range(lstm_layers)])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, self.seq, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn, [-1, args.hidden_units*2])
        w1 = tf.Variable(
            tf.random_normal([args.hidden_units*2, args.hidden_units*2], stddev=self.stddev),
            name='w1',
            trainable=True
        )
        b1 = tf.Variable(
            tf.random_normal([args.hidden_units*2], stddev=self.stddev),
            name='b1',
            trainable=True
        )
        pred = tf.matmul(output, w1) + b1
        
        test_item_emb =  tf.concat([item_emb_table, self.content_emb], -1)

        self.pos_logits = tf.matmul(pred, tf.transpose(test_item_emb))

        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0], args.maxlen])
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.pos_logits, labels = self.pos[:,-1])

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        tf.summary.scalar('loss', tf.reduce_sum(self.loss))
        # self.auc = tf.reduce_sum(
        #     ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        # ) / tf.reduce_sum(istarget)

        if reuse is None:
            # tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        # else:
        #     tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, seq):
        return sess.run(self.pos_logits,
                        {self.input_seq: seq, self.is_training: False})
