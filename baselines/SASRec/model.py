from modules import *


class Model():
    def __init__(self, usernum, itemnum, args, content_emb, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)

        self.content_emb = tf.Variable(content_emb, dtype=tf.float32, trainable=False)
        self.seq_content = tf.nn.embedding_lookup(self.content_emb, self.input_seq, max_norm=1)

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

            # Positional Encoding
            t, pos_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))

            # self.seq = tf.concat([self.seq, self.seq_content], -1)
            self.seq *= mask

            print('seq shape', self.seq.get_shape().as_list())

            # Build blocks

            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):

                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   num_units=args.hidden_units,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   is_training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                           dropout_rate=args.dropout_rate, is_training=self.is_training)
                    self.seq *= mask

            self.seq = normalize(self.seq)
        print('seq shape', self.seq.get_shape().as_list())
        # pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
        # neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
        # pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
        # neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

        # print('pos emb', pos_emb.get_shape().as_list())
        # print('neg emb', neg_emb.get_shape().as_list())
        # print('seq emb', seq_emb.get_shape().as_list())
        test_item_emb =  tf.concat([item_emb_table, self.content_emb], -1)
        self.item_emb_table = item_emb_table
        
        # self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        # self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum + 1])
        # self.test_logits = self.test_logits[:, -1, :]

        # # prediction layer
        # self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        # self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)

        # # ignore padding items (0)
        # istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        # self.loss = tf.reduce_sum(
        #     - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
        #     tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        # ) / tf.reduce_sum(istarget)
        # # self.loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.test_logits, labels = self.pos[:,-1])) /10

        seq_emb = tf.concat([seq_emb, tf.reshape(self.seq_content, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])], -1)

        self.pos_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.pos_logits = tf.reshape(self.pos_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum + 1])
        # self.pos_logits = tf.reduce_sum(self.pos_logits, 1)
        self.pos_logits = self.pos_logits[:, -1, :]

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
