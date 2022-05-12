# -*- coding: utf-8 -*-
"""
Created on Feb 26, 2017
@author: Weiping Song
"""
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.contrib import legacy_seq2seq
import numpy as np

class GRU4Rec:
    def __init__(self, args, content_weight):
        self.args = args
        if args.hidden_act == 'tanh':
            self.hidden_act = self.tanh
        elif args.hidden_act == 'relu':
            self.hidden_act = self.relu
        else:
            raise NotImplementedError

        self.build_model(content_weight)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)

    ########################ACTIVATION FUNCTIONS#########################
    def linear(self, X):
        return X
    def tanh(self, X):
        return tf.nn.tanh(X)
    def softmax(self, X):
        return tf.nn.softmax(X)
    def softmaxth(self, X):
        return tf.nn.softmax(tf.tanh(X))
    def relu(self, X):
        return tf.nn.relu(X)
    def sigmoid(self, X):
        return tf.nn.sigmoid(X)

    def build_model(self, content_weight):
        # input X and target Y, last state of last batch, lengths of sessions in current batch.
        self.X = tf.placeholder(tf.int32, [None, None], name='input')
        self.Y = tf.placeholder(tf.int32, [None, None], name='output')
        self.sess_len = tf.count_nonzero(self.X, 1)
        self.mask = tf.reshape(tf.to_float(tf.not_equal(self.X, 0)), (-1,))
        batch_size = tf.shape(self.X)[0]

        self.content_emb = tf.Variable(content_weight, dtype=tf.float32, trainable=False)
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('LSTM_layer'):
            sigma = self.args.sigma if self.args.sigma != 0 else np.sqrt(6.0 / (self.args.n_items + self.args.rnn_size))
            if self.args.init_as_normal:
                initializer = tf.random_normal_initializer(mean=0, stddev=sigma)
            else:
                initializer = tf.random_uniform_initializer(minval=-sigma, maxval=sigma)
            embedding = tf.get_variable('embedding', [self.args.n_items + 1, self.args.rnn_size])
            softmax_W = tf.get_variable('softmax_w', [self.args.rnn_size*2, 1 + self.args.n_items])
            softmax_b = tf.get_variable('softmax_b', [self.args.n_items + 1])

            cells = []
            for _ in range(self.args.layers):
                cell = rnn.BasicLSTMCell(self.args.rnn_size*2, activation=self.hidden_act)
                if self.args.is_training and (self.args.keep_prob < 1.0):
                    cell = rnn.DropoutWrapper(cell, output_keep_prob=self.args.keep_prob)
                cells.append(cell)
            self.cell = cell = rnn.MultiRNNCell(cells)


            zero_state = cell.zero_state(batch_size, dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.X)
            inputs_cont = tf.nn.embedding_lookup(self.content_emb, self.X, max_norm=1)
            inputs = tf.concat([inputs, inputs_cont], -1)
            outputs, state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=self.sess_len, initial_state=zero_state)
            self.final_state = state
            # output = tf.reshape(outputs[:,-1,:], [-1, self.args.rnn_size*2])
            output = tf.reshape(outputs, [-1, self.args.rnn_size*2])

        if self.args.is_training:
            # logits = tf.matmul(output, softmax_W) + softmax_b
            logits = tf.matmul(output, tf.concat([embedding, self.content_emb], -1), transpose_b=True)
            self.logits = logits
            label = tf.reshape(self.Y, (-1,))
            loss = legacy_seq2seq.sequence_loss_by_example([logits], [self.Y], [tf.ones([tf.shape(logits)[0]])])
            self.nan = tf.reduce_sum(tf.to_float(tf.is_nan(loss)))
            mask_loss = loss * self.mask
            self.cost = tf.reduce_sum(mask_loss) / tf.reduce_sum(self.mask)
            # self.cost = tf.reduce_sum(loss)

        self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.args.learning_rate, self.global_step, self.args.decay_steps, self.args.decay, staircase=True)) 
        optimizer = tf.train.AdamOptimizer(self.lr)
        
        tvars = tf.trainable_variables()
        gvs = optimizer.compute_gradients(self.cost, tvars)
        if self.args.grad_cap > 0:
            capped_gvs = [(tf.clip_by_norm(grad, self.args.grad_cap), var) for grad, var in gvs]
        else:
            capped_gvs = gvs 
        self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

