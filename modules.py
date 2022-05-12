# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import numpy as np
from util import *
# from numpy.random import seed
# seed(21)
# import sys
# sys.path.append("..") 
# from STAMP.FwNn3AttLayer import FwNnAttLayer

def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True, stddev=0.02, l2_reg=0.0, scope="embedding", with_t=False, reuse=False):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.
    '''
    init_table = np.random.normal(0, stddev, [vocab_size, num_units])
    if zero_pad:
        init_table[0] = [0.0] * num_units
    lookup_table = tf.Variable(init_table, dtype=tf.float32, trainable=True, name=scope)
    outputs = tf.nn.embedding_lookup(lookup_table, inputs, max_norm=1)

    if scale:
        outputs = outputs * (num_units ** 0.5)
    if with_t: return outputs, lookup_table
    else: return outputs

def linear_2d(inputs, edim1, out_edim, stddev=0.05, scope="linear_2d_layer", active='tanh', reuse=False):
    '''
    the linear transformation layer.
    inputs: input 2d vector # shape: [batch_size, edim1]
    return = input * w + b # shape: [batch_size, out_edim]
    '''
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.Variable(tf.random_normal([edim1, out_edim], stddev=stddev), name='w1', trainable=True)
        b = tf.Variable(tf.random_normal([out_edim], stddev=stddev), name='b1', trainable=True)
        res = tf.matmul(inputs, w) + b
        if active!=None:
            res = activer(res, active)
        return res

def linear_3d(inputs, edim1, out_edim, stddev=0.05, scope="linear_3d_layer", active='tanh', reuse=False):
    '''
    the linear transformation layer.
    inputs: input 3d vector # shape: [batch_size, time_step, edim1]
    return = input * w + b # shape: [batch_size, time_step, out_edim]
    '''
    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope, reuse=reuse):
        w = tf.Variable(tf.random_normal([edim1, out_edim], stddev=stddev), name='w_3d', trainable=True)
        w = tf.reshape(tf.tile(w, [batch_size, 1]), [batch_size, edim1, out_edim])
        res = tf.matmul(inputs, w)
        if active!=None:
            res = activer(res, active)
        return res

def single_attention_layer(inputs, content, edim1, edim2, scope="content_attention", hidden_size=250, stddev=0.05, reuse=False):
    '''
    add content attention
    inputs: input 3d vector # shape: [batch_size, time_step, edim]
    return weighted sum # shape: [batch_size, edim]
    '''
    time_step = tf.shape(inputs)[1]
    edim = tf.shape(inputs)[2]
    with tf.variable_scope(scope, reuse=reuse):
        alpha = count_alpha_s(inputs, content, edim1, edim2, hidden_size, stddev) # [batch_size, time_step]
        res = tf.matmul(tf.reshape(alpha, [-1, 1, time_step]), inputs) #  [batch_size, 1, edim]
        res = tf.reshape(res, [-1, edim])
        return res, alpha

def count_alpha_s(inputs, content, edim1, edim2, hidden_size, stddev, active='sigmoid'):
    '''
    count attention weights
    inputs: input 3d vector # shape: [batch_size, time_step, edim1]
    content: input 3d vector # shape: [batch_size, time_step, edim2]
    return alpha = softmax(tanh(w1*inputs+w2*content)) # shape: [batch_size, time_step]
    '''
    time_step = tf.shape(inputs)[1]
    res_input = linear_3d(inputs, edim1, hidden_size, stddev, "input_linear_trans", active=None) # [batch_size, time_step, hidden_size]
    res_cont = linear_3d(content, edim2, hidden_size, stddev, "cont_linear_trans", active=None) # [batch_size, time_step, hidden_size]
    res_sum = res_input + res_cont
    res_act = activer(res_sum, active) # [batch_size, time_step, hidden_size]
    res_act = linear_3d(res_act, hidden_size, 1, stddev, "res_linear_trans", active=None) # [batch_size, time_step, 1]
    res_act = tf.reshape(res_act, [-1, time_step])
    alpha = normalizer(res_act)
    return alpha

def multi_attention_layer(inputs, content, interval, click_time, edim1, edim2, edim3, scope="multi_attention", hidden_size=250, stddev=0.05, reuse=False):
    '''
    add attention
    inputs: input 3d vector # shape: [batch_size, time_step, edim1]
    content: input 3d vector # shape: [batch_size, time_step, edim2]
    interval: numerical time # shape: [batch_size, time_step, 1]
    click_time: session_level time vector # shape: [batch_size, edim3]
    return weighted sum # shape: [batch_size, edim1]
    '''
    time_step = tf.shape(inputs)[1]
    edim = tf.shape(inputs)[2]
    with tf.variable_scope(scope, reuse=reuse):
        alpha = count_alpha_m(inputs, content, interval, click_time, edim1, edim2, edim3, hidden_size, stddev) # [batch_size, time_step]
        res = tf.matmul(tf.reshape(alpha, [-1, 1, time_step]), inputs) #  [batch_size, 1, edim]
        res = tf.reshape(res, [-1, edim])
        return res, alpha

def count_alpha_m(inputs, content, interval, click_time, edim1, edim2, edim3, hidden_size, stddev, active='sigmoid'):
    '''
    count attention weights
    return alpha = softmax(tanh(w1*inputs+w2*content+w3*interval)) + softmax(tanh(inputs*w4*click_time)) # shape: [batch_size, time_step]
    '''
    time_step = tf.shape(inputs)[1]
    res_input = linear_3d(inputs, edim1, hidden_size, stddev, "input_linear_trans", active=None) # [batch_size, time_step, hidden_size]
    res_cont = linear_3d(content, edim2, hidden_size, stddev, "cont_linear_trans", active=None) # [batch_size, time_step, hidden_size]
    res_sum = res_input + res_cont
    if interval!=None:
        res_inter = linear_3d(interval, 1, hidden_size, stddev, "inter_linear_trans", active=None) # [batch_size, time_step, hidden_size]
        res_sum += res_inter
    res_act1 = activer(res_sum, active) # [batch_size, time_step, hidden_size]
    res_act1 = linear_3d(res_act1, hidden_size, 1, stddev, "res_linear_trans", active=None) # [batch_size, time_step, 1]
    res_act1 = tf.reshape(res_act1, [-1, time_step])
    alpha = normalizer(res_act1)

    if click_time!=None:
        query = linear_2d(click_time, 128, hidden_size, stddev, "query_trans1", active='relu') # [batch_size, hidden_size]
        query = linear_2d(query, hidden_size, edim1, stddev, "query_trans2") # [batch_size, edim1]
        res_act2 = tf.matmul(inputs, tf.expand_dims(query, -1)) # [batch_size, time_step, 1]
        alpha2 = tf.reshape(normalizer(res_act2), [-1, time_step])
        alpha += alpha2
    
    # if interval!=None:
    #     click_time = tf.reshape(tf.tile(click_time, [time_step, 1]), [-1, time_step, edim3])
    #     query = linear_3d(click_time, edim3, hidden_size, stddev, "inter_query_trans1", active=None) # [batch_size, time_step, hidden_size]
    #     key = linear_3d(interval, edim3//2, hidden_size, stddev, "inter_query_trans2", active=None) # [batch_size, time_step, hidden_size]
    #     res_act3 = tf.matmul(tf.transpose(tf.expand_dims(query, -1), perm=[0, 1, 3, 2]), tf.expand_dims(key, -1)) # [batch_size, time_step, 1]
    #     alpha3 = tf.reshape(normalizer(res_act3), [-1, time_step])
    #     alpha += alpha3

    return alpha

# def multi_attention_layer(inputs, interval, click_time, edim1, edim2, scope="multi_attention", hidden_size=250, stddev=0.05, reuse=False):
#     '''
#     add attention
#     inputs: input 3d vector # shape: [batch_size, time_step, edim1]
#     interval: numerical time # shape: [batch_size, time_step, 1]
#     click_time: session_level time vector # shape: [batch_size, edim2]
#     return weighted sum # shape: [batch_size, edim1]
#     '''
#     time_step = tf.shape(inputs)[1]
#     edim = tf.shape(inputs)[2]
#     with tf.variable_scope(scope, reuse=reuse):
#         alpha = count_alpha_m(inputs, interval, click_time, edim1, edim2, hidden_size, stddev) # [batch_size, time_step]
#         res = tf.matmul(tf.reshape(alpha, [-1, 1, time_step]), inputs) #  [batch_size, 1, edim]
#         res = tf.reshape(res, [-1, edim])
#         return res, alpha

# def count_alpha_m(inputs, interval, click_time, edim1, edim2, hidden_size, stddev, active='sigmoid'):
#     '''
#     count attention weights
#     return alpha = softmax(tanh(w1*inputs+w2*content+w3*interval)) + softmax(tanh(inputs*w4*click_time)) # shape: [batch_size, time_step]
#     '''
#     time_step = tf.shape(inputs)[1]
#     res_input = linear_3d(inputs, edim1, hidden_size, stddev, "input_linear_trans", active=None) # [batch_size, time_step, hidden_size]
#     res_sum = res_input
#     if interval!=None:
#         res_inter = linear_3d(interval, 1, hidden_size, stddev, "inter_linear_trans", active=None) # [batch_size, time_step, hidden_size]
#         res_sum += res_inter
#     res_act1 = activer(res_sum, active) # [batch_size, time_step, hidden_size]
#     res_act1 = linear_3d(res_act1, hidden_size, 1, stddev, "res_linear_trans", active=None) # [batch_size, time_step, 1]
#     res_act1 = tf.reshape(res_act1, [-1, time_step])
#     alpha = normalizer(res_act1)

#     if click_time!=None:
#         query = linear_2d(click_time, edim2, hidden_size, stddev, "query_trans1", active='relu') # [batch_size, hidden_size]
#         query = linear_2d(query, hidden_size, edim1, stddev, "query_trans2") # [batch_size, edim1]
#         res_act2 = tf.matmul(inputs, tf.expand_dims(query, -1)) # [batch_size, time_step, 1]
#         alpha2 = tf.reshape(normalizer(res_act2), [-1, time_step])
#         alpha += alpha2
#     return alpha

def normalize(inputs, epsilon = 1e-8, scope="ln", reuse=None):
    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True, causality=False, scope="multihead_attention", reuse=None, with_qk=False):
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: return Q,K
    else: return outputs

def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention", dropout_rate=0.2, is_training=True, reuse=None):
    '''Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs