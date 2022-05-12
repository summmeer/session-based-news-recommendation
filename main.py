# coding=utf-8
from optparse import OptionParser
import tensorflow as tf
import pandas as pd
import numpy as np
import os, argparse
from util import *
# the data path.
import random
random.seed(2020)
from numpy.random import seed
seed(2020)

def load_datas(args):
    '''
    loda data and pre_embedding.
    '''
    print( "load the datasets.")

    dataset = data_partition(args.datapath + args.dataset + args.split_way, args.foldnum)
    train_data, test_data, item_dict, neighbor, content_emb, publish_time, candidate_mask = dataset
    item_freq_dict_norm = pickle.load(open(args.datapath + args.dataset + args.split_way + 'item_freq_dict_norm_' + str(args.foldnum) + '.txt', 'rb'))

    args = vars(args)
    args["itemnum"] = len(item_dict)
    
    reverse_item = {}
    for idx, cnt in item_dict.items():
        reverse_item[cnt-1] = idx
    args['reverse_item'] = reverse_item

    # df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
    # category_id = df_data['category_id'].tolist()
    ## adressa
    # category_id = pickle.load(open('/home/sansa/dataset/Adressa/articles_category.pkl', 'rb'))
    category_id = pickle.load(open('/home/sansa/dataset/MIND/articles_category.pkl', 'rb'))

    args['category_id'] = category_id
    args['item_freq_dict_norm'] = item_freq_dict_norm

    args['publish_time'] = publish_time[0]
    args['publish_time_MWDHM'] = publish_time[1]
    args['content_emb'] = content_emb
    # args['candidate_mask'] = candidate_mask
    print('------', len(item_dict), len(publish_time[1]))

    return train_data, test_data, neighbor, args, item_dict


def main(args):
    '''
    model: model to use
    '''
    is_train = args.train
    is_save = args.save
    model_path = args.modelpath
    input_data = args.inputdata
    modelname = args.model

    train_data, test_data, neighbor, args, item_dict = load_datas(args)

    with tf.Graph().as_default():
        # build model
        tf.set_random_seed(2020)
        modelname = __import__(modelname, fromlist=True)
        model = getattr(modelname, "Seq2SeqAttNN")(args)

        if is_save or not is_train:
            saver = tf.train.Saver(max_to_keep=30)
        else:
            saver = None
        # run
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            if is_train:
                print('Begin Training')
                model.train(sess, item_dict, train_data, neighbor, args, test_data, saver)
            else:
                if input_data == "train":
                    sent_data = train_data
                    print('Begin Testing. Test data is train data')
                else:
                    sent_data = test_data
                    print('Begin Testing. Test data is test data')
                saver.restore(sess, model_path)
                model.test(sess, sent_data, args)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # DATASET PARAMETERS
    parser.add_argument('--datapath', default='./data/', type=str,
                        help='Location of pre-processed dataset')
    parser.add_argument('--dataset', default='mind/TCAR-mid/', type=str, help='Dataset')
    parser.add_argument('--split_way', default='Normal/', type=str, choices=['Normal/', 'TrainLen/', 'TestLen/'], help='Choose different split ways')
    parser.add_argument('--foldnum', default=1, type=int, help='The fold number of pre-processed dataset')
 
    # TRAIN PARAMETERS
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
    parser.add_argument('--maxlen', default=20, type=int, help='Number of max window size')
    parser.add_argument('--neg_num', default=20, type=int, help='Number of neg samples')

    # MODEL PARAMETERS
    parser.add_argument('--model', default="model_combine", type=str, help="Model to use")
    parser.add_argument('--hidden_size', default=250, type=int)
    parser.add_argument('--time_hidden_size', default=64, type=int)
    parser.add_argument('--max_grad', default=150, type=int)
    parser.add_argument('--stddev', default=0.05, type=float)
    parser.add_argument('--emb_stddev', default=0.002, type=float)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)  

    # OTHER SETTING.
    parser.add_argument('--save', default=False, type=bool, help='Save model and test results')
    parser.add_argument('--is_print', default=False, type=bool, help='Save model and test results')
    parser.add_argument('--train', default=True, type=bool, help='Train or just test the specified model')
    parser.add_argument('--modelpath', default='./ckpt/', type=str, help='File to save model checkpoints')
    parser.add_argument('--inputdata', default='test', type=str, help='Use train or test data to test model')
    parser.add_argument('--threshold_acc', default=0.27, type=float, help='Accuracy threshold to save')

    args = parser.parse_args()
    
    main(args)
