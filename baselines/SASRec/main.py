import os
import pickle
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
import pandas as pd
from utilglobo import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='../../data/mind/TCAR-mid/Normal/')
parser.add_argument('--fold', default=1)
parser.add_argument('--train_dir', default='default')
parser.add_argument('--neg_sample_num', default=20, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=0.0008, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_units', default=250, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()
# if not os.path.isdir(args.dataset + '_' + args.train_dir):
#     os.makedirs(args.dataset + '_' + args.train_dir)
# with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

dataset = data_partition(args.dataset, args.fold)
[train_id, train_session, train_timestamp, test_id, test_session, test_timestamp, test_predict, item_dict, content_weight] = dataset

# df_data = pd.read_csv('/home/sansa/dataset/globo/articles_metadata.csv', header=0, sep=',')
# category_id = df_data['category_id'].tolist()
## adressa
category_id = pickle.load(open('/home/sansa/dataset/MIND/articles_category.pkl', 'rb'))
reverse_item = {}
for idx, cnt in item_dict.items():
    reverse_item[cnt] = idx

item_freq_dict_norm = pickle.load(open(args.dataset + '/item_freq_dict_norm_' + str(args.fold) + '.txt', 'rb'))

num_batch = len(train_session) // args.batch_size
neg_sample_num = args.neg_sample_num

print('# train', len(train_id), '# test', len(test_id))
print('batch num', ' ', num_batch)

# f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

session_num = len(train_session)
itemnum = len(item_dict)
sampler = WarpSampler(train_session, session_num, itemnum, num_batch, neg_sample_num, batch_size=args.batch_size, maxlen=args.maxlen)
model = Model(session_num, itemnum, args, content_weight)
sess.run(tf.initialize_all_variables())

# writer = tf.summary.FileWriter('./logs', sess.graph)

T = 0.0
t0 = time.time()
table_dict = {}
for epoch in range(1, args.num_epochs + 1):
    #for step in tqdm(range(num_batch+1), total=num_batch+1, ncols=70, leave=False, unit='b'):
    for step in range(num_batch+1):
        seq, pos, neg = sampler.next_batch(step)
        #print(len(seq),len(pos),len(neg))
        #print(len(seq[0]),len(pos[0]),len(neg[0]))
        table, loss, _, merged = sess.run([model.item_emb_table, model.loss, model.train_op, model.merged],
                                {model.input_seq: seq, model.pos: pos, model.neg: neg, 
                                model.is_training: True})
    print('epoch',epoch,np.mean(loss))
    # table_dict[epoch] = table
    # writer.add_summary(merged, epoch)
    # pickle.dump(table_dict, open('./embeddingTable', 'wb'))

    if epoch % 1 == 0:
        t1 = time.time() - t0
        T += t1
        print('Evaluating')
        t_test = evaluate(model, test_id, test_session, item_dict, itemnum, test_predict, args, sess, args.fold, epoch, category_id, reverse_item, item_freq_dict_norm)
        print ('epoch:%d, time: %f(s), (NDCG@20: %.4f, HR@20: %.4f)' % (epoch, T, t_test[0], t_test[1]))

        # f.write(str(t_test) + '\n')
        # f.flush()
        t0 = time.time()

# writer.close()
# f.close()
print("Done")
