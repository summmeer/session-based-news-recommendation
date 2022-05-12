import pickle
import argparse
import random
import numpy as np


def get_neighbor(publish_time):
    sort_pt = np.argsort(np.array(publish_time))
    window = 100
    neighbor_dict = {}
    cnt = 0
    item_num = len(publish_time)
    for i in sort_pt:
        left = max(0, cnt-window)
        right = min(cnt+window, item_num)
        neighbor_dict[i] = sort_pt[left:right]
        cnt += 1
    for i, lt in neighbor_dict.items():
        if len(lt)<100:
            neighbor_dict[i] = np.append(neighbor_dict[i], np.array(random.sample(range(0, item_num-1), 100-len(lt))))
    return neighbor_dict

def main(args):
    if args.split_way == 'Normal':
        folds = [1,2]
    else:
        folds = [1,2,3]
    # folds = [1]
    savefile = 'TCAR-mid'
    for fold in folds:
        publish_time = pickle.load(open('../data/adressa/' + savefile + '/' + args.split_way + '/publish_time_' + str(fold) + '.txt', 'rb'))[0]
        neighbor_dict = get_neighbor(publish_time)
        print(len(neighbor_dict))
        pickle.dump(neighbor_dict, open('../data/adressa/' + savefile + '/' + args.split_way + '/neighbor_' + str(fold) + '.txt', 'wb'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split_way', default='Normal', type=str, choices=['Normal', 'TrainLen', 'TestLen'], help='Choose different split ways')
    args = parser.parse_args()

    main(args)