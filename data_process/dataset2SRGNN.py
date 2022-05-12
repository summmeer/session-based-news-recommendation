# import pickle
# import argparse

# def main(args):
#     if args.split_way == 'Normal':
#         folds = [1,2,3,4]
#     else:
#         folds = [1,2,3]
#     # folds = [1]
#     savefile = 'SRGNN-mid'
#     for fold in folds:
#         train_data = pickle.load(open('../data/globo/' + savefile + '/' + args.split_way + '/train_session_' + str(fold) + '.txt', 'rb'))
#         item_dict = pickle.load(open('../data/globo/' + savefile + '/' + args.split_way + '/item_dict_' + str(fold) + '.txt', 'rb'))
#         pickle.dump((train_data[1], train_data[3]), open('../data/globo/' + savefile + '/' + args.split_way + '/train_' + str(fold) + '_'+str(len(item_dict))+'.txt', 'wb'))
#         test_data = pickle.load(open('../data/globo/' + savefile + '/' + args.split_way + '/test_session_' + str(fold) + '.txt', 'rb'))
#         pickle.dump((test_data[1], test_data[3]), open('../data/globo/' + savefile + '/' + args.split_way + '/test_' + str(fold) + '_'+str(len(item_dict))+'.txt', 'wb'))

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--split_way', default='Normal', type=str, choices=['Normal', 'TrainLen-1', 'TestLen-1'], help='Choose different split ways')
#     args = parser.parse_args()

#     main(args)

import pickle
import argparse

def main():
    folds = [1, 2]
    # folds = [1]
    for fold in folds:
        train_data = pickle.load(open('../data/adressa/TCAR-mid/Normal/train_session_' + str(fold) + '.txt', 'rb'))
        item_dict = pickle.load(open('../data/adressa/TCAR-mid/Normal/item_dict_' + str(fold) + '.txt', 'rb'))
        pickle.dump((train_data[1], train_data[3]), open('../data/adressa/SRGNN-mid/Normal/train_' + str(fold) + '_'+str(len(item_dict))+'.txt', 'wb'))
        test_data = pickle.load(open('../data/adressa/TCAR-mid/Normal/test_session_' + str(fold) + '.txt', 'rb'))
        pickle.dump((test_data[1], test_data[3]), open('../data/adressa/SRGNN-mid/Normal/test_' + str(fold) + '_'+str(len(item_dict))+'.txt', 'wb'))

if __name__ == '__main__':

    main()