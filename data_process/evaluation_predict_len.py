import pickle
import argparse
import math

def evaluation(test_file='../saved/CAR+P_Normal_predict_exa_4_1.txt', maxid=0, display=1):
    # assume user/item index starting from 1
    file = open(test_file, 'r')
    batch_in = []
    batch_out = []
    batch_pred = []
    cntline = 0
    for line in file.readlines():
        cntline += 1
        if cntline<=display:
            print(line)
        in_ = [int(x) for x in line.split("# batch in: [")[1].split("]")[0].split(", ")]
        # out_ = int(line.split("# batch out: ")[1].split("[")[0])
        out_ = int(line.split("# batch out: ")[1].split(" #")[0])
        pred_split = line.split("# batch pred: [")[1].split("]")[0].split(", ")
        if pred_split[0] == '':
            pred_ = []
        else:
            pred_ = [int(x) for x in pred_split]
        batch_in.append(in_)
        batch_out.append(out_)
        batch_pred.append(pred_)
    file.close()
    input_len_dict = {}
    for in_, out_, pred_ in zip(batch_in, batch_out, batch_pred):
        while in_[-1] == 0:
            in_ = in_[:-1]
        input_len = len(in_)
        hr = 0
        if out_ in pred_:
            hr = 1
        if input_len in input_len_dict:
            input_len_dict[input_len] += [hr]
        else:
            input_len_dict[input_len] = [hr]

    print('total case..', len(batch_in))

    len_score = [0]*9
    over9 = []
    for k, v in input_len_dict.items():
        if k<9:
            len_score[k] = sum(v)/len(v)
        else:
            over9 += v
    # len_score[9] = (sum(over9)/len(over9))

    print('hr@20 for diff lens')
    print('len of over 9 seq', len(over9))
    for s in len_score:
        print(s)
    print(sum(over9)/len(over9))

    return batch_in, batch_out, batch_pred

def main(args):
    # filepath = args.data_path + args.baseline + '/saved/' + args.attr + args.split_way + '_predict_' + str(args.fold) + '_' + str(args.best_epoch) + '.txt'
    # evaluation(filepath, args.maxid)
    evaluation()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../baselines/', type=str, help='Location of file')
    parser.add_argument('--baseline', default='SR-GNN', type=str, help='Location of file')
    parser.add_argument('--split_way', default='Normal', type=str, choices=['Normal', 'TrainLen', 'TestLen'], help='Choose different split ways')
    parser.add_argument('--attr', default='', type=str, help='Other part in file name')
    parser.add_argument('--fold', default=1, type=int, help='Choose fold')
    parser.add_argument('--best_epoch', default=0, type=int, help='Choose epoch')
    parser.add_argument('--maxid', default=0, type=int, help='Choose maxid')

    args = parser.parse_args()

    main(args)