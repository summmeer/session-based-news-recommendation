import pickle
import argparse
import math

def evaluation(test_file='../../STAMP-test/test_batch.txt', maxid=0, display=1):
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
    recall_cold = 0
    recall_notcold = 0
    ndcg = 0
    cntCold = 0
    item_deversity_cold = {}
    item_deversity_notcold = {}
    item_deversity_label_cold = {}
    item_deversity_label_notcold = {}
    item_deversity_all = {}
    item_deversity_label_all = {}
    cntPredictCold_cold = 0
    cntPredictCold_notcold = 0
    for in_, out_, pred_ in zip(batch_in, batch_out, batch_pred):
        isCold = True
        for item in in_:
            if maxid==0:
                if not item==maxid:
                    isCold = False
            else:
                if item<=maxid:
                    isCold = False
        if isCold:
            cntCold+=1
            for item in pred_:
                item_deversity_cold[item] = item_deversity_cold.get(item, 0) + 1
                item_deversity_all[item] = item_deversity_all.get(item, 0) + 1
                if maxid==0:
                    if item==maxid:
                        cntPredictCold_cold += 1
                else:
                    if item>maxid:
                        cntPredictCold_cold += 1
            if out_ in pred_:
                recall_cold += 1
                rank = pred_.index(out_) + 1
                ndcg += 1 / math.log(rank + 1, 2)
            item_deversity_label_cold[out_] = item_deversity_label_cold.get(out_, 0) + 1
            item_deversity_label_all[out_] = item_deversity_label_all.get(out_, 0) + 1
        else:
            for item in pred_:
                item_deversity_notcold[item] = item_deversity_notcold.get(item, 0) + 1
                item_deversity_all[item] = item_deversity_all.get(item, 0) + 1
                if maxid==0:
                    if item==maxid:
                        cntPredictCold_notcold += 1
                else:
                    if item>maxid:
                        cntPredictCold_notcold += 1
            if out_ in pred_:
                recall_notcold += 1
                rank = pred_.index(out_) + 1
                ndcg += 1 / math.log(rank + 1, 2)
            item_deversity_label_notcold[out_] = item_deversity_label_notcold.get(out_, 0) + 1
            item_deversity_label_all[out_] = item_deversity_label_all.get(out_, 0) + 1
    print('#### cold ratio: {}, {:.2f}%'.format(cntCold, cntCold/(len(batch_out))*100))
    print('% cold predictions in cold situation {:.2f}%, in notcold situation {:.2f}%.'.format(cntPredictCold_cold/(cntCold)*100, cntPredictCold_notcold/(len(batch_out)-cntCold)*100))
    print('# recall_cold: {}, {:.2f}%'.format(recall_cold, recall_cold/(cntCold)*100))
    print('# deversity of cold', len(item_deversity_cold), len(item_deversity_cold)/len(item_deversity_label_cold), 'in', len(item_deversity_label_cold))
    print('# recall_notcold: {}, {:.2f}%'.format(recall_notcold, recall_notcold/(len(batch_out)-cntCold)*100))
    print('# deversity of not cold', len(item_deversity_notcold), len(item_deversity_notcold)/len(item_deversity_label_notcold), 'in', len(item_deversity_label_notcold))
    print('#### total recall', (recall_cold+recall_notcold)/len(batch_out)*100)
    print('#### total ndcg', ndcg/len(batch_out)*100)
    print('#### total deversity', len(item_deversity_all))
    print('total cases', cntline)
    return batch_in, batch_out, batch_pred

def main(args):
    filepath = args.data_path + args.baseline + '/saved/' + args.attr + args.split_way + '_predict_' + str(args.fold) + '_' + str(args.best_epoch) + '.txt'
    evaluation(filepath, args.maxid)

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