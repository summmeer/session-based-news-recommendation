This is the source code of paper "Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation", which is accepted at SIGIR 2022.

By leveraging different kinds of implicit feedback, we alleviate the trade-off between the precision and diversity and cold-start problem, which is effective for real-world application.

Our model is named TCAR (Temporal and Content Aware Recommendation System)

## Dataset prepare
We use 3 dataset:
- Globo.com <https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom> download the data to the folder `../data/globo/`
- Adressa <https://reclab.idi.ntnu.no/dataset/> (Need contact them for the full dataset) download the data to the folder `../data/adressa/`. We use a crawler to get the titles of news articles in this dataset, the code is in ```data_process/get_content_vec.py``` 
- MIND <https://msnews.github.io/> download the data to the folder `../data/mind/`

## Session data preprocessing
create middle file:
```
cd data_process
python globo_preprocess.py --use_preprocess
```
create file for TCAR (cnt for cold item, with augmentation)
```
python globo_preprocess.py --content_info
python globo_preprocess.py --split_way=TrainLen --content_info
python globo_preprocess.py --split_way=TestLen --content_info
```
create file for CBCF (cnt for cold item, without augmentation)
```
python globo_preprocess.py --train_augment
python globo_preprocess.py --split_way=TrainLen --train_augment
python globo_preprocess.py --split_way=TestLen --train_augment
```

create file for GRU-STAMP-mid (ignore the cold item, no augmentation)
```
python dataset2STAMP.py

python globo_preprocess.py --cold_start --train_augment --split_way=TrainLen
python dataset2STAMP.py --split_way=TrainLen

python globo_preprocess.py --cold_start --train_augment --split_way=TestLen
python dataset2STAMP.py --split_way=TestLen
```

create file for SR-GNN (ignore cold item, with data augmentation)
```
python dataset2SRGNN.py

python globo_preprocess.py --cold_start --split_way=TrainLen
python dataset2SRGNN.py --split_way=TrainLen

python globo_preprocess.py --cold_start --split_way=TestLen
python dataset2SRGNN.py --split_way=TestLen
```
Then you can run baselines: CBCF, STAN, GRU4Rec, SASRec, [STAMP](https://github.com/uestcnlp/STAMP), [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN), SGNN-HN(upon request) , CPRS.

For Adressa and MIND, the procedure is the same.

## TCAR data pre-processing
To leverage the negative feedback, you need to generate the inferenced the impression lists and sample negative points from them. The code is in ```data_process/generate_neighbor.py```

## Training & testing
Need to specify the model name, dataset, foldnum, etc. (See `python main.py -h`)
```
python main.py --foldnum=0  --epoch=10
```

## Experiment Results
We use F1 score to measure the trade-off between the accuracy and diversity.

![image](https://github.com/summmeer/session-based-news-recommendation/blob/master/results.png)

## Citation

Shansan Gong and Kenny Q. Zhu. Positive, Negative and Neutral: Modeling Implicit Feedback in Session-based News Recommendation. In *Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.*