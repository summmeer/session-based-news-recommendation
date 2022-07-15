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

```
@inproceedings{10.1145/3477495.3532040,
    author = {Gong, Shansan and Zhu, Kenny Q.},
    title = {Positive, Negative and Neutral: Modeling Implicit Feedback in Session-Based News Recommendation},
    year = {2022},
    isbn = {9781450387323},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3477495.3532040},
    doi = {10.1145/3477495.3532040},
    abstract = {News recommendation for anonymous readers is a useful but challenging task for many news portals, where interactions between readers and articles are limited within a temporary login session. Previous works tend to formulate session-based recommendation as a next item prediction task, while they neglect the implicit feedback from user behaviors, which indicates what users really like or dislike. Hence, we propose a comprehensive framework to model user behaviors through positive feedback (i.e., the articles they spend more time on) and negative feedback (i.e., the articles they choose to skip without clicking in). Moreover, the framework implicitly models the user using their session start time, and the article using its initial publishing time, in what we call neutral feedback. Empirical evaluation on three real-world news datasets shows the framework's promising performance of more accurate, diverse and even unexpectedness recommendations than other state-of-the-art session-based recommendation approaches.},
    booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {1185–1195},
    numpages = {11},
    keywords = {news recommendation, cold-start, session-based, time aware, implicit feedback},
    location = {Madrid, Spain},
    series = {SIGIR '22}
}
```
