import fasttext
import fasttext.util
import numpy as np
import json
import pickle
from bs4 import BeautifulSoup
import urllib.request
import time
import random

from urllib.parse import unquote_plus

def removeAllapostrophe(string):
    string=string.replace('.','')
    string=string.replace(',','')
    string=string.replace('_',' ')
    string=string.replace('?','')
    string=string.replace('"','')
    string=string.replace('/',' ')
    string=string.replace('\\','')
    string=string.replace('`','')
    string=string.replace(';','')
    string=string.replace('!','')
    string=string.replace(':','')
    string=string.replace("'",'')
    string=string.replace("#",'')
    string=string.replace("%",'')
    string=string.replace("+",'')
    string=string.replace("//",' ')
    string=string.replace("adressa",'')
    string=string.replace("html",'')
    string=string.replace("no",'')
    string=string.replace("1",' ')
    return string

def cleanUrl(url):
    title=unquote_plus(url)
    title=removeAllapostrophe(title)
    return title

parser='html.parser'
def getUrlTitle(url):
    req = urllib.request.Request(url, headers = {
        'Connection': 'Keep-Alive',
        'Accept': 'text/html, application/xhtml+xml, */*',
        'Accept-Language': 'en-US,en;q=0.8,zh-Hans-CN;q=0.5,zh-Hans;q=0.3',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko'
    })
    SUCCESS = False
    error_count = 0
    while not SUCCESS:
        SUCCESS = True
        try:
            resp = urllib.request.urlopen(req, timeout=10)
            soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
            title = soup.title.text
            return title[:-13].replace('\n','')
        except Exception as e:
            print (e)
            error_count += 1
            if (error_count >= 5):
                return ''
            print("Exception at " + url)
            sleepTime = random.randint(3, 5)
            time.sleep(sleepTime)
            SUCCESS = False

# Loading model 

ft = fasttext.load_model('/home/sansa/fastText/cc.no.300.bin')
# ft = fasttext.load_model('cc.en.300.bin')
print(ft.get_dimension())
fasttext.util.reduce_model(ft, 250)
print(ft.get_dimension())

articles_id_list = pickle.load(open('../data/adressa/articles_list.txt', 'rb'))
for line in open('/home/sansa/dataset/Adressa/AdressaDataLSTMRecommender/articles_unfiltered_withword.json'):
    articles = json.loads(line)

miss = 0
total = 0
# articles_content = {}
# articles_content_vector = {}
articles_content = pickle.load(open('../data/adressa/articles_titles_3.pkl', 'rb'))
articles_content_vector = pickle.load(open('../data/adressa/articles_embeddings_3.pkl', 'rb'))
for a_id in articles_id_list:
    if a_id in articles_content:
        continue
    total += 1
    title = ' '.join(articles[a_id]['title'])
    if len(title)>0:
        vector = ft.get_sentence_vector(title)
        articles_content_vector[a_id] = vector
        articles_content[a_id] = title
    else:
        title = cleanUrl(a_id)
        if len(title)>0:
            vector = ft.get_sentence_vector(title)
            articles_content_vector[a_id] = vector
            articles_content[a_id] = title
        else:
            miss += 1
print('miss', miss, 'total', total)
pickle.dump(articles_content, open('../data/adressa/articles_titles_4.pkl', 'wb'))
pickle.dump(articles_content_vector, open('../data/adressa/articles_embeddings_4.pkl', 'wb'))

# title = cleanUrl('http://adressa.no/bolig/boligguiden/trondheim/trondheim-%c3%b8st/bolig1193325.html')
# print(title)
