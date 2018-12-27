# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 11:20:48 2018

@author: User
"""

# 사용 라이브러리.
import nltk
import re
import collections
import itertools
import lda
import requests
import csv
import time
import math
import operator
import numpy as np
import pandas as pd
from collections import defaultdict
from pandas import read_table
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# 데이터 크롤링 과정.
# 긍정 리뷰 크롤링
j = 0
pos_review = []
for i in range(1, 10001):
    json_url = 'https://steamcommunity.com/app/578080/homecontent/?userreviewsoffset={0}&p={1}&workshopitemspage={1}&readytouseitemspage={1}&mtxitemspage={1}&itemspage={1}&screenshotspage={1}&videospage={1}&artpage={1}&allguidepage={1}&webguidepage={1}&integratedguidepage={1}&discussionspage={1}&numperpage=10&browsefilter=toprated&browsefilter=toprated&appid=578080&appHubSubSection=16&appHubSubSection=16&l=english&filterLanguage=default&searchText=&forceanon=1'.format(j, i)
    i += 1
    j += 10 

    html = requests.get(json_url).text
    soup = BeautifulSoup(html, 'html.parser')
    
    for div in soup.find_all("div", {'class':"date_posted"}): 
        div.decompose()

    for div in soup.find_all("div", {'class':'early_access_review'}): 
        div.decompose() 

    bus = soup.select('div[class=apphub_CardTextContent]')

    for text in bus:
        txt = text.get_text()
        pos = str(txt)
        pos = pos.strip()
        pos = pos.replace("\t", '')
        pos = pos.replace('\n', '')
        pos_review.append(pos)
    
    if j % 5 == 0:
        time.sleep(10)
    else: continue

# 크롤링 데이터 CSV로 저장.         
csv = open('pos_review.csv', 'w', encoding = 'UTF-8')

for i in range(len(pos_review)):
    csv.write(pos_review[i] + '\n')
csv.close()

# 부정 리뷰 크롤링
j = 0
neg_review = []
for i in range(1, 10001):
    json_url = 'https://steamcommunity.com/app/578080/homecontent/?userreviewsoffset={0}&p={1}&workshopitemspage={1}&readytouseitemspage={1}&mtxitemspage={1}&itemspage={1}&screenshotspage={1}&videospage={1}&artpage={1}&allguidepage={1}&webguidepage={1}&integratedguidepage={1}&discussionspage={1}&numperpage=10&browsefilter=toprated&browsefilter=toprated&appid=578080&appHubSubSection=17&appHubSubSection=17&l=english&filterLanguage=default&searchText=&forceanon=1'.format(j, i)
    i += 1
    j += 10 

    html = requests.get(json_url).text
    soup = BeautifulSoup(html, 'html.parser')
    
    for div in soup.find_all("div", {'class':"date_posted"}): 
        div.decompose()

    for div in soup.find_all("div", {'class':'early_access_review'}): 
        div.decompose() 

    bus = soup.select('div[class=apphub_CardTextContent]')

    for text in bus:
        txt = text.get_text()
        neg = str(txt)
        neg = neg.strip()
        neg = neg.replace("\t", '')
        neg = neg.replace('\n', '')
        neg_review.append(neg)
    
    if j % 5 == 0:
        time.sleep(10)
    else: continue

# 크롤링 데이터 CSV로 저장.         
csv = open('neg_review.csv', 'w', encoding = 'UTF-8')

for i in range(len(neg_review)):
    csv.write(neg_review[i] + '\n')
csv.close()

# CSV 데이터의 분리된 row 값 전처리
review = pd.read_csv('C:\\Users\\user\\Desktop\\pos_review.csv', header = None, encoding = 'utf-8')
#test = review.head(100)

li_review = review.values.tolist()

re_list = []
for i in range(len(li_review)):
    line = ''
    for j in range(len(li_review[0])):
        
        text = str(li_review[i][j])
        if text.lower() != 'nan':
            line += text
    re_list.append(line)

    
df = pd.DataFrame(re, columns = ['review'])    
df.to_csv('C:\\Users\\user\\Desktop\\merge_positive.csv', sep = '\t', encoding = 'utf-8', index = False)

# 데이터 전처리 과정
# data csv 파일 가져오기.
data = pd.read_csv(r'E:/unstruct/merge.csv')

# df.dataframe 저장용 리스트 만들기.
data_list = []
# 리스트에 csv 파일 한줄씩 가져오기.
for i in data['0']:
    data_list.append(str(i))
print(len(data_list))

# 위에 리스트 파일 DataFrame으로 저장.
df = pd.DataFrame(data_list,columns=['review'])
df.to_csv(r'E:\unstruct\test.csv',columns=['review'])

data_list = pd.read_csv(r'E:/unstruct/merge.csv')
data_list = data_list['review']
data_list[0:10]

# 영어 숫자 특수기호 남기기
pre_list = []
for i in data_list:
    i = str(i)
    text = re.sub('[^a-zA-Z0-9]',' ',i).strip()
    text = re.sub('[,]','',text)
    text = re.sub('  ','',text)
    if(text != ''):
        if(text[0] !='?'):
            pre_list.append(text)

# 파일 저장
df = pd.DataFrame(pre_list,columns=['review'])
df.to_csv(r'E:\unstruct\Merge_data.csv',columns=['review'])

# 불필요 어구 제거.
nnp_list = []

for i in pre_list:
    origin_words = nltk.word_tokenize(i)
    data_pos = nltk.pos_tag(origin_words)
    words_nnp = [word for word,pos in data_pos if pos in ['NN','NNP','VBG','JJ','JJS','JJR','RB','RBS','RBR']]
    words_nnp = [w for w in words_nnp if not w in stopwords.words('english')]
    nnp_list.append(words_nnp)

# 불필요 제거 어구 리스트 1차원 감소.
nnp_list_1d = list(itertools.chain.from_iterable(nnp_list))
nnp_list_1d[0:20]

# Stopwords 지정.
stop_words = ['game', 'time', 'server', 'getting', 'pubg', 'next', 'this', 'gamei', 'get', 'please', 'battlegrounds', 'stuff', 'region', 'playerunknown', 'bluehole', 'chinese', 'chinaregion', 'ng', 'engine', 'got', 'im', 't', 'xd', 'ram', 'tho', 'asian', 'but', 'busyservers', 'crash', 'no', 'of', 'so', 'me', 'don', 'too', 'for', 'you', 'my', 'gamethe', 'na', 'd', 'devs', 'ur', 'can', 'graphic', 'pc', 'the', 'able', 'a', 'negative', 'due', 's', 'steam', 'regionlockchina', 'stupid', 'complaining', 'in', 'it', 'china', 'computer', 'lag', 'problem', 'gameit', 'are', 'poor', 'itit', 'lack', 'hacking', 'iti', 'anti', 'and', 'gtx', 'to', 'is', 'br', 'ive', 'terrible', 'gear', 'wrong','pubg','lot','access','pc','thing','something','bluehole','im','battleground','battlegrounds','regionlockchina','playerunknown','is','issue','the','dont','lot','gtx','bluehole','alot','end','access','devs','fix','hacker','to','hek','br','beta','hate','my','too','im','one','someone','issue','please','development','na','every','trash','be','br','it','log','im', 'iti', 'pc', 'this', 'to', 'pubg', 'a', 'review', 'lot', 'access', 'year', 'guy', 'steam', 'playerunknown', 'bluehole', 'thing', 'china', 'one', 's', 'na', 'you', 'e', 'don', 'too', 'for', 'cheater', 'battlegrounds', 'gtx', 'dont', 'hardware', 'on', 'devs', 'bugs', 'blue', 'n', 'ban', 'can', 'eu', 'cheating', 'l', 'dude', 'h', 'v', 'kinda', 'your', 'i7', 'would', 'ping', 'be', 'xd', 'ton', 'of', 'f', 'cant', 'but', 'af', 'garbage', 'trash', 'hole', 'its', 'till', 'cheat', 'alpha', 'developer', 'test', 'win', 'gon', 'gameif', 'cpu', 'numba', 'pace', 'choice', 'pls', 'line', 'laggy', 'cod', 'r', 't', 'p', 'minute', 'network', 'ng', 'cs', 'regionlockchina', 'dev', 'man', 'the', 'in', 'me', 'g', 'k', 'gameyou', 'bp', 'gameit', 'is',  'u', 'my', 'are', 'br', 'it', 'hek', 'gamei', 'number', 'alot', 'ram', 'bit', 'just', 'chinaregion', 'd', 'buggy', 'com', 'i5', 'none', 'doot', 'issue', 'hacker', 'hate', 'gamethe', 'gamebut', 'so', 'busyservers', 'bad', 'gpu', 'all', 'rig', 'desync', 'z', 'chinese', 'list', 'let', 'hey', 'games', 'isnt', 'value', 'okay', 'cons', 'do', 'with', 'crap', 'specs', 'no','sometimes', 'instead' ,'yet','also','ok','far','ever','nothing','bug','still','hard','very','always','way','much','lock','however','even','many','away','already','frustration','not','otherwise','little','seriously','reviews','difficult','tl','gay','never','often','gameits','gamein','critical','therethe','lots','everywhere', 'similar','useless','error','early','last']
stop_words_list = np.unique(stop_words)
stop_words_list = stop_words_list.tolist()

# !,? 제거하기.
words_list = []
for i in nnp_list_1d:
#    if(i!='?' and i!='!'):
    text = str(i)
    i = text.lower()
    if(i not in stop_words_list):
        words_list.append(i)
    
print(words_list[0:10])
len(words_list)

# 토픽 모델링 과정.
# WordCount
word_count = collections.Counter(words_list)
result = word_count.most_common(1000)
print(len(word_count))
print(result)

# Countervoctorzier & LAD
c_vetorizer = CountVectorizer(analyzer='word')
count = c_vetorizer.fit_transform(words_list)

model = lda.LDA(n_topics = 8, n_iter = 1000, random_state = 1)
model.fit(count)

# LDA result
topic_vocab = c_vetorizer.get_feature_names()
topic_word = model.topic_word_
n_top_word = 100
dist = []
for i, topic_dist in enumerate(topic_word):
    dist.append(topic_dist)
    topic_words = np.array(topic_vocab)[np.argsort(topic_dist)][:-n_top_word:-1]
    print('Topic', i+1, topic_words)


# 나이브 베이지 과정
# 나이브 베이지안 모델
class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def count_words(self, training_set):
        # 학습데이터는 게임리뷰 본문(doc), 라벨(label)으로 구성
        # 나이브 베이지안 stopwords = 1이상의 값을 가지는 단어와 불필요 상위 단어.
        stop_words = 'game','pubg','play','playing','time','gameplay','very','going','review','lul','experience','u','point','long','reason','pc','highly','community','everything','current','steam','too','this','next','playerunknown','in','all','pan','soon','anything','open','even','still','far','much','early','much','really','lot','ever','new','access','many','t','gmae','first','way','also','way','free','actually','back','someone','dont','something','nothing','LUL','instead','the','thing','a','bluehole','trying','already','almost','not','team','high','amount','away','able','and','always','everyone','year','to','day','guy','is','you','i','maybe','else','na','s','don','so','it','im','cant'
        counts = defaultdict(lambda : [0, 0])
        for doc, label in training_set:
            # 영화리뷰가 text일 때만 카운트
            if self.isNumber(doc) is False:
                # 리뷰를 띄어쓰기 단위로 토크나이징
                words = doc.split()                           
                # 토픽 모델링과 같은 전처리.
                data_pos = nltk.pos_tag(words)
                words_nnp = [word for word,pos in data_pos if pos in ['NN','NNP','VBG','JJ','JJS','JJR','RB','RBS','RBR']]
                words = [w for w in words_nnp if not w in stopwords.words('english')]
                
                for word in words:
                    text = str(word)
                    i = text.lower()
                    if(i not in stop_words):
                        # 라벨이 1이면 0값 지정
                        counts[i][0 if label == 1 else 1] += 1
        return counts
    # 예외 처리
    def isNumber(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    # 모델 결과 반환.
    def word_probabilities(self, counts, total_class0, total_class1, k):
        # 단어의 빈도수를 [단어, p(w|긍정), p(w|부정)] 형태로 반환
        return [(w,
                 (class0 + k) / (total_class0 + 2*k),
                 (class1 + k) / (total_class1 + 2*k))
                for w, (class0, class1) in counts.items()]

    def class0_probability(self, word_probs, doc):
        # input 띄어쓰기 처리
        docwords = doc.split()
        # 초기값은 모두 0으로 처리
        log_prob_if_class0 = log_prob_if_class1 = 0.0

        for word, prob_if_class0, prob_if_class1 in word_probs:
            
            # 만약 리뷰에 word가 나타나면 해당 단어가 나올 log 확률을 더해 줌
            if word in docwords:
                log_prob_if_class0 += math.log(prob_if_class0)
                log_prob_if_class1 += math.log(prob_if_class1)
            # 만약 리뷰에 word가 나타나지 않는다면 해당 단어가 나오지 않을 log 확률을 더해 줌. 
            # 나오지 않을 확률은 log(1-나올 확률)로 계산
            else:
                log_prob_if_class0 += math.log(1.0 - prob_if_class0)
                log_prob_if_class1 += math.log(1.0 - prob_if_class1)

        prob_if_class0 = math.exp(log_prob_if_class0)
        prob_if_class1 = math.exp(log_prob_if_class1)
        return prob_if_class0 / (prob_if_class0 + prob_if_class1)
    # 모델 학습
    def train(self, corpus):
        training_set = corpus

        # calss0 = 긍정리뷰 수 / class1 = 부정리뷰 수 
        num_class0 = 95000
        num_class1 = 95000

        # train
        word_counts = self.count_words(training_set)
        self.word_probs = self.word_probabilities(word_counts, num_class0, num_class1, self.k)
    # 모델 테스트 
    def classify(self, doc):
        return self.class0_probability(self.word_probs, doc)

# 긍정 리뷰 불러오기
trainfile_path=r'E:/unstruct/Merge_data_po.csv'
corpus = read_table(trainfile_path, sep=',', encoding='utf-8')
train_po = corpus[0:95000]
corpus_po = np.array(train_po)
# 부정 리뷰 불러오기
trainfile_path=r'E:/unstruct/Merge_data_ne.csv'
corpus = read_table(trainfile_path, sep=',', encoding='utf-8')
train_ne = corpus[0:95000]
corpus_ne = np.array(train_ne)
# 긍정 부정 리뷰 병합 베이지안 모델 input
nb_data = np.concatenate((corpus_po, corpus_ne))

# 모델 학습
model = NaiveBayesClassifier()
model.train(nb_data)
# 긍정 단어 가중치로 딕셔너리 만들기
dict_p = {}
for i in model.word_probs:
    dict_p[i[0]] = i[1]
# 부정 단어 가중치로 딕셔너리 만들기    
dict_n = {}
for i in model.word_probs:
    dict_n[i[0]] = i[2]
# 각 딕셔너리 값을 내림차순 정렬
sort_dict_p = sorted(dict_p.items(), key=operator.itemgetter(1), reverse=True)
sort_dict_n = sorted(dict_n.items(), key=operator.itemgetter(1), reverse=True)
# 각 리뷰별 상위 단어 추출
for i in sort_dict_p[0:30]:
    print(i[0])
for i in sort_dict_n[0:30]:
    print(i[0])
# 나이브 베이지안 모델 테스트
a = 'Australian servers are here The Good Great graphicsPvp combat is responsive and balancedVariety of weapons attatchments armour and health pickups vehicles tooDevs that listen to the community feedbackThe Bad Typical bugs and glitches you d expect from an early access gamee g menu freezing after game The target area is a bit annoying it s hard to focus on killing when you re too busy trying to make it into the target area and not automatically killed within the time limitHalf of the time i recieve reward points they arent even credited to my accountPurchased the first crate for 700 coins and did not even receive my items Recommendations Maybe introduce other modes andor smaller areas on the mapE g Close quarters All in all this game has great potential and is off to a great startI m excited to see what the devs have in store for us'
model.classify(a)

#감성 사전 구축 과정
# 긍정 토픽
ps_comment = ['good','fun','great','really','better','best','pretty','amazing','awesome','awsome','early','well','goodgame','gg','fantastic','enjoyable','wow','favourite','happy','love','exciting','favorite','hilarious','adrenaline','interesting','excellent']
ps_charctor = ['battleroyale','competetive','real','realistic','addictive','hardcore','strategic','military','faster','pvp','massive']
ps_style = ['battle','fps','royale','gameplay','combat', 'random', 'war','hide','wearing','eating','hunting','hitting','hiding','loot','shooter','aim','aiming','survival','running','picking','kill','killing']
ps_feature = ['sniping','shot','map','bike','squad','inventory','dinner','chicken','scope','box','biggest','miltiple','customization','weapon','winner','clothing','fpp','circle','rating','zone']
ps_graphic = ['air','art','detail','night', 'weather', 'graphics', 'graphic', 'character' ]
ps_othergame = ['battlefield', 'dayz' ,'h1z1', 'h1', 'overwatch', 'csgo', 'cs', 'arma', 'fortnite']
ps_another = ['twitch', 'wadu', 'youtube','streamer' ]
# 부정 토픽
ne_comment = ['bad', 'worst', 'waste', 'unreal', 'problem', 'trash', 'trerrible', 'wrong', 'stupid', 'crash', 'ridiculous', 'garbage', 'refund', 'serious', 'anymore', 'never', 'sad', 'bye', 'boring',  'horrible' ]
ne_envoir = ['server','bug','lag','laggy','optimization','unplayable','matchmaking', 'waiting' ]
ne_another = ['chinese','region','regionlock','lock','regionlockchina','china','cheat','cheating','hack','dev','development','fixing']

# 토픽 모델링 단어에 대해 나이브베이지안 확률값 매핑
ps_comment_dic = {}
for topic in ps_comment:
    for word,value in dict_p.items():
        if(topic == word):
            ps_comment_dic[word] = value
print(len(ps_comment_dic))
ps_charctor_dic = {}
for topic in ps_charctor:
    for word,value in dict_p.items():
        if(topic == word):
            ps_charctor_dic[word] = value
print(len(ps_charctor_dic))
ps_style_dic = {}
for topic in ps_style:
    for word,value in dict_p.items():
        if(topic == word):
            ps_style_dic[word] = value
print(len(ps_style_dic))         
ps_feature_dic = {}
for topic in ps_feature:
    for word,value in dict_p.items():
        if(topic == word):
            ps_feature_dic[word] = value
print(len(ps_feature_dic))
ps_graphic_dic = {}
for topic in ps_graphic:
    for word,value in dict_p.items():
        if(topic == word):
            ps_graphic_dic[word] = value
print(len(ps_graphic_dic))            
ps_othergame_dic = {}
for topic in ps_othergame:
    for word,value in dict_p.items():
        if(topic == word):
            ps_othergame_dic[word] = value
print(len(ps_othergame_dic))
ps_another_dic = {}
for topic in ps_another:
    for word,value in dict_p.items():
        if(topic == word):
            ps_another_dic[word] = value
print(len(ps_another_dic))            
ne_comment_dic = {}
for topic in ne_comment:
    for word,value in dict_n.items():
        if(topic == word):
            ne_comment_dic[word] = value
print(len(ne_comment_dic))            
ne_envoir_dic = {}
for topic in ne_envoir:
    for word,value in dict_n.items():
        if(topic == word):
            ne_envoir_dic[word] = value
print(len(ne_envoir_dic))
ne_another_dic = {}
for topic in ne_another:
    for word,value in dict_n.items():
        if(topic == word):
            ne_another_dic[word] = value
print(len(ne_another_dic))
# 딕셔너리 매핑 확인
print(ps_comment_dic)
print(ps_charctor_dic)
print(ps_style_dic)

# 긍정리뷰 평가
def positvie_review(doc):
    doc = doc
    comment = 0
    charctor = 0
    style = 0
    feature = 0
    graphic = 0
    othergame =0
    antoher = 0
    total = 0

    words = doc.split()                           
    data_pos = nltk.pos_tag(words)
    words_nnp = [word for word,pos in data_pos if pos in ['NN','NNP','VBG','JJ','JJS','JJR','RB','RBS','RBR']]
    words = [w for w in words_nnp if not w in stopwords.words('english')]
    # 각 토픽별 점수 계산
    for word in words:
        for _,i in ps_comment_dic.items():
            if (word == _):
                comment += i
        for _,i in ps_charctor_dic.items():
            if (word == _):
                charctor += i
        for _,i in ps_style_dic.items():
            if (word == _):
                style += i
        for _,i in ps_feature_dic.items():
            if (word == _):
                feature += i
        for _,i in ps_graphic_dic.items():
            if (word == _):
                graphic += i
        for _,i in ps_othergame_dic.items():
            if (word == _):
                othergame += i
        for _,i in ps_othergame_dic.items():
            if (word == _):
                antoher += i

    total = comment + charctor + style + feature + graphic + othergame + antoher
    print('------------------------------------------------------리뷰 감성 분석------------------------------------------------------')
    print('-------------------------------------------------------토픽별 점수--------------------------------------------------------')
    print('게임평  : '+str(comment)+'  | 게임성격 : '+str(charctor)+'  | 게임스타일 : '+str(style)+'  | 게임요소 : '+str(feature)+'  | 그래픽 : '+str(graphic)+'  | 타게임 : '+str(othergame)+'  | 그외 : '+str(antoher))
    print('총 합계 : '+str(total))
    print('--------------------------------------------------------------------------------------------------------------------------')
    
#부정 리뷰 평가
def negative_review(doc):
    doc = doc
    comment = 0
    envoir = 0
    antoher = 0
    total = 0

    words = doc.split()                           
    data_pos = nltk.pos_tag(words)
    words_nnp = [word for word,pos in data_pos if pos in ['NN','NNP','VBG','JJ','JJS','JJR','RB','RBS','RBR']]
    words = [w for w in words_nnp if not w in stopwords.words('english')]
    # 각 토픽별 점수 계산
    for word in words:
        for _,i in ne_comment_dic.items():
            if (word == _):
                comment += i
        for _,i in ne_envoir_dic.items():
            if (word == _):
                envoir += i
        for _,i in ne_another_dic.items():
            if (word == _):
                antoher += i

    total = comment + envoir + antoher
    print('--------------------------------------------리뷰 감성 분석--------------------------------------------')
    print('---------------------------------------------토픽별 점수----------------------------------------------')
    print('게임평  : '+str(comment)+'  | 게임환경 : '+str(envoir)+'  | 그외 : '+str(antoher))
    print('총 합계 : '+str(total))
    print('-------------------------------------------------------------------------------------------------------')
    
# 모델 평가
po = '긍정 리뷰'
ne = '부정 리뷰'
positvie_review(po)
negative_review(ne)
