import csv
import string
import pandas as pd
from pyjarowinkler import distance
import nltk.corpus
from nltk.tokenize import word_tokenize
from sklearn import model_selection,naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import json
import os
import io
import googleapiclient.discovery
import urllib.request, json
from collections import defaultdict

class preprocessing:

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def jaro_distance(sentence,dictionary):
        for k in range(len(sentence)):
            wordx = ["a",0]
            for x, w_dic in enumerate(dictionary):
                jaro_dist = distance.get_jaro_distance(sentence[k], w_dic, winkler=True, scaling=0.1)
                if(jaro_dist > wordx[1]):
                    wordx[1] = jaro_dist
                    wordx[0] = w_dic
            sentence[k] = wordx[0]
        return " ". join(sentence)

    def normalization_per_word(word,dictionary):
        Kata_nonformal = {
            1 : {
            "slang" : ["gak","nda","gk","ndak","nd","tdk","no","nope","g"],
            "baku"    : "tidak"
            },
            2 : {
            "slang" : ["makasih","maasih","makasi","thank","thanks"],
            "baku"    : "terima kasih"
            },
            3 : {
            "slang" : ["kntl","ktl"],
            "baku"    : "kontol"
            },
            4 : {
            "slang" : ["dgn"],
            "baku"    : "dengan"
            },
            5 : {
            "slang" : ["dlm"],
            "baku"    : "dalam"
            },
            6 : {
            "slang" : ["boong"],
            "baku"    : "bohong"
            },
            7 : {
            "slang" : ["jd","jdii","jdinya"],
            "baku"    : "jadi"
            },
            8 : {
            "slang" : ["sj","doang"],
            "baku"    : "saja"
            },
            9 : {
            "slang" : ["jgn","tjangan"],
            "baku"    : "jangan"
            },
            10 : {
            "slang" : ["jg","jga"],
            "baku"    : "juga"
            },
            11 : {
            "slang" : ["banget","bet"],
            "baku"    : "sangat"
            },
            12 : {
            "slang" : ["makin"],
            "baku"    : "semakin"
            }
        }
        jaro = ["a",0]
        for x in enumerate(Kata_nonformal):
            for y in enumerate(Kata_nonformal[x[1]]["slang"]):
                if(word == y[1]):
                    return Kata_nonformal[x[1]]["baku"]

        for x, w_dic in enumerate(dictionary):
            jaro_dist = distance.get_jaro_distance(word, w_dic, winkler=True, scaling=0.1)
            if(jaro_dist > jaro[1]):
                jaro[1] = jaro_dist
                jaro[0] = w_dic
        return jaro[0]

    def normalization(sentence,dictionary):
        for k in range(len(sentence)):
            sentence[k] = preprocessing.normalization_per_word(sentence[k],dictionary)
        return " ". join(sentence)

    def stemming_create():
        factory = StemmerFactory()
        return factory.create_stemmer()

    def stopword_removal(text):
        file1 = open("stopword.txt","r")
        stop_words = set(word_tokenize(file1.read()))
        file1.close()
        filtered_text = [w for w in text if not w in stop_words]
        return filtered_text
    

class youtube_mining:
    
    def comment_mining(link, max_result, search_term):
        dict1 = defaultdict(dict)
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
        DEVELOPER_KEY = "AIzaSyDu05uUqdtZAtx3GkRp7pRqq8We-cVm-EI"
        api_service_name = "youtube"
        api_version = "v3"

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey = DEVELOPER_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            maxResults= max_result,
            searchTerms=search_term,
            textFormat="plainText",
            videoId=link
        )
        response = request.execute()
        for x, val in enumerate(response["items"]):
            dict1[x]['content'] = (str(val["snippet"]["topLevelComment"]["snippet"]["textDisplay"]))
            dict1[x]['writer'] = (str(val["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"]))
        return dict1
    
    def video_description(link):
        DEVELOPER_KEY = "AIzaSyDu05uUqdtZAtx3GkRp7pRqq8We-cVm-EI"
        dict1 = defaultdict(dict)       
        link_get = "https://www.googleapis.com/youtube/v3/videos?part=id%2C+snippet&id="+link+"&key=" + DEVELOPER_KEY
        response = urllib.request.urlopen(link_get)
        data = json.loads(response.read())
        for x, val in enumerate(data["items"]):
            dict1[x]['title'] = (str(val["snippet"]["title"]))
            dict1[x]['description'] = (str(val["snippet"]["description"]))
            dict1[x]['thumbnail'] = (str(val["snippet"]["thumbnails"]["medium"]["url"]))           
        return dict1

class classification:

    def svm(X,y,x_test,kernel1,degree1,nilai_c,coef01,):
        SVM = svm.SVC(C=nilai_c, kernel=kernel1, degree=degree1, gamma='auto', coef0=coef01)
        SVM.fit(X,y)
        predictions_NZ = SVM.predict(x_test)
        return predictions_NZ

