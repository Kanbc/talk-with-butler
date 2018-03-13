import numpy as np
import numpy as np
import pandas as pd
import numpy as np
import re
from scipy import sparse
from stemming.porter2 import stem
from sklearn.metrics.pairwise import cosine_similarity
from train_script import topicData
from train_script import eventData
from train_script import callData
from train_script import waterData
from train_script import visitorData

def getVocabListOpt():
    vocabs = pd.read_csv('vocab_opt.csv')
    vocabs['number'] = vocabs.index + 1
    return vocabs

def processTextOpt(email_contents):
    vocabList = getVocabListOpt()
    
    email_contents = email_contents.lower()

    strip_all_html = re.compile('[>,<,<*>]') 
    email_contents = re.sub(strip_all_html, '', email_contents)
    strip_all_html2 = re.compile('\s') # \s is equivalent to the class [ \t\n\r\f\v].
    email_contents = re.sub(strip_all_html2, ' ', email_contents)
    
    hundle_number = re.compile('\d+')
    email_contents = re.sub(hundle_number, 'number', email_contents)

    hundle_url = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    email_contents = re.sub(hundle_url, 'httpaddr', email_contents)

    hundle_email = re.compile('[\w\.-]+@[\w\.-]+')
    email_contents = re.sub(hundle_email, 'emailaddr', email_contents)

    hundle_dollar = re.compile('[$]+')
    email_contents = re.sub(hundle_dollar, 'dollar', email_contents)
    
    non_cha_alp = re.compile("[^a-zA-Z0-9]+")
    email_contents = re.sub(non_cha_alp, ' ', email_contents)
    
    words = [stem(word) for word in email_contents.split(" ") if len(word) > 0]
    
    word_indices = []
    for w in words:
        match = sum(vocabList['word'] == w)
        if(match>0):
            word_indices.append(vocabList.loc[(vocabList['word'] == w),'number'].astype(int).values[0])
    
    return word_indices

def textFeaturesOpt(word_indices):
    vocabList = getVocabListOpt()
    features = vocabList['number'].astype(int).isin(word_indices) + 0
    return np.array(features)

def getInitialFeatureVectorOpt():
    topic = topicData()
    event = eventData()
    call = callData()
    water = waterData()
    visitor = visitorData()
    # Topic
    topic_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(topic)):
        topic_features = topic_features + textFeaturesOpt(processTextOpt(topic[i]))

    # Event
    event_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(event)):
        event_features = event_features + textFeaturesOpt(processTextOpt(event[i]))
    
    # Call     
    call_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(call)):
        call_features = call_features + textFeaturesOpt(processTextOpt(call[i]))
    
    # Water
    water_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(water)):
        water_features = water_features + textFeaturesOpt(processTextOpt(water[i]))
    
    # Visitor
    visitor_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(visitor)):
        visitor_features = visitor_features + textFeaturesOpt(processTextOpt(visitor[i]))
    
    topic_features = (topic_features >= 1).astype(int)
    event_features = (event_features >= 1).astype(int)
    call_features = (call_features >= 1).astype(int)
    water_features = (water_features >= 1).astype(int)
    visitor_features = (visitor_features >= 1).astype(int)
    
    return np.array([topic_features, event_features, call_features, water_features, visitor_features])

def most_similarity(initial_features,text_feature):
    A = np.vstack((initial_features,text_feature))
    A_sparse = sparse.csr_matrix(A)
    similarities = cosine_similarity(A_sparse)
    text_vs_initial = similarities[similarities.shape[0]-1,0:similarities.shape[1]-1]
    prob_of_menu = np.max(text_vs_initial)
    menu = np.argmax(text_vs_initial)
    if(prob_of_menu < 0.3):
        return "other"
    else:
        if(menu == 0):
            return "topic"
        elif(menu == 1):
            return "event"
        elif(menu == 2):
            return "call"
        elif(menu == 3):
            return "water"
        else:
            return "visitor"

def butler_menu(text):
    answer = most_similarity(getInitialFeatureVectorOpt(),textFeaturesOpt(processTextOpt(text)))
    return answer
