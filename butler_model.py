import numpy as np
import pandas as pd
import re
from scipy import sparse
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from train_script import topicData
from train_script import eventData
from train_script import callData
from train_script import waterData
from train_script import visitorData
from train_script import packageData
from numpy import linalg as LA
from flask.json import jsonify

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
    ps = PorterStemmer()
    words = [ps.stem(word) for word in email_contents.split(" ") if len(word) > 0]
    
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
    package = packageData()
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

    # Package
    package_features = np.zeros(len(getVocabListOpt()))
    for i in range(len(package)):
        package_features = package_features + textFeaturesOpt(processTextOpt(package[i]))
    
    topic_features = topic_features/LA.norm(topic_features)
    event_features = event_features/LA.norm(event_features)
    call_features = call_features/LA.norm(call_features)
    water_features = water_features/LA.norm(water_features)
    visitor_features = visitor_features/LA.norm(visitor_features)
    package_features = package_features/LA.norm(package_features)
    
    return np.array([topic_features, event_features, call_features, water_features, visitor_features, package_features])

def menu_name(position):
    if(position == 0):
        return "topic"
    elif(position == 1):
        return "event"
    elif(position == 2):
        return "call"
    elif(position == 3):
        return "water"
    elif(position == 4):
        return "visitor"
    elif(position == 5):
        return "package"
    else:
        return "other"

def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1            
            else:
                m2 = x
    return m2 if count >= 2 else None


def most_similarity(initial_features,text_feature):
    A = np.vstack((initial_features,text_feature))
    A_sparse = sparse.csr_matrix(A)
    similarities = cosine_similarity(A_sparse)
    text_vs_initial = similarities[similarities.shape[0]-1,0:similarities.shape[1]-1]

    first_prob = np.max(text_vs_initial)
    second_prob = second_largest(text_vs_initial)
    if(first_prob < 0.3):
        return jsonify({'menu': ["other"], 'score': { 'topic': text_vs_initial[0],'event': text_vs_initial[1], 'call': text_vs_initial[2], 'water': text_vs_initial[3], 'visitor': text_vs_initial[4], 'package': text_vs_initial[5]} })

    first_position = np.argmax(text_vs_initial)
    second_position = np.where(text_vs_initial==second_prob)[0][0]
    if(first_prob - second_prob < 0.2):
        return jsonify({'menu': [menu_name(first_position),menu_name(second_position)], 'score': { 'topic': text_vs_initial[0],'event': text_vs_initial[1], 'call': text_vs_initial[2], 'water': text_vs_initial[3], 'visitor': text_vs_initial[4], 'package': text_vs_initial[5]} })
    else:
        return jsonify({'menu': [menu_name(first_position)], 'score': { 'topic': text_vs_initial[0],'event': text_vs_initial[1], 'call': text_vs_initial[2], 'water': text_vs_initial[3], 'visitor': text_vs_initial[4], 'package': text_vs_initial[5]} })

def butler_menu(text):
    text_vector = textFeaturesOpt(processTextOpt(text))
    if LA.norm(text_vector) == 0:
        text_unit_vector = text_vector
    else:
        text_unit_vector = text_vector/LA.norm(text_vector)
    answer = most_similarity(getInitialFeatureVectorOpt(),text_unit_vector)
    return answer
