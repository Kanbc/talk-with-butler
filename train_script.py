import numpy as np
import pandas as pd
import re
from scipy import sparse
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# ----- Data ------
def topicData():
    topic = pd.read_csv('training_data/topic.csv',header=None).values.T[0]
    return topic

def eventData():
    event = pd.read_csv('training_data/event.csv',header=None).values.T[0]
    return event

def callData():
    call = pd.read_csv('training_data/call.csv',header=None).values.T[0]
    return call

def waterData():
    water = pd.read_csv('training_data/water.csv',header=None).values.T[0]
    return water

def visitorData():
    visitor = pd.read_csv('training_data/visitor.csv',header=None).values.T[0]
    return visitor

def packageData():
    package = pd.read_csv('training_data/package.csv',header=None).values.T[0]
    return package

# ----- English vocab list -----
def getVocabList():
    vocabs = pd.read_csv('vocab2.csv')
    vocabs['number'] = vocabs.index + 1
    vocabs['word'] = vocabs['0']
    del vocabs['0']
    return vocabs

# --- Process text to feature vector ---
def processText(email_contents):
    vocabList = getVocabList()
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

def textFeatures(word_indices):
    vocabList = getVocabList()
    features = vocabList['number'].astype(int).isin(word_indices) + 0
    return np.array(features)

def getInitialFeatureVectoc():
    topic = topicData()
    event = eventData()
    call = callData()
    water = waterData()
    visitor = visitorData()
    package = packageData()
    # Topic
    topic_features = np.zeros(len(getVocabList()))
    for i in range(len(topic)):
        topic_features = topic_features + textFeatures(processText(topic[i]))

    # Event
    event_features = np.zeros(len(getVocabList()))
    for i in range(len(event)):
        event_features = event_features + textFeatures(processText(event[i]))
    
    # Call     
    call_features = np.zeros(len(getVocabList()))
    for i in range(len(call)):
        call_features = call_features + textFeatures(processText(call[i]))
    
    # Water
    water_features = np.zeros(len(getVocabList()))
    for i in range(len(water)):
        water_features = water_features + textFeatures(processText(water[i]))
    
    # Visitor
    visitor_features = np.zeros(len(getVocabList()))
    for i in range(len(visitor)):
        visitor_features = visitor_features + textFeatures(processText(visitor[i]))
    
    # Package
    package_features = np.zeros(len(getVocabList()))
    for i in range(len(package)):
        package_features = package_features + textFeatures(processText(package[i]))
    
    topic_features = (topic_features >= 1).astype(int)
    event_features = (event_features >= 1).astype(int)
    call_features = (call_features >= 1).astype(int)
    water_features = (water_features >= 1).astype(int)
    visitor_features = (visitor_features >= 1).astype(int)
    package_features = (package_features >= 1).astype(int)
    
    return np.array([topic_features, event_features, call_features, water_features, visitor_features, package_features])

def createVocabOpt():
    vocabs = pd.read_csv('vocab2.csv')
    
    topic = topicData()
    event = eventData()
    call = callData()
    water = waterData()
    visitor = visitorData()
    package = packageData()

    topic_features = np.zeros(len(vocabs))
    for i in range(len(topic)):
        topic_features = topic_features + textFeatures(processText(topic[i]))

    event_features = np.zeros(len(vocabs))
    for i in range(len(event)):
        event_features = event_features + textFeatures(processText(event[i]))
    
    call_features = np.zeros(len(vocabs))
    for i in range(len(call)):
        call_features = call_features + textFeatures(processText(call[i]))
    
    water_features = np.zeros(len(vocabs))
    for i in range(len(water)):
        water_features = water_features + textFeatures(processText(water[i]))
    
    visitor_features = np.zeros(len(vocabs))
    for i in range(len(visitor)):
        visitor_features = visitor_features + textFeatures(processText(visitor[i]))
    
    package_features = np.zeros(len(vocabs))
    for i in range(len(package)):
        package_features = package_features + textFeatures(processText(package[i]))
    
    all_words = topic_features + event_features + call_features + water_features + visitor_features + package_features
    vocabs = vocabs[all_words >= 1]
    vocabs['word'] = vocabs['0']
    del vocabs['0']
    
    vocabs.to_csv('vocab_opt.csv',index=False)
    
    return "Create Done!"

def trainingML():
    createVocabOpt()
    return "Training Done!"