import testsets
import evaluation
import re
import time
import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from gensim.models import word2vec
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

start=time.time() #calculate running time
train = pd.read_csv("twitter-training-data.txt",  header=0,names=["usr_id","sentiment","content"],delimiter="\t", quoting=3)#load training data

def lemmatize_all(sent): #using pos_tag to lemmatize words
    word_lem = WordNetLemmatizer()
    word_list=[]
    for word, tag in pos_tag(word_tokenize(sent)):
        if tag.startswith('N'): #
            wordnet_pos=wordnet.NOUN
            word_list.append(word_lem.lemmatize(word, wordnet_pos))
        elif tag.startswith('V'):
             wordnet_pos=wordnet.VERB
             word_list.append(word_lem.lemmatize(word, wordnet_pos))
        elif tag.startswith('J'):
             wordnet_pos=wordnet.ADJ
             word_list.append(word_lem.lemmatize(word, wordnet_pos))
        elif tag.startswith('R'):
             wordnet_pos=wordnet.ADV
             word_list.append(word_lem.lemmatize(word, wordnet_pos))
        else:
            wordnet_pos=wordnet.NOUN
            word_list. append(word_lem.lemmatize(word, wordnet_pos))
    return word_list

def pre_text(text1):#preprocess twitters
    pre_text=[]
    for i in range(len(text1)):
        lemt_text=[]
        words=[]
        text = text1.loc[i,"content"] #read each twitter
        text = re.sub(r'[A-Za-z]+://[^\s]*', 'URLLINK',text)#'URLLINK',text) can turn all URLs into 'URLLINK'
        text = re.sub(r'(@[A-Za-z0-9_]+)','USERMENTION',text)#replace all usermentions
        text = re.sub(r'(#[A-Za-z0-9_]+)','HASHTAG',text)#replace all hashtags
        text = re.sub(r"n\'t", " not", text)# restore all negative abbreviations
        text = re.sub(r"won't", "will not", text)# restore all negative abbreviations
        text = text.lower()
        text = re.sub(r'(.)\1+',r'\1\1',text) #remove elongated words
        text = re.sub(r"[:|;][\-|\^|o|\']?[d|\)|\>|\]|\*]+|=d|lol|:\'\-\)","positive",text) #turn all smileys(like :))))))))) :) :-) :-)) :^) LOL :o) :'-) =D  :-D :] ;> :-* ) into “positive”
        text = re.sub(r"[:|;][\'|\-]?[\(]+|:\'\-\(","negative",text)#turn all frowns(like :( :'( :-( :'-( ) into “negative”
        text = re.sub("[^a-zA-Z]", " ",text)#remove non-alphanumeric characters and one character
        text = re.sub(" \w{1} ", " ",text)
        lemt_text+=(lemmatize_all(text))  # apply lemmatize to every word in twitter   
        lemt_str=" ".join(str(v) for v in lemt_text)   
        words = [w for w in lemt_str.split() if not w in set(stopwords.words("english")) and w !=" "] #remove stop words, because they are meaningless, like will, should, I.
        pre_text.append(words)        
    return pre_text   

sens_train = pre_text(train)
def read_test(testset):#return usr_id for predictions 
    tweetid =[]
    with open(testset, 'r') as fh:
        for line in fh:
            fields = line.split('\t')
            tweetid .append( fields[0])
    return tweetid

#for lexicon
with open(r"opinion-lexicon-English\positive-words.txt","r") as t: # read positive-words.txt and store positive-words in a set
    token_pos=set(t.read().split()) 
with open(r"opinion-lexicon-English\negative-words.txt","r") as t: # read negative-words.txt and store negative-words in a set
    token_neg=set(t.read().split())

def get_lexicon_feature(sentense):# get features besed on lexicon
    lex_feature=[]
    for i in range(len(sentense)):
        num_pos=0
        num_neg=0
        num_neu=0
        for j in range(len(sentense[i])):#count number of positive words and negative woeds
            if sentense[i][j] in token_pos:#if it is positive,number of positive add 1
               num_pos+=1
            elif sentense[i][j] in token_neg:#if it is negative,number of negative add 1
                num_neg+=1
            else:#if it is neither positive nor negative,number of neutral add 1
                num_neu+=1
        lex_feature.append([num_pos,num_neg,num_neu])
    return lex_feature

#for ngram
def get_ngram_sentence(sentense): #produce sentence for ngram
    ngram_sen=[]
    for item in sentense:
        temp_sen=" ".join(str(v) for v in item) #change list to string
        ngram_sen.append(temp_sen)
    return ngram_sen
        
def get_ngram_tfidf(ngram_sen):
    # produce unigrams and bigrams
    vectorizer= CountVectorizer(ngram_range=(1, 2), decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1,max_features = 10000)
    pretext_vec=vectorizer.fit_transform(ngram_sen) 
    tfidf_transformer = TfidfTransformer() #get tf-idf values
    ngram_tfidf = tfidf_transformer.fit_transform(pretext_vec)
    return ngram_tfidf

#for word2vec: the output is not good
#num_features = 300  # Word vector dimensionality
#min_word_count = 10 # Minimum word count
#num_workers = 4     # Number of parallel threads
#context = 10        # Context window size
#downsampling = 1e-3 
#
#def featureVecMethod(words, model, num_features):
#    # Pre-initialising empty numpy array for speed
#    featureVec = np.zeros(num_features,dtype="float32")
#    nwords = 0
#    
#    #Converting Index2Word which is a list to a set for better speed in the execution.
#    index2word_set = set(model.wv.index2word)
#    
#    for word in  words:
#        if word in index2word_set:
#            nwords = nwords + 1
#            featureVec = np.add(featureVec,model[word])
#    
#    # Dividing the result by number of words to get average
#    featureVec = np.divide(featureVec, nwords)
#    return featureVec
#        
#def getAvgFeatureVecs(reviews, model, num_features):
#    counter = 0
#    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
#    for review in reviews:           
#        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
#        counter = counter+1       
#    return reviewFeatureVecs        
#    
for classifier in ['Lexicon_SVM', 'Ngram_tfidf_MultinomialNB', 'Lexicon_MultinomialNB']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'Lexicon_SVM':
        print('Training ' + classifier)
        clf = svm.SVC().fit(get_lexicon_feature(sens_train),train["sentiment"])#train model

    elif classifier == 'Ngram_tfidf_MultinomialNB':
        print('Training ' + classifier)
        ngram_train=get_ngram_sentence(sens_train)
        clf = MultinomialNB().fit(get_ngram_tfidf(ngram_train),train["sentiment"])   #train model
#    elif classifier == 'word2vec':
#        print('Training ' + classifier)
#        model = word2vec.Word2Vec(sens_train,workers=num_workers,size=num_features,min_count=min_word_count,window=context,sample=downsampling)
#        trainDataVecs = getAvgFeatureVecs(sens_train, model, num_features) 
#        model.init_sims(replace=True)
#        clf = svm.SVC().fit(trainDataVecs,train["sentiment"])       

    elif classifier == 'Lexicon_MultinomialNB':
        print('Training ' + classifier)
        clf =  MultinomialNB().fit(get_lexicon_feature(sens_train),train["sentiment"])#train model
        
    for testset in testsets.testsets:
        test = pd.read_csv(testset, header=0,names=["usr_id","sentiment","content"],delimiter="\t", quoting=3)#read test files
        test_id=read_test(testset) #get the user_ids in test file for produce the dictionary of Predictions
        sens_test = pre_text(test) #preprocess test data
        if classifier == 'Lexicon_SVM':        
            test_feature=get_lexicon_feature(sens_test) #produce features for test

        elif classifier == 'Ngram_tfidf_MultinomialNB': 
            ngram_test=get_ngram_sentence(sens_test) 
            test_feature=get_ngram_tfidf(ngram_test) #produce features for test
#        elif classifier == 'word2vec':
#            test_feature = getAvgFeatureVecs(sens_test, model, num_features)
        else:
            test_feature=get_lexicon_feature(sens_test) #produce features for test
            
        test_pred=clf.predict(test_feature) #use trained model and test feature to get predicted sentiments
        print(np.mean(test_pred == test["sentiment"])) #calculate the accuaracy of prediction       
        predictions = dict(zip(test_id,test_pred)) #get the dictionary of Predictions
        evaluation.evaluate(predictions, testset, classifier)# calculate f1
        evaluation.confusion(predictions, testset, classifier) # calculate confusion metrix
#        print(test_pred)
end = time.time() #record end time
print ("running time is",end-start,"second") #print running time 











































