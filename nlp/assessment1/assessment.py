# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
import json
import nltk
import time
import math

from collections import Counter
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import trigrams,bigrams

def lemmatize_all(sent): #using pos_tag to lemmatize words
    word_lem = WordNetLemmatizer()
    word_list=[]
    for word, tag in pos_tag(word_tokenize(sent)):
        if tag.startswith('N'):
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
            word_list.append(word_lem.lemmatize(word, wordnet_pos))
    return word_list



ptn1=re.compile(r'[^\s0-9A-Za-z]') # Remove all non-alphanumeric characters except spaces
ptn2=re.compile(r'\b\w[1]\b') # Remove words with only 1 character
ptn3=re.compile(r'\b\d+\b') # Remove numbers that are fully made of digits
ptn4=re.compile(r'[A-Za-z]+://[^\s]*')  # Remove URLs

start = time.time()# record start time

with open(r"positive-words.txt","r") as t: # read positive-words.txt and store positive-words in a set
    token_pos=Counter(t.read().split())

with open(r"negative-words.txt","r") as t: # read negative-words.txt and store negative-words in a set
    token_neg=Counter(t.read().split())



with open(r"signal-news1.jsonl", "r") as f: # open json file

    total_pos=0 #  counter: calculate all positive-words
    total_neg=0 #  counter: calculate all negative-words
    pos_story=0 #  counter: calculate all positive-stories
    neg_story=0 #  counter: calculate all negative-stories
    all_str=[]  #  counter: store words after using lemmatization

    for line in f.readlines(): # using readlines because the file is too large, we cannot read all line in a time
        pos_num=0 #  counter: calculate all positive-words in each story
        neg_num=0 #  counter: calculate all negative-words in each story
        dic=json.loads(line)
        str1=dic['content']  # read the text in content
        str1=str1.lower()
        str1 = re.sub(ptn4,"",str1) # replace URLs with ""
        str1 = re.sub(ptn1,"",str1) # replace all non-alphanumeric characters except spaces with ""
        str1 = re.sub(ptn3,"",str1) # replace numbers that are fully made of digits with ""
        str1 = re.sub(ptn2,"",str1) # replace words with only 1 character with ""
        lemt_list=lemmatize_all(str1) # lemt_list:store a new story after text preprocessing and lemmatization
        all_str+=lemmatize_all(str1)  # all_str:store all new story after text preprocessing and lemmatization
        for item in lemt_list: # record positive-words numbers and negative-words numbers in each story
            pos_num+=token_pos[item]
            neg_num+=token_neg[item]
#        print(pos_num,neg_num)
        if pos_num>neg_num:
            pos_story+=1 # if a story with more positive than negative words, positive story number +1 ; vice versa
        elif pos_num<neg_num:
            neg_story+=1
        total_pos+=pos_num  # total positive-words add positive-words in each stroy every time
        total_neg+=neg_num  # total neagtive-words add negative-words in each stroy every time
    print("this corpus contains" ,pos_story,"positive stories")
    print("this corpus contains" ,neg_story,"negative stories")
    print("this corpus contains" ,total_pos,"positive words")
    print("this corpus contains" ,total_neg,"negative words")

str3=" ".join(all_str) # list to str   words_tokenize needs string
n=len(all_str) # n is number of tokens
v=len(set(all_str)) # v is vocabulary size
print("This corpus contains",n,"tokens and",v,"vocabularies.")
tokens=nltk.word_tokenize(str3)
finder=nltk.collocations.TrigramCollocationFinder.from_words(tokens) #discovering trinary phrases and sorting them, typically building a searcher using the function from_words()
trigram_measures=nltk.collocations.TrigramAssocMeasures()
top_25=finder.nbest(trigram_measures.pmi, 25) #Using point mutual information to calculate the score of each n-element phrase
print("top 25 trigrams: ",top_25)

with open(r"signal-news1.jsonl","r") as f:
    train_str=[] # store words on the first 16000 rows of corpus
    for line in f.readlines()[:16000]:
        train_dic=json.loads(line)
        train_str1=train_dic['content']
        train_str1=train_str1.lower()
        train_str1 = re.sub(ptn4,"",train_str1) # preprocessing train sentences
        train_str1 = re.sub(ptn1,"",train_str1)
        train_str1 = re.sub(ptn3,"",train_str1)
        train_str1 = re.sub(ptn2,"",train_str1)
        train_str += train_str1.split()

train_v=len(set(train_str)) # train_v is vocabulary size on the first 16000 rows
train_str=" ".join(train_str)
tokens=nltk.word_tokenize(train_str)

trigram_list=list(trigrams(tokens)) #using list store trigrams on the first 16000 rows
trigram_count=Counter(trigram_list) # convert trigrams to Counter in order to speed up the running rate
bigram_list=list(bigrams(tokens)) #using list store bigrams on the first 16000 rows
bigram_count=Counter((bigram_list)) # convert bigrams to Counter in order to speed up the running rate
new_word=['is','this'] # store whole sentences in new_word
for i in range(8):  # loop 8 times to find 8 new words
    new_tri=[] #store matched trigram
    for item in trigram_list: #find matched trigram
        if item[0:2]==(new_word[i],new_word[i+1]):
            new_tri.append(item)
    t= Counter(new_tri).most_common(1)[0] # find the most frequent matched trigram
    nword=t[0][2] # read the third element in the most frequent matched trigram
    new_word.append(nword) # add the third element in the most frequent matched trigram into list
    i+=1
print(new_word)


with open(r"signal-news1.jsonl", "r") as f:
    test_str=[] # store words after the first 16000 rows of corpus
    test_trigram_list=[] # store trigrams after the first 16000 rows of corpus
    for line in f.readlines()[16000:]: #
        test_dic=json.loads(line)
        str4=test_dic['content']
        str4=str4.lower()
        for item in nltk.sent_tokenize(str4):
            str4 = re.sub(ptn4,"",item)
            str4 = re.sub(ptn1,"",str4)
            str4 = re.sub(ptn3,"",str4)
            str4 = re.sub(ptn2,"",str4)
            test_str += str4.split()
            if len(str4.split())<3: # ignore sentences which are less than 3 words
                continue
            test_tokens=nltk.wordpunct_tokenize(str4)
            test_trigram_list+=list(trigrams(test_tokens)) # add trigrams after the first 16000 rows of corpus

#test_bigram_list=list(bigrams(test_tokens))
p=0 #using for perplexity computation
test_len=len(test_trigram_list) # the number of trigrams after the first 16000 rows of corpus
for item in test_trigram_list:
    if item in trigram_count and item[0:2] in bigram_count: # calculate the occurrences of trigrams and corresponding bigrams
        p=p+math.log((trigram_count[item]+1)/(bigram_count[item[0:2]]+train_v),2)
    else:
        p=p+math.log((1/train_v),2)

plex=pow(2,-p/test_len) #perplexity computation
print(plex)

end = time.time() #record end time
print ("running time is",end-start,"second") #print running time
