# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import stopwords
import unicodedata
import os
import math 
import pickle
#import dictio.py
#from hello.py import query 

count=0
dictio={}
doc_words=[]
pdfList = []
ranked_doc={}
    
def word_pro(w):          #For stemming, lemmatization of the words 
    ps=PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    w=w.lower()
    w=ps.stem(w)
    w=lemmatizer.lemmatize(w)
    w=unicodedata.normalize('NFKD',unicode(w)).encode('ascii','ignore') #to remove the unicode appearing in front of every stemmed word
    return w
    

#for proceesing of query and generating the list of appropriate documents                
def query_pro(query): 
    idf={}
    pdfList = []
    #opening the dictionary of words produced from documents
    with open('dictio.pickle','rb') as handle1: 
        dictio=pickle.load(handle1)
    
    #opening idf score of each word in the dictionary
    with open('idf.pickle','rb') as handle2:
        idf=pickle.load(handle2)
    
    #opening the list of all words present in documents    
    with open('doc_words.pickle','rb') as handle2:
        doc_words=pickle.load(handle2)
        
    with open('pdflist.pickle','rb') as handle2:
        pdfList=pickle.load(handle2)
        
    #tokenizing query
    q_tokens=word_tokenize(query)
    
    d_tf={}
    q_tf={}
    
    score={}
    doc_ID=[]
    q_words=[]
    #processing words in query and creating of list of these words
    for token in q_tokens:
        w=word_pro(token)
        q_words.append(w)
    #calculating the term-frequency of words in a query
    for word in q_words:
        q_tf[word]=float(float(q_words.count(word))/float(len(q_words)))
        
      
    
    for word in q_words:
        d_tf[word]={}
        #checking whether the word is present in dictionary
        for dic_w in dictio:
            if dic_w==word:
                # for creating a list of document IDs whose words are present in query
                for ID in dictio[dic_w]:
                    flag=0
                    for id2 in doc_ID:
                        if ID==id2:
                            flag=1
                    if flag==0:
                        doc_ID.append(ID)
                    #for calculating term frequency of words in query w.r.t document 
                    d_tf[word][ID]=float(math.log(1+dictio[word][ID]))
        if not word in dictio:
            for ID in doc_ID:
                idf[word]=0
                d_tf[word][ID]=0
                
                    
   
    
    for word in d_tf:
        for ID in doc_ID:
            if(not ID in d_tf[word]):
                d_tf[word][ID]=0
                
     
    sq_len={}
    nor_len={}
    
    #for calculating normalised length of each document 
    for ID in doc_ID:
        sq_len[ID]=0
        nor_len[ID]=0
        for word in q_words:
            sq_len[ID]=float(sq_len[ID]+d_tf[word][ID]**2)
                    
        nor_len[ID]=math.sqrt(sq_len[ID])
                    
                   
    
    #scoring the documents
    for ID in doc_ID:
        score[ID]=0.0
        for word in q_words:
            if(nor_len[ID]!=0.000):
                score[ID]+=float((q_tf[word]*idf[word]*d_tf[word][ID])/nor_len[ID])
                
    final_doc=[]
    
    no_word=0
    
    for ID in doc_ID:
        if score[ID]==0.0:
            no_word=no_word+1;
      
    
    if not doc_ID:
        return 0
    if no_word==len(q_words):
         return 0
    
    score_sorted = sorted(score, key=score.get, reverse=True)
    
    #for creating a list of appropriate documents and sorting them in descending order according to their score
    for r in score_sorted:
        final_doc.append(pdfList[r])
        
    #for creating list of top ten documents 
    if len(final_doc)<10:
        for j in range(len(final_doc)):
            ranked_doc[j+1]=final_doc[j]
    else:
        for j in range(0,10):
            ranked_doc[j+1]=final_doc[j]
            
    
    return ranked_doc
   


                    
    
      
