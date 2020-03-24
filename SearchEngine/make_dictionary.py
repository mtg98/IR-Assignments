# -*- coding: utf-8 -*-
import pyPdf
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import unicodedata
import os
import math 
import pickle


#for stemming, lemmatization and removing unicode character from extracted words
def word_pro(w):
    ps=PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    w=w.lower()
    w=ps.stem(w)
    w=lemmatizer.lemmatize(w)
    w=unicodedata.normalize('NFKD',unicode(w)).encode('ascii','ignore')
    return w

#for calculating idf of each extracted word
def idf_cal(word,n_doc):
    idf=float(math.log(float(count)/float(n_doc)))
            
    return idf

top = os.getcwd()+"\IR_Data"
doc_words=[]
dictio={}
count=0
keywords=[]

#accessing the names of all documents and storing them in a list
pdfList = []
dirs=[]
for root,  dirs , files in os.walk(top, topdown=False):
   for fileName in files:
     pdfList.append(os.path.join(root, fileName))
         
for file in dirs:
    files.append(str(file))
    
    
for item in pdfList:
    content = ""
    # Load PDF into pyPDF
    pdf = pyPdf.PdfFileReader(open(item, "rb"))
    # Iterate pages
    for i in range(0, pdf.getNumPages()):
        # Extract text from page and add to content
        content += pdf.getPage(i).extractText() + "\n"
        # Collapse whitespace
    content = " ".join(content.replace(u"\xa0", " ").strip().split())
    content.encode("ascii", "ignore")
    tokens = word_tokenize(content)
    stop_words=stopwords.words('english')
    punctuations = ['(',')',';',':','[',']',',','.','?','-','...',"'",'"','.',' ']
    ext_word = [word for word in tokens if not word in stop_words and not word in punctuations]
    
    doc_words.append([])
    
    for w in ext_word:
        w=word_pro(w)
        if(not all(c.isalpha() for c in w)):
            continue
        duplicate=0
        for k in range(len(keywords)):
            if w==keywords[k]:
                duplicate = 1
                break
        if duplicate == 0:
           keywords.append(w)
           dictio[w]={}
        doc_words[count].append(w)
        if count not in dictio[w]:
            dictio[w][count]=1
        else:
            dictio[w][count]=dictio[w][count]+1
    print "dictio %d appended" %count 
    count=count+1
    
    
print doc_words[0]

with open('pdflist.pickle', 'wb') as handle:
    pickle.dump(pdfList, handle , protocol=pickle.HIGHEST_PROTOCOL)
    
with open('keywords.pickle', 'wb') as handle:
    pickle.dump(keywords, handle , protocol=pickle.HIGHEST_PROTOCOL)
    
with open('doc_words.pickle', 'wb') as handle:
    pickle.dump(doc_words, handle, protocol=pickle.HIGHEST_PROTOCOL)

print "doc_words made"

# for j in range(len(files)):
    # for word in keywords:
        # for doc_word in doc_words[j]:
            # if word==doc_word:
                # if word not in dictio:
                    # dictio[word]={}
                    # dictio[word][j]=1
                # else:
                    # if j not in dictio[word]:
                        # dictio[word][j]=1
                    # else:
                        # dictio[word][j]=dictio[word][j]+1
    # print "File %d made" %j
                        
                        
for word in dictio:
    for ID in dictio[word]:
        dictio[word][ID]=float(dictio[word][ID])/float(len(doc_words[ID]))                           

with open('dictio.pickle', 'wb') as handle:
    pickle.dump(dictio, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print dictio

idf={}

for word in dictio:
    n_doc=len(dictio[word])
    idf[word]=idf_cal(word,n_doc)

with open('idf.pickle', 'wb') as handle:
    pickle.dump(idf, handle, protocol=pickle.HIGHEST_PROTOCOL)


                       