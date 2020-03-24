import nltk
from nltk.translate import AlignedSent
from nltk.translate import ibm1
import json
import operator

with open("data1.json") as f1:
    data = json.load(f1)
    
sentences_en = []
sentences_fr = []

for i in range(len(data)):
	sentences_en.append(data[i]['en'].split())


for i in range(len(data)):
	sentences_fr.append((data[i]['fr']).split())


print(sentences_fr)
print("\n")
print(sentences_en)

words_fr = []
words_en = []
for i in range(len(data)):
    for index in range(len(data[i]['fr'].split())):
        words_fr.append((data[i]['fr'].split())[index])
        
    for index in range(len(data[i]['en'].split())):
        words_en.append((data[i]['en'].split())[index])
    
words_fr = list(set(words_fr))
words_en = list(set(words_en))

length = len(sentences_en)      #same as length of sentences_fr

bitext = []
for i in range(length):
	bitext.append(AlignedSent(sentences_fr[i], sentences_en[i]))

ibm1 = ibm1.IBMModel1(bitext, 100)

#print(ibm1.translation_table)

#for f in words_fr:
#	print(f, " ", "-", " ", max(ibm1.translation_table[f].items(), key=operator.itemgetter(1))[0])

final_dict = {}
for f in words_fr:
	final_dict[f] = max(ibm1.translation_table[f].items(), key=operator.itemgetter(1))[0]

for i in range(len(data)):
	print(sentences_en[i])
	print(sentences_fr[i])
	for j in range(len(sentences_fr[i])):
		word = final_dict[(sentences_fr[i])[j]]
		flag=1
		for k in range(len(sentences_en[i])):
			if (sentences_en[i])[k]==word:
				print("(", j, k, ")")
				flag=0
		if flag==1:
			print("(",j,"NULL",")")