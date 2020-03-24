from nltk.translate.phrase_based import phrase_extraction
import json

with open("data1.json") as f1:
    data = json.load(f1)
    
sentences_en = []
sentences_fr = []

for i in range(len(data)):
    sentences_en.append((data[i]['en']).strip())

for i in range(len(data)):
    sentences_fr.append((data[i]['fr']).strip())


length_dataset = len(sentences_en)      #same as sentences_fr

with open("file.txt", 'r') as file:
    final_dict = json.load(file)

count_f = {}
fr_phrases = []
en_phrases = []

t = {}

def phrase_ext():
    phrase_list = []
    for i in range(length_dataset):
        alignment_list = []
        list_fr = sentences_fr[i].split()
        list_en = sentences_en[i].split()
        for j in range(len(list_en)):
            alignment_tuple =()
            word = final_dict[list_en[j]]
            for k in range(len(list_fr)):
                if list_fr[k]== word:
                    alignment_tuple =  (j, k)
                    alignment_list.append(alignment_tuple)

        #print(alignment_list)
        phrases = phrase_extraction(sentences_en[i], sentences_fr[i], alignment_list)
        #print(phrases)
        phrases = list(sorted(phrases))
        for l in range(len(phrases)):
            #print(phrases[l])
            l_phrases = list(phrases[l])
            phrase_list.append(l_phrases)
            
    return phrase_list


phrase_list = phrase_ext()
#print(len(phrase_list))

for i in range(len(phrase_list)):
    #print(phrase_list[i])
    l_phrases = phrase_list[i]
    fr_phrases.append(l_phrases[3])
    en_phrases.append(l_phrases[2])

for e in en_phrases:
    trans_prob = {}
    for f in fr_phrases:
        trans_prob[f] = 0
    t[e]=trans_prob

for i in range(len(phrase_list)):
    l_phrases = phrase_list[i]
    t[l_phrases[2]][l_phrases[3]] += 1

#print(t)

for f in fr_phrases:
    count_f[f] = fr_phrases.count(f)

#print(count_f)
check = {}
for key in en_phrases:
    check[key]=0

for i in range(len(phrase_list)):
    l_phrases = phrase_list[i]
    #print(l_phrases[2],"   ",l_phrases[3])
    #print("t:",t[l_phrases[2]][l_phrases[3]])
    #print("c:",count_f[l_phrases[3]])
    if check[l_phrases[2]] ==0:
        t[l_phrases[2]][l_phrases[3]] = t[l_phrases[2]][l_phrases[3]]/ count_f[l_phrases[3]]
        check[l_phrases[2]]=1

    #print(t[l_phrases[2]][l_phrases[3]])
    
#print(t)

for i in range(len(phrase_list)):
    l_phrases = phrase_list[i]
    print("(", l_phrases[2],",",l_phrases[3], ")", "-", t[l_phrases[2]][l_phrases[3]])

#print(t)






        




                

    





    

    