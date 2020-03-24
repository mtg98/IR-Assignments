import json
import copy
import operator

with open("data1.json") as f1:
    data = json.load(f1)
    
sentences_en = []
sentences_fr = []

for i in range(len(data)):
	sentences_en.append((data[i]['en']).strip())

for i in range(len(data)):
	sentences_fr.append((data[i]['fr']).strip())

print(sentences_fr)
#print("\n")
#print(sentences_en)
    
words_fr = []
words_en = []
for i in range(len(data)):
    for index in range(len(data[i]['fr'].split())):
        words_fr.append((data[i]['fr'].split())[index])
        
    for index in range(len(data[i]['en'].split())):
        words_en.append((data[i]['en'].split())[index])
    
words_fr = list(set(words_fr))
words_en = list(set(words_en))

#print(words_fr)
#print("\n")
#print(words_en)
    
initial_count = len(words_en)     #count same for both french and their respective english words
#initial_count= len(words_fr)

t = {}


#Initializing probability of all words and their possible translations
for e in words_en:
	trans_prob = {}
	for f in words_fr:
		trans_prob[f] = (1 / initial_count)
	t[e]=trans_prob

print(t)
print("\n")


list_length = len(sentences_fr)   # same as length of sentences_en list

t_copy = {}                 #dictionary copied at beginning of each while iteration to check convergence of t[e/f] values


# ****** convergence condition check : *******

#def check_convergence():
#	print(t_copy)
#	for e in words_en:
#		for f in words_fr:
#			print(t_copy[e][f])
#			difference = (t[e][f])-(t_copy[e][f])
#			if abs(difference)>0.0001:
#				return 1
#	return 0


#change = 1
#while change != 0 :                           #convergence condition checked here by change variable
#for e in words_en:
#		difference = 0.0
#		for f in words_fr:
#			#print(t_copy)
#			difference += (t[e][f])-(t_copy[e][f])
#		if abs(difference)<0.0000001:
#			change = 0

# ***********************************************



# Em algorithm applied below for IBM model1

i = 0
while i < 1000:
	
	t_copy = t.copy()
	i += 1
	#print(i)
	#print("\n")

	count={}
	total={}
	total_s={}
	
	for e in words_en:
		count_prob = {}
		for f in words_fr:
			count_prob[f] = 0.0
		count[e]=count_prob

	for f in words_fr:
		total[f] = 0.0

	for j in range(list_length):
		
		e_sen = sentences_en[j]
		ewords = e_sen.split()
		#print(ewords)

		f_sen = sentences_fr[j]
		fwords = f_sen.split()

		for e in ewords:
			total_s[e] = 0.0
			#print(fwords)
			for f in fwords:
				#print(f)
				total_s[e] += t[e][f]

		for e in ewords:
			for f in fwords:
				count[e][f] += (t[e][f]/total_s[e])
				total[f] += (t[e][f]/total_s[e])

	for f in words_fr:
		for e in words_en:
			t[e][f] = (count[e][f]/total[f])


print(t)
print("\n")
print("The most relevant Foreign translation for each english word after training data through IBM1 and EM Model :")

#for e in words_en:
#	print(e, " ", "-", " ", max(t[e].items(), key=operator.itemgetter(1))[0])
	
final_dict = {}
for e in words_en:
	final_dict[e] = max(t[e].items(), key=operator.itemgetter(1))[0]

#print(final_dict)
#print("\n")
with open("file.txt", 'w') as file:
	file.write(json.dumps(final_dict))

for i in range(len(data)):
	list_en = sentences_en[i].split()
	list_fr = sentences_fr[i].split()
	print(sentences_en[i])
	print(sentences_fr[i])
	for j in range(len(list_en)):
		word = final_dict[list_en[j]]
		flag=1
		for k in range(len(list_fr)):
			if list_fr[k]==word:
				print("(", j, k, ")")
				flag=0
		if flag==1:
			print("(",j,"NULL",")")



	#print("\n")



