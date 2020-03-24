import math
import numpy as np
import copy
import time

def read_file_here(name):
	f=open(name,"r")
	data=f.readlines()
	ratings=[]
	for line in data:
		ratings_row=[]
		line=line.split("\t")    #tabs delimiter here
		user=int(line[0])
		ratings_row.append(user)

		movie=int(line[1])
		ratings_row.append(movie)
		score_rate=float(line[2])

		ratings_row.append(score_rate)
		
		ratings.append(ratings_row)

	max_movie=0
	max_user=ratings[len(ratings)-1][0]
	for rate in ratings:
		if rate[1]>max_movie:
			max_movie=rate[1]      #gives maximum movie id

	rating_matrix=np.zeros((max_user, max_movie))
	for rate in ratings:
		rating_matrix[rate[0]-1][rate[1]-1]=rate[2]
	#filling rating matrix with the ratings which are available
	rating_matrix = rating_matrix.transpose()
	return rating_matrix


data_matrix = read_file_here('ml-100k/ua.base')                    #Read data to be trained into the data_matrix

test_matrix = read_file_here('ml-100k/ua.test')          #Read data to be tested into the test_matrix

start=time.time()           # Start time
print("Starting time calculation for prediction... ")

predicted_values = np.zeros((test_matrix.shape[0], test_matrix.shape[1]))     #Matrix created to store predicted values

all_possible_k = []         #List to store all possible values of k

data_matrix = data_matrix.transpose()
for user in range(len(data_matrix)):
	all_possible_k.append(np.count_nonzero(data_matrix[user]))
k = min(all_possible_k)                     #minimum possible k
data_matrix = data_matrix.transpose()

original = np.copy(data_matrix)

movie_magnitude = {}        #dictionary to hold all movies normalised values

for movie in range(len(data_matrix)):
	total_user_ratings=0
	value = 0
	for user in range(len(data_matrix[movie])):
		if data_matrix[movie][user]!=0:
			total_user_ratings+=1
	if sum(data_matrix[movie]) == 0:
		mean = 0
	else:
		mean = float(sum(data_matrix[movie]))/total_user_ratings    #calculate average rating of each movie

	value = 0
	for user in range(len(data_matrix[movie])):
		if data_matrix[movie][user]!=0:
			data_matrix[movie][user] -= mean
		value += ((data_matrix[movie][user]) ** 2)
	value = math.sqrt(value)
	movie_magnitude.update({movie+1:value})                       #store magnitude of each movie vector


square_error = 0
num_ratings = 0

# Predicting movie ratngs for each movieID
for movieID in range(len(data_matrix)):
	if movieID < test_matrix.shape[0]:       #not predicting for movies not in test_matrix
		similarity = {}                      #finding similarity for each all other movies wrt the given movieID
		for movie in range(len(data_matrix)):
			if movie == movieID:
				similarity.update({movie+1: 1})
			elif float(movie_magnitude[movie+1]) == 0 or float(movie_magnitude[movieID+1]) == 0:
				similarity.update({movie+1: 0})
			else:
				product = np.dot(data_matrix[movie], data_matrix[movieID])
				value = np.sum(product)
				value = value / (float(movie_magnitude[movie+1])*float(movie_magnitude[movieID+1]))
				similarity.update({movie+1: value})
		
		for userID in range(len(test_matrix[movieID])):          #for each userID in test ratings for movieID
			if test_matrix[movieID][userID]!=0:                  #predicting for non-zero entries in test matrix
				if original[movieID][userID]==0:                 #predicting for missing entries in original i.e. data to 
				                                                 #be trained 
					
					temp_similarity = copy.deepcopy(similarity)
					k_temp = k

					numerator = 0
					denominator = 0

					while k_temp != 0:
						m = max(temp_similarity.keys(), key=(lambda key: float(temp_similarity[key])))
						if m-1 != movieID and original[m-1][userID] != 0:
							denominator += abs(float(temp_similarity[m]))
							numerator += (float(temp_similarity[m]) * float(original[m-1][userID]))
							k_temp -= 1
						temp_similarity.pop(m)                  #remove similarity values already been used from dictionary

					if denominator == 0:
						pred_value = 0                         #zero for ratings which cannot be determined
					else: 
						pred_value = float(numerator)/denominator
						
					square_error += ((test_matrix[movieID][userID] - pred_value) ** 2)  #calculate difference between actual
					                                                                    # and predicted ratings
					num_ratings += 1

					predicted_values[movieID][userID] = pred_value             #store predicted ratings

# End time for prediction
print("Total time taken for prediction is : %s seconds" %(time.time()-start))

#Root Mean Square Error(RMSE)
rmse = math.sqrt(square_error/num_ratings)
print('RMSE is: ' + str(rmse))

# Spearman coefficient
spearmen_coeff = 1 - ((6*square_error)/((num_ratings ** 3) - num_ratings))
print('Spearman coefficient: ' + str(spearmen_coeff))

# Precision on top N
predicted_values = predicted_values.transpose()
test_matrix = test_matrix.transpose()
hits = 0
total = 0
for user in range(len(predicted_values)):
	t = np.count_nonzero(predicted_values[user])
	if k<t:
		N = k
	else:
		N = t
	total += N
	
	threshold = np.array([j for j in range(len(predicted_values[user])) if predicted_values[user][j] >= 4])
	   #set threshold as movies with ratings equal to or greater than 4 
	topN_movies = threshold.argsort()[-N:][::-1]
	
	for i in topN_movies:
		if predicted_values[user][i] >= test_matrix[user][i]:
			hits += 1 

precision = 100 * float(hits)/total           
predicted_values = predicted_values.transpose()
test_matrix = test_matrix.transpose()

print('Precision on top K: '+str(precision))

print('Collaborative Filtering done')
