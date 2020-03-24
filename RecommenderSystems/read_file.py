
import math
import numpy as np
from numpy import linalg


def read_file(name):
	f=open(name,"r")
	reader=f.readlines()
	ratings=[]
	count = 0
	for line in reader:
		ratings_row=[]
		line=line.split()
		#if count==0:
		#	count=count+1
		#	continue

		user=int(float(line[0]))
		ratings_row.append(user)

		movie=int(float(line[1]))
		ratings_row.append(movie)

		rating=float(line[2])
		ratings_row.append(rating)

		#print(ratings_row)
		ratings.append(ratings_row)
		count +=1

	print(count)
	max_movie=0
	max_user=int(ratings[len(ratings)-1][0])        #checking maximum userID
	for rate in ratings:
		if(rate[1]>max_movie):
			max_movie=int(rate[1])        			#checking  maximum movieID

	rating_matrix=np.zeros((max_user, max_movie))
	for rate in ratings:
		rating_matrix[rate[0]-1][rate[1]-1]=rate[2]    #ratings matrix filled with ratings
	
	for i in range(len(rating_matrix)):
		sum=0
		count=0
		for j in range(len(rating_matrix[i])):
			sum=sum+rating_matrix[i][j]
			count=count+1.0
		avg=sum/count
		for k in range(len(rating_matrix[i])):
			rating_matrix[i][k]=rating_matrix[i][k]-avg
	return rating_matrix
