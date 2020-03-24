import math
import timeit
import numpy as np
from numpy import linalg
from svd_decompose import svd_decompose, eigen
from read_file import read_file


matrix=read_file('ratings.txt')

start=timeit.default_timer()
print("Starting SVD based recommender system evaluation:")

U,sigma,V=svd_decompose(matrix)

final_matrix=(np.dot(U,np.dot(sigma,V)))             #multiply to get final matrix

for i in range(len(final_matrix)):
	for j in range(len(final_matrix[i])):
		final_matrix[i][j]=round(final_matrix[i][j],2)      #final matrix values rounded

print("\nU matrix:")
print(U)

print("\nsigma matrix:")
print(sigma)

print("\nV matrix:")
print(V)

print("\nMultiplication of U,sigma,V:")
print(final_matrix)

print("\nTotalTime taken for prediction:")
stop=timeit.default_timer()
print("%s seconds" %(stop-start))

#Root Mean Square Error
rmse = 0
count = 0
for i in range(0,len(matrix)):
	for j in range(0,len(matrix[i])):
			rmse = rmse+(matrix[i][j]-final_matrix[i][j])**2
			count = count+1
rmse=math.sqrt(rmse/count)

print("\nThe RMSE error is: ")
print(rmse)


#Spearmen coefficient
sum_sp=0
count=0
for i in range(0,len(matrix)):
	for j in range(0,len(matrix[i])):
		sum_sp=sum_sp+(matrix[i][j]-final_matrix[i][j])**2
		count = count + 1

spearmen_coeff = 1-float((6*sum_sp/((count**3)-count)))

print("\nThe Spearman Rank correlation is: ")
print(spearmen_coeff)


#Precision on top K
k=0
trace=sigma.trace()
sum_new=0
sum_old=trace
i=0
while(sum_new<10*sum_old):
	sum_new=sum_new+sigma[i][i]
	sum_old=sum_old-sigma[i][i]
	k=k+1
	i=i+1
	if(i>len(sigma)-1):
		break

print("\nPrecision on top ", k, " values.")
while(k):
	x=len(sigma)-1
	sigma=np.delete(sigma,x,0)
	sigma=np.delete(sigma,x,1)
	U=np.delete(U,len(U[0])-1,1)
	V=np.delete(V,len(V.T[0])-1,0)
	k=k-1

k_matrix=(np.dot(U,np.dot(sigma,V)))
for i in range(len(k_matrix)):
	for j in range(len(k_matrix[i])):
		k_matrix[i][j]=round(k_matrix[i][j],2)

match=0.00
count=0.00
for i in range(0,len(matrix)):
	for j in range(0,len(matrix[i])):
		count=count+1
		old_value=int(round(matrix[i][j]))
		new_value=int(round(k_matrix[i][j]))
		if (old_value==new_value):
			match=match+1

precision=(match*100)/count

print(precision)

