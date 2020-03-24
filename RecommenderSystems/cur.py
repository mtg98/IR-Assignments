import timeit
import math
import numpy as np
from numpy import linalg
from svd_decompose import svd_decompose, eigen
from read_file import read_file

#Here duplications may or may not occur in the selection of rows and columns for matrix

#function to select random rows and columns for CUR decomposition 
def select(matrix):
	matrix=matrix.tolist()
	prob=[0]*len(matrix)
	srow=[]
	sq_mat=np.square(matrix)
	sum_matrix=np.sum(sq_mat)

	np.random.seed(37)   #provides seed for random function so that same rows selected every time when processed but randomly                   
	
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			sq_row=np.square(matrix[i])
			sum_row=np.sum(sq_row)
		prob[i]=sum_row/sum_matrix

	rank=linalg.matrix_rank(matrix)                  
	srow=np.random.choice(len(matrix),rank,p=prob)      #randomly selects arrays from rank of ratings matrix
	#Duplicate selections are allowed

	lrow = len(srow)

	matrix2 = []

	for i in range(lrow):
		matrix2.append(matrix[srow[i]])
	return matrix2,srow

#In order to select top k values of probability for error calculation of precision on top k for CUR	
def select_top_k(matrix):               
	matrix=matrix.tolist()
	prob=[0]*len(matrix)
	srow=[]
	sq_mat=np.square(matrix)
	sum_matrix=np.sum(sq_mat)

	for i in range(len(matrix)):
		sum_row=0
		for j in range(len(matrix[i])):
			sum_row += matrix[i][j]**2
		prob[i]=sum_row/sum_matrix

	rank=linalg.matrix_rank(matrix)
	prob_sort=sorted(prob,reverse=True)
	prob=np.asarray(prob)
	k=0
	sum_old=sum(prob_sort)
	sum_new=0
	i=0
	while(sum_new<10*sum_old):
		if(i<len(prob_sort)):
			sum_new=sum_new+prob_sort[i]
			sum_old=sum_old-prob_sort[i]
			i=i+1
			k=k+1
		else:
			break
	srow=prob.argsort()[::-1][:k]

	matrix2 = []

	for j in range(len(srow)):
		matrix2.append(matrix[srow[j]])
	return matrix2,srow

#CUR based recommender system-
matrix=read_file("ratings.txt")

start=timeit.default_timer()
print("Starting CUR based recommender system evaluation:")

R,srow=select(matrix)
C,scol=select(matrix.transpose())

lrow=len(srow)
lcol=len(scol)

w = np.zeros((lrow,lcol))
for i in range(lrow):
	for j in range(lcol):
		w[i][j]=matrix[srow[i]][scol[j]]

U,sigma,V=svd_decompose(w)

U=U.transpose()
V=V.transpose()
l_sigma=len(sigma)
for i in range(l_sigma):
	if (abs(sigma[i][i])!=0):
		sigma[i][i]=1/sigma[i][i]           # (w+) : psuedoinverse matrix of sigma
		
SU=np.dot(sigma,U)
w=np.dot(V,SU)
C=(np.matrix(C)).transpose()
R=np.matrix(R)
wR=np.dot(w,R)
final_matrix=np.dot(C,wR)                       #final matrix from dot product by CUR

#print(final_matrix)
final_matrix = final_matrix.tolist()


print("\nTime taken for prediction:")
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


#Precision on top K for CUR
R,srow=select_top_k(matrix)
C,scol=select_top_k(matrix.transpose())
lrow=len(srow)
lcol=len(scol)

w = np.zeros((lrow,lcol))
for i in range(lrow):
	for j in range(lcol):
		w[i][j]=matrix[srow[i]][scol[j]]

U,sigma,V=svd_decompose(w)

U=U.transpose()
V=V.transpose()
l_sigma=len(sigma)
for i in range(l_sigma):
	if (abs(sigma[i][i])!=0):
		sigma[i][i]=1/sigma[i][i]           # (w+) : psuedoinverse matrix of sigma
		
SU=np.dot(sigma,U)
w=np.dot(V,SU)
C=(np.matrix(C)).transpose()
R=np.matrix(R)
wR=np.dot(w,R)
final=np.dot(C,wR)                       #final matrix from dot product by CUR

k_matrix=final.tolist()

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

print("\nThe precision on top k is: ")
print(precision)
