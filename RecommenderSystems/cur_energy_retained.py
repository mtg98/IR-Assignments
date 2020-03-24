import timeit
import math
import numpy as np
from numpy import linalg
from svd_decompose import svd_decompose, eigen
from read_file import read_file

#Here duplications are not allowed while slection of rows and columns for matrix

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

	matrix2 = []
	
	for i in range(len(srow)):
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
print("Starting CUR with 90% energy retention based recommender system:")

R,srow=select(matrix)
C,scol=select(matrix.transpose())

lrow=len(srow)
lcol=len(scol)

w = np.zeros((lrow,lcol))
for i in range(lrow):
	for j in range(lcol):
		w[i][j]=matrix[srow[i]][scol[j]]

U,sigma,V=svd_decompose(w)

#l_sigma=len(sigma)

#smaller singular values are removed until the retained energy
sum_eig_sq=0
for i in range(len(sigma)):
	sum_eig_sq += (sigma[i][i]**2)
sum_current=sum_eig_sq

while (sum_current>(0.9*sum_eig_sq)):               #energy retention in this case is 90%
	temp=sigma[len(sigma)-1][len(sigma)-1]
	temp=temp**2
	sum_deleted=sum_current-temp
	if(sum_deleted>(0.9*sum_eig_sq)):
		sum_current=sum_deleted
		x=len(sigma)-1
		sigma=np.delete(sigma,x,0)                 #row and column corresponding to smallest singular value
		sigma=np.delete(sigma,x,1)                   # removed from sigma matrix
		
		U=np.delete(U,len(U[0])-1,1)             #remove corresponding column to singular value removed from sigma in U matrix             
		V=np.delete(V,len(V.T[0])-1,0)           #remove corresponding row to singular value removed from sigma in V matrix
	else:
		break                                    #if retained energy falls below 90%

for i in range(len(sigma)):
	if (abs(sigma[i][i])!=0):
		sigma[i][i]=1/sigma[i][i]           # (w+) : psuedoinverse matrix of sigma

sigma = np.square(sigma)
		
U=U.transpose()
V=V.transpose()

SU=np.dot(sigma,U)
w=np.dot(V,SU)
C=(np.matrix(C)).transpose()
R=np.matrix(R)
wR=np.dot(w,R)
final_matrix=np.dot(C,wR)                       #final matrix from dot product by CUR

#print(final_matrix)
final_matrix = final_matrix.tolist()


print("\nTotal Time taken for prediction:")
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

sum_eig_sq=0
for i in range(len(sigma)):
	sum_eig_sq += (sigma[i][i]**2)
sum_current=sum_eig_sq

while (sum_current>(0.9*sum_eig_sq)):               #energy retention in this case is 90%
	temp=sigma[len(sigma)-1][len(sigma)-1]
	temp=temp**2
	sum_deleted=sum_current-temp
	if(sum_deleted>(0.9*sum_eig_sq)):
		sum_current=sum_deleted
		x=len(sigma)-1
		sigma=np.delete(sigma,x,0)                 #row and column corresponding to smallest singular value
		sigma=np.delete(sigma,x,1)                   # removed from sigma matrix
		
		U=np.delete(U,len(U[0])-1,1)             #remove corresponding column to singular value removed from sigma in U matrix             
		V=np.delete(V,len(V.T[0])-1,0)           #remove corresponding row to singular value removed from sigma in V matrix
	else:
		break                                    #if retained energy falls below 90%

for i in range(len(sigma)):
	if (abs(sigma[i][i])!=0):
		sigma[i][i]=1/sigma[i][i]           # (w+) : psuedoinverse matrix of sigma

sigma = np.square(sigma)
		
U=U.transpose()
V=V.transpose()

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
		if(old_value==new_value):
			match=match+1       #if rounded new value matches to the old value

precision=(match*100)/count

print("\nThe precision on top k is: ")
print(precision)
