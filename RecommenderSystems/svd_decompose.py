import math
import numpy as np
from numpy import linalg


def eigen(matrix):
	evalues,evectors=linalg.eig(matrix)   #eig function from numpy.linalg returns eigenvalues and its corresponding eigenvectors
	                                   #evalues and evectors contain complex values
	complex_set={}                                
	l=len(evalues)
	for i in range(0,l):
		complex_set[evalues[i]]=evectors[:,i]   #dictionary of evalues:evectors
		
	evalues=sorted(evalues)
	eigen_dict={}
	for i in evalues:
		eigen_dict[round(i.real,2)]=complex_set[i].real
	return eigen_dict                         #dictionary of real valued evalues and evectors


def svd_decompose(matrix):         #function to return 3 matrices on decomposition : A=U*sigma*V
	matrix_T=matrix.transpose()
	A=np.dot(matrix,matrix_T)
	Ux=eigen(A)
	B=np.dot(matrix_T,matrix)
	Vx=eigen(B)

	eigenvalues=[]
	for eigenvalue in Ux:
		if abs(eigenvalue)!=0:
				round_eigenvalue=round(eigenvalue,2)
				eigenvalues.append(round_eigenvalue)         #values list has non-zero rounded eigenvalues

	eigenvalues=sorted(eigenvalues,reverse=True)                  #descending order

	l_eig = len(eigenvalues)
	sigma=np.zeros((l_eig,l_eig))                     # diagonal matrix
	np.fill_diagonal(sigma,eigenvalues,wrap=True)       #fill diagonal elements with eigenvalues
	sigma=np.sqrt(sigma)                                #diagonal matrix has squared root elements
	l_sigma=len(sigma)


	len_u1 = len(Ux[eigenvalues[0]])     #length of eigenvector in terms of its components
	len_v1 = len(Vx[eigenvalues[0]])
	
	U=np.zeros((len_u1,l_eig))       #make matrix to hold eigenvalues and its corresponding eigenvector
	V=np.zeros((len_v1,l_eig))
	
	for i in range(l_sigma):              #to arrange eigenvectors vertically for each eigenvalue

		m=len(Ux[eigenvalues[i]])
		for j in range(m):
			U[j][i]=Ux[eigenvalues[i]][j]      

		n=len(Vx[eigenvalues[i]])
		for j in range(n):
			V[j][i]=Vx[eigenvalues[i]][j]

	V=V.transpose()

	#The above breakdown of matrix A unstable , on breaking down might lead to A = -A which is false
    #To avoid above condition for each row in V matrix and its corresponding column in U matrix
    # if any of the two negative, multiply U matrix's column by -1
	for i in range(l_sigma):                   
		v_row = V[i]
		v_row_matrix = np.matrix(v_row)
		v_row_transpose_matrix = v_row_matrix.T
		v_column_vector = np.dot(matrix, v_row_transpose_matrix)

		u_array = []
		for row in U:
			column_no = 0
			for column in row:
				if column_no == i:             #looking for column corresponding to row
					u_array.append(column)
					break
				column_no += 1

		u_vector = np.matrix(u_array)
		u_column_vector = u_vector.T
		flag = False
		len_v = len(v_column_vector) 

		for j in range(len_v):
			if u_column_vector[j] != 0.0:
				if v_column_vector[j]/u_column_vector[i] < 0.0:      #checking the condition
					flag = True
					break

		if flag == True:
			len_u_array = len(u_array)
			for k in range(len_u_array):
				U[k][i]=-1.0*U[k][i]          #sign changed if flag condition is true
	
	return U,sigma,V