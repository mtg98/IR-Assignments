--------------------------------------------------------------------------
-----------------Recommender Systems Techniques-----------------
--------------------------------------------------------------------------

Introduction
----------------

A recommender system is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item.


Dependencies
---------------

1. numpy can be installed by using the command "pip install numpy". Visit: http://www.numpy.org/ for more.
2. math, timeit and copy package comes in inbuilt.


Files
-------------
Following files have been built in the project-
* collaborative.py, collaborative_baseline.py,  read_file.py, svd_decompose.py, svd.py, svd_energy_retained.py , cur.py , cur_energy_retained.py


Dataset
-----------
https://grouplens.org/datasets/movielens/ : ml-100k


Procedure
-------------

The program will read the data from an input file in the form of a matrix.
Following recommnender system approaches have been employed -
1. Collaborative without baseline approach
2. Collaborative baseline approach
3. SVD
4. SVD with 90% retained energy
5. CUR 
6. CUR with 90% retained energy

Error calculations have been don using-
1. RMSE (Root Mean Squeare Error)
2. Precision on top K
3. Spearman's index 

