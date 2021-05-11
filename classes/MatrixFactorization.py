#python libraries
import pandas as pd
import numpy as np

#local classes
import context
import scripts
from scripts._MF_als import get_MF_with_ALS
from scripts._MF_gd import get_MF_with_GD


class MatrixFactorization:
  
  def __init__(self, Y, Y_test, U, I, total_ratings, total_ratings_test, optimization):
    """
    args:
      - Y:       matrix of train ratings (ratings allow to complete the missing values)
	    - Y_test:  matrix of test ratings  (true ratings to be compared with predictions)
      
      - U: total number of users
      - I: total number of items (movies)
      
      - total_ratings     : total number of observed ratings (trainset)
      - total_ratings_test: total number of observed ratings (test-set)
      
      - Optimization: 'SGD' or 'ALS':  
    """
    self.Y = Y
    self.Y_test = Y_test
    self.optimization = optimization
    self.U = U
    self.I = I
    self.total_ratings = total_ratings
    self.total_ratings_test = total_ratings_test


  def get_matrices(self, K=50, C=0, tup=None, n_epochs=50, squared=True, lr=8e-5):
    """
		args:
		  - K:        number of factors in the latent vectors
	      - C:        regularization factor
	      - lr: 	  learning rate
	      - tup:      range tuple to initialize the low rank matrices 
	      - n_epochs: number of epochs
	      - squared:  MSE loss if Talse, RMSE loss if True (default: True)
	      - beta:     for momentum calculation
    """
    
    if self.optimization == 'SGD':
      return get_MF_with_GD(self.Y, self.Y_test, self.U, self.I, self.total_ratings, self.total_ratings_test, K, C, tup, n_epochs, squared, lr)


    elif self.optimization == 'ALS':
      return get_MF_with_ALS(self.Y, self.Y_test, self.U, self.I, self.total_ratings, self.total_ratings_test, K, C, tup, n_epochs, squared)

    else:
      print("You can only choose 'SGD' or 'ALS'")
   
    
    
