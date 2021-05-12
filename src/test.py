#import local modules
import context
import scripts
import classes

import classes.RecommendMovies as RecommendMovies
import scripts.dataExtract as dataExtract

#import python libraries
import pandas as pd
import numpy as np
import pickle
import torch


userID = 330

#load the training data
train, _ = dataExtract.get_data() # ['userId', 'movieId', 'rating'] (dataframes)

# get a recommender object
ncf = RecommendMovies.NCF(train)

#load the models
with open('P1.pkl','rb') as f:
	P1 = pickle.load(f)
with open('Q1.pkl','rb') as f:
"""
	Q1 = pickle.load(f)
with open('P2.pkl','rb') as f:
	P2 = pickle.load(f)
with open('Q2.pkl','rb') as f:
	Q2 = pickle.load(f)
model = torch.load("NCF.pth")
"""
#*********************
# Already liked movies
#*********************
top_liked_by_user, new_predicted_to_user = ncf.predictMF(userID, P1, Q1, train)
print("****" * 8)
print(f"Top 5 liked movies by user {userID}")
print("****" * 8)
# 1/ Real Ratings
print(top_liked_by_user)



#=============
#  MF w/ ALS
#=============
	
print("✰---" * 14)
print(f"Top 5 new movie recommendations for user {userID} -- MF (ALS)")
print("---✰" * 14)
# 2/ Predicted Ratings
print(new_predicted_to_user)


"""
#=============
#  MF w/ GD
#=============
	
#_, new_predicted_to_user = ncf.predictMF(userID, P2, Q2, train)
print("✰---" * 13)
print(f"Top 5 movie recommendations for user {userID} -- MF (GD)")
print("---✰" * 13)
# 2/ Predicted Rating s
print(new_predicted_to_user)
  
  
  
#=============
#     NCF
#=============
#_, new_predicted_to_user = ncf.predictNN(userID, model, train)
print("✰---" * 12)
print(f"Top 5 movie recommendations for user {userID} -- NN")
print("---✰" * 12)
print(new_predicted_to_user)
"""
