#local classes
import context
import scripts
import classes

import scripts.dataExtract as dataExtract
import scripts.preprocess as preprocess

from classes.NeuralCollaborativeFiltering import NCF_Recommender
from classes.MatrixFactorization import MatrixFactorization

#python libraries
import pandas as pd
import numpy as np
import torch
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#*********************
# 1. GLOBAL VARIABLES
#*********************

train, test = dataExtract.get_data() # ['userId', 'movieId', 'rating'] (dataframes)

df_user_item, df_user_item_test = preprocess.getUserItemMatrix(train, test) # train and test User-Item matrices (dataframes)

Y = df_user_item.values.copy()            # User-Item matrix (values)
Y_test = df_user_item_test.values.copy()  # User-Item matrix (values) - Test Set

total_ratings = len(train)      # total observed ratings
total_ratings_test = len(test)  # total observed ratings - Test Set


unique_users = train['userId'].unique()    
unique_movies = train['movieId'].unique()

U = len(unique_users)  # total users   = 6687
I = len(unique_movies) # total movies = 5064

k = 50

#**********************
# 2. Optional functions
#**********************
def print_info():
	print("Number of unique users: ", U)
	print("Number of unique movies: ", I)
	print("Length of the dataset: ", len(train))
	print("Length of the sparse matrix: ", U*I) #UxI

	known_ratings_percentage = 100*(len(train))/(U*I)
	print(f"The percentage of known ratings: θ% = {known_ratings_percentage:.2f}%")

print_info()


#**************************
# 3. RECOMMENDERS TRAINING
#**************************

# 1/ MATRIX FACTORIZATION: with ALS (Alternating Least Squares) - verbose = True
print(); print("-----" * 14)
print("1. Completing the User-Item matrix with Matrix Factorization: ALS")
print("-----" * 14)

MFObject1 = MatrixFactorization(Y, Y_test, U, I, total_ratings, total_ratings_test, 'ALS')
P1, Q1, train_loss1, test_loss1 = MFObject1.get_matrices(k=20, C=2e-1, tup=(0, 1/np.sqrt(k)), n_epochs=50, squared=True)

with open('P1.pkl','wb') as f:
	pickle.dump(P1, f)
with open('Q1.pkl','wb') as f:
	pickle.dump(Q1, f)


# 2/ MATRIX FACTORIZATION: with GD (Gradient Descent)			- verbose = True
print(); print("-----" * 14)
print("2. Completing the User-Item matrix with Matrix Factorization: GD")
print("-----" * 14, '\n')

MFObject2 = MatrixFactorization(Y, Y_test, U, I, total_ratings, total_ratings_test, 'SGD')
P2, Q2, train_loss2, test_loss2 = MFObject2.get_matrices(k=50, C=0, tup=(0,1), n_epochs=40, squared=True, lr=8e-5)

with open('P2.pkl','wb') as f:
	pickle.dump(P2, f)
with open('Q2.pkl','wb') as f:
	pickle.dump(Q2, f)

# 3/ DEEP LEARNING (Neural Collaborative Filtering)				- verbose = True
print(); print("-----" * 14)
print("3. Completing the User-Item matrix Neural Collaborative Filtering")
print("-----" * 14, '\n'); 
model = NCF_Recommender(n_users=max(unique_users)+1, n_movies=max(unique_movies)+1, train=train, lr=1e-3, k=50).to(device)
model.fit(n_epochs=2)
torch.save(model, "NCF.pth")
print("/!\ Note /!\: The training of the NN model is completed for 2 epochs only as a demo.\ncuda must be installed to be able to train for more epochs.")
