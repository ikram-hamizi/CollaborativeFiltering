#local classes
import context
import scripts
import scripts.datExtract as datExtract
import scripts.preprocess as preprocess

from classes.NeuralCollaborativeFiltering import NeuralCollaborativeFiltering
from classes.MatrixFactorization import MatrixFactorization

#python libraries
import pandas as pd
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#*********************
# 1. GLOBAL VARIABLES
#*********************

train, test = dataExtract.get_data() # ['userId', 'movieId', 'rating'] (dataframes)

df_user_item, df_user_item_test = getUserItemMatrix(train, test) ##train and test user-item matrices (dataframes)

Y = df_user_item.values.copy()
Y_test = df_user_item_test.values.copy()

unique_users = train['userId'].unique()    
unique_movies = train['movieId'].unique() #5064

U = len(unique_users)  #total users   = 6687
I = len(unique_movies) # total movies = 5064

total_ratings = len(train)
total_ratings_test = len(test)

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
	print(f"The percentage of known ratings: θ% = {known_ratings_percentage}%")

print_info()


#**************************
# 3. RECOMMENDERS TRAINING
#**************************

# 1/ MATRIX FACTORIZATION: with ALS (Alternating Least Squares) - verbose = True
print(); print("-----" * 5)
print("1. Completing the User-Item matrix with Matrix Factorization: ALS")
print("-----" * 5)

MFObject1 = MatrixFactorization(Y, Y_test, U, I, total_ratings, total_ratings_test, 'ALS')
P1, Q1, train_loss1, test_loss1 = MFObject1.get_matrices(K=k, C=0.02, tup=(0,1), n_epochs=50, squared=True)


# 2/ MATRIX FACTORIZATION: with GD (Gradient Descent)			- verbose = True
print(); print("-----" * 5)
print("2. Completing the User-Item matrix with Matrix Factorization: GD")
print("-----" * 5, '\n')

MFObject2 = MatrixFactorization(Y, Y_test, U, I, total_ratings, total_ratings_test, 'SGD')
P2, Q2, train_loss2, test_loss2 = MFObject2.get_matrices(K=k, C=0, tup=(0,1), n_epochs=50, squared=True, lr=1e-5 beta=0.9)

	
# 3/ DEEP LEARNING (Neural Collaborative Filtering)				- verbose = True
print(); print("-----" * 5)
print("3. Completing the User-Item matrix Neural Collaborative Filtering")
print("-----" * 5, '\n'); 
model = NeuralCollaborativeFiltering.NCF_Recommender(n_users=max(unique_users)+1, n_movies=max(unique_movies)+1).to(device)
model.fit(n_epochs=50)
#torch.save(model, "NCF.pth")


