import torch
from zipfile import ZipFile
import pandas as pd
import torch

#local modules
import RecommendMovies
import NeuralCollaborativeFiltering
from NeuralCollaborativeFiltering import NCF_Recommender #get attribute NCF_Recommender


torch.set_default_dtype(torch.float64)

#0. read the data
with ZipFile("collaborative-filtering.zip", 'r') as f:
  f.extractall()

train = pd.read_csv("collaborative-filtering/train.csv")

#1. load the predictor model
model = torch.load("NCF_model.pth", map_location=torch.device('cpu'))

#2. prompt the user
userID = int(input("Enter user ID:"))

#3. create an NCF object
ncf = RecommendMovies.NCF(train)
top_liked_by_user, new_predicted_to_user = ncf.predictNN(userID, model)

#4. print
# 1/ Real Ratings
print("****" * 8)
print(f"Top 5 liked movies by user {userID}")
print("****" * 8)
print(top_liked_by_user)

print()

# 2/ Predicted Rating s
print("✰---" * 12)
print(f"Top 5 movie recommendations for user {userID} -- NN")
print("---✰" * 12)
print(new_predicted_to_user)
