import pandas as pd
import numpy as np
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class NCF:

	def __init__(self, train):
		self.movieIds = NCF._get_list_of_all_movies(train)
		self.train = train


	# Function: returns a dataframe of unique movieIds
	def _get_list_of_all_movies(df_train):
		return pd.DataFrame(data=df_train['movieId'].unique(), columns=['movieId']) #movieIds 


	# Function: recommends movies
	def _recommend_movies(userID, df_movieIds, df_train, model, n=5):
	
		""" Function: returns top-5 recommended movies for userID
		args:
		userID: userId in training set
		df_movieIds: all unique movies in the training set (DataFrame of all unique movieId's) 
		df_train: training set (DataFrame with at least three columns: userId, movieId, rating)
		model: NN model if technique
		n: number of top-n movies recommended
		"""

		#1. >get       watched movieIds by the user  +  sort them by rating (descending)
		movies_watched = df_train[df_train.userId == (userID)].sort_values(['rating'], ascending=False) # get all userId: movieId, rating

		#2. >get >NOT< watched movieIds by the user
		movies_not_watched = df_movieIds[~df_movieIds['movieId'].isin(movies_watched['movieId'])]


		#3. preprocess the data for the model
		user = [userID] * len(movies_not_watched) #userID (repeated)
		new_movies = movies_not_watched['movieId'].tolist() #non-watched movies
		
		user = torch.tensor(user).type(torch.long).to(device=device) 
		new_movies = torch.tensor(new_movies).type(torch.long).to(device=device)
		
		rating_preds = model(user, new_movies).squeeze() # get a list of predicted ratings from the model
		
		new_movies = new_movies.tolist()
		rating_preds = rating_preds.tolist()
		
		data = {"movieId":new_movies, userID:rating_preds}
		
		#4. get predictions for NON-watched movies
		recommendations = pd.DataFrame(data).sort_values(by=userID, ascending=False).iloc[:n] # this list only has non-watched movies
		
		if 'index' in recommendations.columns:
			recommendations.drop(columns='index') 
		
		return movies_watched.iloc[:n], recommendations


	def predictNN(self, userID, model):
		# Get top 5 recommneded movies for user with id "330" with NN technique
		y_test_df, y_predicted_df = NCF._recommend_movies(userID, self.movieIds, self.train, model=model)
		return y_test_df, y_predicted_df
