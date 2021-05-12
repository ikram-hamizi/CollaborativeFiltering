import pandas as pd
import numpy as np


  
class NCF: 
	
	def __init__(self, train):
		self.movieIds = NCF._get_list_of_all_movies(train)
		self.train = train
	
	# Function: returns a dataframe of unique movieIds
	@staticmethod
	def _get_list_of_all_movies(df_train):
	  return pd.DataFrame(data=df_train['movieId'].unique(), columns=['movieId']) #movieIds 
	
	
	# Function: recommends movies
	def _recommend_movies(userID, df_movieIds, df_train, technique, n=5, df_preds_MF=None, model=None):

	  """ Function: returns top-5 recommended movies for userID
	      args:
		  userID: userId in training set

		  df_movieIds: all unique movies in the training set (DataFrame of all unique movieId's) 

		  df_train: training set (DataFrame with at least three columns: userId, movieId, rating)

		  technique: 'NN' for Neural Network and 'MF' for Matrix Factorization

		  df_preds_MF: reconstructed matrix from matrix factorization if technique == 'MF'

		  model: NN model if technique == 'NN'
	  """

	  if technique != 'NN' and technique != 'MF':
	    print("You must enter the 'technique' type correctly: 'MF' for Matrix Factorization or 'NN' for Neural Network")
	    return;

	  #1. >get       watched movieIds by the user  +  sort them by rating (descending)
	  movies_watched = df_train[df_train.userId == (userID)].sort_values(['rating'], ascending=False) # get all userId:	movieId, rating


	  #2. >get >NOT< watched movieIds by the user
	  movies_not_watched = df_movieIds[~df_movieIds['movieId'].isin(movies_watched['movieId'])]

	  # MATRIX FACTORIZATION
	  if technique == 'MF':
	    #3.` get all movieIds that exist in the dataset + sort them by rating (descending)
	    sorted_user_predictions = df_preds_MF[df_preds_MF.index==userID].sort_values(by=userID, axis=1, ascending=False)
	    sorted_user_predictions = pd.DataFrame(np.transpose(sorted_user_predictions.values), columns=sorted_user_predictions.index, index=sorted_user_predictions.columns)
	 

	    #4.` get predictions for NON-watched movies
	    recommendations = (movies_not_watched).merge(
		pd.DataFrame(sorted_user_predictions).reset_index(),
		how = 'left', left_on = 'movieId', right_on = 'movieId').rename(
		    columns = {userID: 'Predictions'}).sort_values(
		        'Predictions', ascending = False).iloc[:n]
	  
	  # NEURAL COLLABORATIVE FILTERING
	  elif technique == 'NN':
	    user = [userID] * len(movies_not_watched) #userID (repeated)
	    new_movies = movies_not_watched['movieId'].tolist() #non-watched movies

	    user = torch.tensor(user).to(device=device, dtype=torch.long) 
	    new_movies = torch.tensor(new_movies).to(device=device, dtype=torch.long)

	    rating_preds = model(user, new_movies).squeeze() # get a list of predicted ratings from the model

	    # 3.`` prepare the data for the model
	    new_movies = new_movies.cpu().detach().tolist() 
	    rating_preds = rating_preds.cpu().detach().tolist()
	    data = {"movieId":new_movies, userID:rating_preds}

	    #4.`` get predictions for NON-watched movies
	    recommendations = pd.DataFrame(data).sort_values(by=userID, ascending=False) # this list only has non-watched movies
	    
	  if 'index' in recommendations.columns:
	    recommendations.drop(columns='index') 

	  return movies_watched.iloc[:n], recommendations

	def predictMF(self, userID, P, Q, df_train):
	  # Get the predictions from Matrix Factorization technique.
	  y_predicted = P @ Q.T
	  preds_df = pd.DataFrame(y_predicted, columns = df_user_item.columns)

	  # Get top 5 recommneded movies for user with id "330"
	  y_test_df, y_predicted_df = NCF._recommend_movies(userID, movieIds, df_train, technique='MF', df_preds_MF=preds_df)

	  return y_test_df, y_predicted_df

	def predictNN(self, userID, model, df_train):
	  # Get top 5 recommneded movies for user with id "330" with NN technique
	  y_test_df, y_predicted_df = NCF._recommend_movies6(userID, movieIds, df_train, technique='NN', model=model)

	  return y_test_df, y_predicted_df
