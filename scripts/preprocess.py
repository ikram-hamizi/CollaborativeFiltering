import pandas as pd
import numpy as np


def getUserItemMatrix(train, test):
	df_user_item = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
	df_user_item_test = test.pivot(index='userId', columns='movieId', values='rating').fillna(0)

	# Note: with this dataset, train and test have different shapes. To fix this, we do the following:

	# 1. Add missing userId's and movieId's to the test set
	df_user_item_test = df_user_item_test.reindex(
	    columns= df_user_item_test.columns.union(df_user_item.columns),
	    index  = df_user_item_test.index.union(df_user_item.index))

	# 2. Fill the NaN values with 0's
	df_user_item_test = df_user_item_test.fillna(0)

	# 3. Check if the training set and the test set are disjoint
	assert np.all(df_user_item_test.values * df_user_item.values == 0), "Trainset and test set are not disjoint. They share some similar user-item rating values"

	return df_user_item, df_user_item_test
