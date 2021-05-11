import pandas as pd
import wget
from zipfile import ZipFile

"""Get training and testing set"""
def get_data():
	# Download the data
	wget.download("https://github.com/Gci04/AML-DS-2021/raw/main/data/collaborative-filtering.zip")

	# Extract the data
	with ZipFile("collaborative-filtering.zip", 'r') as f:
	  f.extractall()

	# Read the data
	train = pd.read_csv("collaborative-filtering/train.csv")
	test  = pd.read_csv("collaborative-filtering/test.csv")

	return train, test

