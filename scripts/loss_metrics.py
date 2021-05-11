import numpy as np 


def MSE(error, A, B, total_ratings, C=0, squared=True): 

  squared_magnitude_P = (A @ A.T).sum()
  squared_magnitude_Q = (B @ B.T).sum()

  # MSE
  regularization = C*(squared_magnitude_P + squared_magnitude_Q)
  loss = ((error**2).sum() + regularization)/(total_ratings)
  
  # RMSE
  if not squared:
    loss = np.sqrt(loss)                                                                   

  return loss