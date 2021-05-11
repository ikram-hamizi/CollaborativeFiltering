import numpy as np

#import local modules
import context
import scripts
from scripts.loss_metrics import MSE

def get_MF_with_ALS(Y, Y_test, U, I, total_ratings, total_ratings_test, K=50, C=0.01, tup=None, n_epochs=50, squared=True):
    """-
    args:
        - Y:        matrix of train ratings (ratings allow to complete the missing values)
        - Y_test:   matrix of test ratings  (true ratings to be compared with predictions)

        - K:        number of factors in the latent vectors
        - C:        regularization factor
        - tup:      range tuple to initialize the low rank matrices 
        - n_epochs: number of epochs
        - squared:  MSE loss if Talse, RMSE loss if True (default: True)

    return:
        - P, Q, train_losses, test_losses
    """

    def ALS(movie_features, fixed_matrix, k, C):
      """
      args:
        movie_features: the User-Item matrix (R)
        fixed_matrix: the fixed latent matrix (use P if updating Q and vice versa)
        k: number of factors in latent vectors
        C: Regularization parameter

      return:
        updated matrix after ALS
      
      e.g.: P = R*Q*(Q.T*Q + λ*I)^{-1}
      """  
      updated_matrix = movie_features @ fixed_matrix @ np.linalg.inv(fixed_matrix.T @ fixed_matrix + C * np.eye(k))
      return updated_matrix

    train_mask = np.nonzero(Y)
    test_mask  = np.nonzero(Y_test)

    def error_with_mask(Y, Y_pred, dataset):
      """
      Function: returns the difference between Y and Y_pred, i.e. error or deviance
      args:
        Y:      matrix of true ratings 
        Y_pred: matrix of predicted ratings 
      """

      if dataset == 'train':
        mask = train_mask
      elif dataset == 'test':
        mask = test_mask
      else:
        return
      error = Y[mask] - Y_pred[mask]
      return error
  
  
    if not isinstance(tup, tuple):
      print("Error: You passed a non-tuple object. \
      Pass a tuple of the form (minimum_value, maximum_value) to initialize the latent vectors.")

    P = np.random.uniform(0, 1/np.sqrt(k), size=(U,k)) #user latent matrix
    Q = np.random.uniform(0, 1/np.sqrt(k), size=(I,k)) #item latent matrix
  
    if tup is not None:
      P = np.random.uniform(tup[0],tup[1], size=(U,k)) #default (0,1)
      Q = np.random.uniform(tup[0],tup[1], size=(I,k)) #default (0,1)

    train_losses = []
    test_losses = []


    loss_type = 'MSE' if squared else 'RMSE'

    # verbose
    print_every = 1

    if 20 <= n_epochs < 50:
      print_every = 2

    elif n_epochs <= 200:
      print_every = 10

    elif n_epochs >= 500:
      print_every = 50 



    for epoch in range(n_epochs):

        P = ALS(Y, Q, k, C)
        Q = ALS(Y.T, P, k, C)

        Y_pred = (P @ Q.T)
        
        train_error = error_with_mask(Y, Y_pred, dataset='train')
        test_error  = error_with_mask(Y_test, Y_pred, dataset='test')

        train_LOSS = MSE(train_error, P, Q, total_ratings, C=C, squared=squared)
        test_LOSS  = MSE(test_error, P, Q, total_ratings_test, squared=squared)

        train_losses.append(train_LOSS)
        test_losses.append(test_LOSS)

        if epoch ==0 or (epoch+1)%print_every == 0:
          print(f"\nEpoch: {epoch+1}/{n_epochs}")
          print(f"Train {loss_type} (λ={C}) = {train_LOSS}")
          print(f"Test  {loss_type}    (λ=0) = {test_LOSS}")

          
    return P, Q, train_losses, test_losses
