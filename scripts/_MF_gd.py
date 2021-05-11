import numpy as np

#import local modules
import context
import scripts
from scripts.loss_metrics import MSE

def get_MF_with_GD(Y, Y_test, U, I, total_ratings, total_ratings_test, k=50, C=0.0, tup=None, n_epochs=50, squared=True, lr=8e-5):
  """-
  args:
      - Y:        matrix of train ratings (ratings allow to complete the missing values)
      - Y_test:   matrix of test ratings  (true ratings to be compared with predictions)
    
      - k:        number of factors in the latent vectors (default: 50)
      - C:        regularization factor (default: 0)
      - tup:      range tuple to initialize the low rank matrices (default: np.full((*,K), 0.1))
      - n_epochs: number of epochs (default: 50)
      - squared:  MSE loss if Talse, RMSE loss if True (default: True)

      - lr:       learning rate

  return:
      - P, Q, train_losses, test_losses
  """
  M_train = Y.copy()
  M_train[M_train>0] = 1

  M_test = Y_test.copy()
  M_test[M_test>0] = 1

  P = np.full((U,k), 0.1) #user latent matrix (Simon Funk)
  Q = np.full((I,k), 0.1) #item latent matrix (Simon Funk)

  if tup is not None:
    P = np.random.uniform(tup[0],tup[1], size=(U,k)) #default (0,1)
    Q = np.random.uniform(tup[0],tup[1], size=(I,k)) #default (0,1)
  
  train_losses_gd = []
  test_losses_gd = []


  # verbose
  print_every = 1

  if 20 <= n_epochs < 50:
    print_every = 2

  elif n_epochs <= 200:
    print_every = 10

  elif n_epochs >= 500:
    print_every = 50 


  # get initial error
  Y_pred = P @ Q.T
  train_error = (Y - Y_pred) * M_train

  loss_type = 'MSE' if squared else 'RMSE'


  for epoch in range(n_epochs):

        P_grad = (C*P - (train_error)   @ Q) 
        Q_grad = (C*Q - (train_error).T @ P)

        P -= (lr*P_grad)
        Q -= (lr*Q_grad)

        Y_pred = (P @ Q.T)
        
        train_error = (Y      - Y_pred) * M_train
        test_error  = (Y_test - Y_pred) * M_test

        train_LOSS = MSE(train_error, P, Q, total_ratings, C=C, squared=squared)
        test_LOSS  = MSE(test_error, P, Q, total_ratings_test, squared=squared)

        train_losses_gd.append(train_LOSS)
        test_losses_gd.append(test_LOSS)

        if epoch ==0 or (epoch+1)%print_every == 0:
          print(f"\nEpoch: {epoch+1}/{n_epochs}")
          print(f"Train {loss_type} (λ={C}) = {train_LOSS}")
          print(f"Test  {loss_type}    (λ=0) = {test_LOSS}")


  return P, Q, train_losses_gd, test_losses_gd
