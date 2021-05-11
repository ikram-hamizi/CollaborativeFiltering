import torch
import torch.nn as nn
import torch.optim as optim


# Setting the environment
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float64)

class NCF_Recommender(nn.Module):
  
  def __init__(self, n_users, n_movies, train, lr=1e-3, k=50):
    
    super().__init__()
     
    self.n_users = n_users
    self.n_movies = n_movies
    self.train = train
    self.lr = lr
    self.k = k
    
    self.emb1 = nn.Embedding(n_users,  k)
    self.emb2 = nn.Embedding(n_movies, k)
	    
    self.hidden = nn.Sequential(
	      nn.Dropout(0.2),
	      nn.Flatten(),
	      nn.Linear(k*2 , 512),
	      nn.Dropout(0.2),
	      nn.ReLU(),
	      nn.Linear(512, 256),
	      nn.Dropout(0.2),
	      nn.ReLU(),
	      nn.Linear(256, 128),
	      nn.Dropout(0.2),
	      nn.ReLU(),
	      nn.Linear(128, 1)
       )
	  
    self.sigmoid = nn.Sigmoid()
    self._init()
  
  def forward(self, users, movies, range=(1,5)):
	  x = torch.cat([self.emb1(users), self.emb2(movies)], dim=1)
	  x = self.hidden(x)
	  x = self.sigmoid(x)

	  min, max = range
	    
	  return x*(max - min + 1) + min - 0.5 #scale output to [1,5]

  # Copied: from AML Lab 8
  def _init(self):
    """
	  Initialize embeddings and hidden layers weights with xavier.
	  """
    def init(m):
      if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    self.emb1.weight.data.uniform_(-0.05, 0.05)
    self.emb2.weight.data.uniform_(-0.05, 0.05)
    self.hidden.apply(init)


  def fit(self, n_epochs=30):
    	
    def _prepare_batches():
        # Creating Batches of size batch_sizee each
        batch_size = 128
        batches = []

        #>>>>temp data
        users = self.train['userId'].values
        movies = self.train['movieId'].values
        ratings = self.train['rating'].values

        R = len(ratings)

        #Create bathces of data
        for i in range(0, R, batch_size):
          limit = min(i+batch_size, R)

          users_batch = torch.tensor(users[i:limit], dtype=torch.long)
          movies_batch = torch.tensor(movies[i:limit], dtype=torch.long)
          ratings_batch = torch.tensor(ratings[i:limit], dtype=torch.float64)

          batches.append((users_batch, movies_batch, ratings_batch))

        #<<<<delete temp data
        del users
        del movies
        del ratings

        return batches

    batches = _prepare_batches()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=self.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=2)
    
    losses = []

    for epoch in range(n_epochs):
      epoch_loss = []
      for users_batch, movies_batch, ratings_batch in batches:
        
        self.zero_grad()
        y_pred = self.forward(users_batch.to(device), movies_batch.to(device)).squeeze()
        loss = criterion(ratings_batch.to(device), y_pred)
        
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
      epoch_loss_mean = np.mean(epoch_loss)
      losses.append(epoch_loss_mean)
      scheduler.step(epoch_loss_mean)
      
      print(f"Epoch {epoch+1}/{n_epochs}, Epoch Loss = {epoch_loss_mean}")

    return losses
