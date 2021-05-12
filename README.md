# CollaborativeFiltering

Advanced Machine Learning - Innopolis University (Spring2021)

Homework 2 Link: https://hackmd.io/@gFZmdMTOQxGFHEFqqU8pMQ/B1z1pP6L_

The task is to recommend new movies from MovieLens to users using Matrix Factorization for Matrix Completion and Deep Learning for Neuarl Collaborative Filtering (NCF).

### Test 1: Git Actions
- The libraries can be installed using `pip install -r requirements.txt`
- The project can be tested on Git Actions by clicking on "Actions" here on this repository and running the available workflow.
- The workflow is defined in `.github/workflows/setup.yml`
- The default userID used for this test is userId=330
- TODO: add cuda installation to `setup.yml` to train the NCF model.


### Test 2: Docker Image
On a terminal, run:
- `docker run -it ikramhub/ncf_recommender`
- Enter a userID after the prompt
- The NCF Neural Network will recommend new Top-5 movies.

Alternatively, you can:
- Download the Docker-Workdir folder
- On a terminal, run `python NCF_Predict.py`

#### The Structure of the repository
```bash
├── classes                              <- Class Modules
│   ├── MatrixFactorization.py           <- For Matrix Completion (SGD and ALS) 
│   ├── NeuralCollaborativeFiltering.py  <- For Training a NCF
│   └── RecommendMovies.py  
│
├── scripts                              <- Standalone scripts (modules)
│   ├── dataExtract.py                   <- Data Extraction script
│   ├── preprocess.py                    <- Data preprcessing (training set to user-item matrix)
│   ├── loss_metrics.py                  <- Script with a loss funciton (MSE and RMSE)
│   └── 2 private scripts 
│
├── src                                  <- Code for use in this project.
│   ├── train.py                         <- matrix completion and model train script
│   └── test.py                          <- matrix/model test script
│ 
├── Docker_Workdir                       <- Docker Work Directory
├── notebooks                            <- Notebook: Data Exploration, Model Training and Testing
├── requirements.txt                            
└── README.md     
```
