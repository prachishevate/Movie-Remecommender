"""
Contains various recommondation implementations
all algorithms return a list of movieids
"""

import numpy as np
from utils import movies

import pickle
from scipy.sparse import csr_matrix
import pandas as pd

movies = pd.read_csv('data/movies.csv')
movies = movies.set_index('movieId')

ratings = pd.read_csv('data/ratings.csv')
rating_per_movie=ratings.groupby('movieId')['userId'].count()
popular_movie=rating_per_movie.loc[rating_per_movie>20]
ratings=ratings.set_index('movieId').loc[popular_movie.index]
ratings=ratings.reset_index()

def recommend_with_NMF(query, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    movies.loc[query.keys()]
    
    with open('model/nmf_recommender.pkl', 'rb') as file:
        model = pickle.load(file)

    # 1. candiate generation
    
    # construct a user vector
    data=list(query.values())      # the ratings of the new user
    row_ind=[0]*len(data)          # we use just a single row 0 for this user
    col_ind=list(query.keys())  
    data, row_ind,col_ind
    # new user vector: needs to have the same format as the training data
    R = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))
    user_vec=csr_matrix((data, (row_ind, col_ind)), shape=(1, R.shape[1]))
    
   
    # 2. scoring
    
    # calculate the score with the NMF model
    scores=model.inverse_transform(model.transform(user_vec))
    scores=pd.Series(scores[0]) # convert to pandas series

    
    # 3. ranking
    scores[query.keys()]=0 # give a zero score to movies the user has allready seen
    scores=scores.sort_values(ascending=False) # sort the scores from high to low 
   
    
    # return the top-k highst rated movie ids or titles
    recommendations=scores.head(k).index
    return movies.loc[recommendations]

def recommend_random(k=3):
    return movies.sample(k)

def recommend_with_cosine_similarity(query, k=10):
    pass

def recommend_neighborhood(query, model, k=3):
    """
    Filters and recommends the top k movies for any given input query based on a trained nearest neighbors model. 
    Returns a list of k movie ids.
    """   
    pass
    

