# import libraries and load data
import pandas as pd
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD
from surprise.prediction_algorithms import KNNWithMeans, KNNBasic, KNNBaseline
from surprise.model_selection import GridSearchCV
import numpy as np
from typing import List

df = pd.read_csv('ratings.csv')

# drop unnecessary columns
new_df = df.drop(columns='timestamp')

# transform dataset into something compatible with surprise
from surprise import Reader, Dataset
reader = Reader()
data = Dataset.load_from_df(new_df, reader)

# print number of users and items
dataset = data.build_full_trainset()
print(f'Number of users: {dataset.n_users}')
print(f'Number of items: {dataset.n_items}')

# perform a grid search with SVD
# this will take several minutes to run
params = {'n_factors': [20, 50, 100],
          'reg_all': [0.02, 0.05, 0.1]}

gs_svd = GridSearchCV(SVD, param_grid=params, n_jobs=-1)
gs_svd.fit(data)

# print best scores
print(gs_svd.best_score)
print(gs_svd.best_params)

# cross validating with KNNBasic
knn_basic = KNNBasic(sim_options={'name':'pearson', 'user_based': True})
cv_knn_basic = cross_validate(knn_basic, data, n_jobs=-1)

for i in cv_knn_basic.items():
    print(i)
print('-----------------------')
print(np.mean(cv_knn_basic['test_rmse']))

# cross validating with KNNBaseline
knn_baseline = KNNBaseline(sim_options={'name':'pearson', 'user_based':True})
cv_knn_baseline = cross_validate(knn_baseline, data)

for i in cv_knn_baseline.items():
    print(i)
print('-----------------------')
print(np.mean(cv_knn_baseline['test_rmse']))


# making recommendations
df_movies = pd.read_csv('movies.csv')

# fit SVD model we had from before
svd = SVD(n_factors=50, reg_all=0.05)
svd.fit(dataset)

# generate predictions
svd.predict(2, 4)


def movie_rater(movie_df, num, genre=None):
    userID = 1000
    rating_list = []
    while num > 0:

        # check for genre - if genre, pull movie from that genre
        if genre:
            movie = movie_df[movie_df['genres'].str.contains(genre)].sample(1)

        # else pull random movie
        else:
            movie = movie_df.sample(1)

        # print movie
        print(movie)

        # get rating from user
        rating = input('How do you rate this movie on a scale of 1-5, press n if you have not seen :\n')
        if rating == 'n':
            continue
        else:
            rating_one_movie = {'userId':userID,'movieId':movie['movieId'].values[0],'rating':rating}
            rating_list.append(rating_one_movie)
            num -= 1
    return rating_list

# output user_rating
user_rating = movie_rater(df_movies, 4, 'Comedy')


"""Making predictions with the New Ratings"""
# add the new ratings to the original ratings DataFrame
new_ratings_df = new_df.append(user_rating, ignore_index=True)
new_data = Dataset.load_from_df(new_ratings_df, reader)

# train model using the new combined data frame
svd_ = SVD(n_factors=50, reg_all=0.05)
svd_.fit(new_data.build_full_trainset())

# make predictions for the user
list_of_movies = []
for m_id in new_df['movieId'].unique():
    list_of_movies.append((m_id, svd_.predict(1000, m_id)[3]))

# order the predictions from highest to lowest rated
ranked_movies = sorted(list_of_movies, key=lambda x:x[1], reverse=True)

# function for recommending movies
def recommended_movies(user_ratings, movie_title_df, n):
    for idx, rec in enumerate(user_ratings):
        title = movie_title_df.loc[movie_title_df['movieId'] == int(rec[0])]['title']
        print('Recommendation # ', idx+1, ': ', title, '\n')
        n -= 1
        if n == 0:
            break

recommended_movies(ranked_movies, df_movies, 5)
