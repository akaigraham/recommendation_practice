from surprise import Dataset
from surprise.model_selection import train_test_split

# load jester dataset from built-in datasets
jokes = Dataset.load_builtin(name='jester')

# split into train and test sets
trainset, testset = train_test_split(jokes, test_size=0.2)

"""Memory Based Methods"""
from surprise.prediction_algorithms import knns
from surprise.similarities import cosine, msd, pearson
from surprise import accuracy

# print number of items and users
print(f'Number of users: {trainset.n_users}')
print(f'Number of items: {trainset.n_items}')

# because we have fewer items than users, will be more efficient to calc
# item-item similarity than user-user similarity
# sim_cos = {'name': 'cosine', 'user_based': False}

# train model - if user_based is set to True above will take a long time
# basic = knns.KNNBasic(sim_options=sim_cos)
# basic.fit(trainset)

# produce similarity metrics of each of the items to one another
# basic.sim

# test model to see how well it performed
# predictions = basic.test(testset)
# print(accuracy.rmse(predictions))

# try different similarity metric to see how it impacts performance
# sim_pearson = {'name':'pearson', 'user_based': False}
# basic_pearson = knns.KNNBasic(sim_options=sim_pearson)
# basic_pearson.fit(trainset)
# predictions = basic_pearson.test(testset)
# print(accuracy.rmse(predictions))

# try knn with means
# sim_pearson = {'name':'pearson', 'user_based':False}
# knn_means = knns.KNNWithMeans(sim_options=sim_pearson)
# knn_means.fit(trainset)
# predictions = knn_means.test(testset)
# print(accuracy.rmse(predictions))

# try with knn baseline
# sim_pearson = {'name':'pearson', 'user_based':False}
# knn_baseline = knns.KNNBaseline(sim_options=sim_pearson)
# knn_baseline.fit(trainset)
# predictions = knn_baseline.test(testset)
# print(accuracy.rmse(predictions))


"""Model Based Methods (Matrix Factorization)"""
# can use grid search to find best params within SVD pipeline

#optimal params
{'n_factors':100, 'n_epochs':10, 'lr_all':0.005, 'reg_all':0.4}

# import libraries
from surprise.prediction_algorithms import SVD
from surprise.model_selection import GridSearchCV

param_grid = {'n_factors':[20, 100], 'n_epochs':[5, 10], \
              'lr_all':[0.002, 0.005], 'reg_all':[0.4, 0.6]}

# grid search
# gs_model = GridSearchCV(SVD, param_grid=param_grid, n_jobs=-1, joblib_verbose=5)
# gs_model.fit(jokes)

# set up Singular-value decomposition with optimal params
svd = SVD(n_factors=100, n_epochs=10, lr_all=0.005, reg_all=0.4)
svd.fit(trainset)
predictions = svd.test(testset)
print(accuracy.rmse(predictions))

# making predictions
# here we are making a prediction for user 34 and item 25 using the SVD model
user_34_prediction = svd.predict('34', '25')

# access estimated rating
user_34_prediction[3]
