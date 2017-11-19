
# coding: utf-8

# In[26]:

# %load test.py

from lsh_tester import *

import os
import pandas as pd
import numpy as np
from scipy import sparse

def lsh_test(datas, queries, rand_num, num_neighbours, mips = False):

    type = 'l2'
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)
    args = {
                'type':      type,
                'k_vec':     [1, 2, 4, 8],
                'l_vec':     [2, 4, 8, 16, 32]
            }
    tester.run(**args)

    """

    type = 'cosine'
    tester = LshTesterFactory.createTester(type, mips, datas, queries, rand_num, num_neighbours)

    args = {
                'type':      type,
                'k_vec':    [1, 2, 4, 8],
                'l_vec':    [2, 4, 8, 16, 32]
            }
    tester.run(**args)

    """

if __name__ == "__main__":
#     gol._init()
    #gol.set_value('DEBUG', True)

    # create a test dataset of vectors of non-negative integers


    # Run this only after downloading the small or large dataset and make sure the path is set

    # This is in case of small dataset
    dataset = "datasets" + os.path.sep + "ml-latest-small"

    # This is in case of large dataset
    # dataset = "datasets"+os.path.sep+"ml-latest"

    # Name of the csv file
    name = "ratings.csv"

    # Read the csv file and get the appropriate column IDs
    ratings_df = pd.read_csv(dataset + os.path.sep + name, names= ["UserID", "MovieID", "Rating", "Timestamp"], header=0)

    # Converting to numbers and other changes
    ratings_df["UserID"] = pd.to_numeric(ratings_df["UserID"], errors='ignore')
    ratings_df["MovieID"] = pd.to_numeric(ratings_df["MovieID"], errors='ignore')
    ratings_df["Rating"] = pd.to_numeric(ratings_df["Rating"], errors='ignore')
    ratings_df["Timestamp"] = pd.to_numeric(ratings_df["Timestamp"], errors='ignore')


    # Create the matrix of the rating of movies with the userIDs, fill with 0 for the missing values 
    R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)  

    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)



    # Divided the dataset into test(Query) and Train(Data Points)
    #convert the rankings dataframe to sparse matrix
    #     sR = sparse.csr_matrix(R_demeaned)
    TRAIN_SIZE = 0.80

    # Create boolean mask
    # np.random creates a vector of random values between 0 and 1
    # Those values are filtered to create a binary mask
    #     msk = np.random.rand(sR.shape[0],sR.shape[1]) < TRAIN_SIZE
    #     r = np.zeros(sR.shape)



    
    
    # train_ratings = R_demeaned[:100]
    # test_ratings = R_demeaned[100:120]
    
    train_ratings = R_demeaned[:int(0.8*len(R_demeaned))]
    test_ratings = R_demeaned[int(0.8*len(R_demeaned)):]
    #mask itself is random
    #     train_ratings[msk] = r[msk]
    #     test_ratings[~msk] = r[~msk] # inverse of boolean mask


    test_r = np.array(test_ratings)
    train_r = np.array(train_ratings)


    num_neighbours = 1
    radius = 0.3
    r_range = 10 * radius


    datas = train_r
    queries = test_r
    lsh_test(datas, queries, r_range, num_neighbours, True)





