import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read in the ratings.csv
df = pd.read_csv('ratings.csv')
df = df.to_numpy()[:, 0:3]
df = df.astype(int)
(n_df, _) = df.shape

# construct partially known matrix M for 200 users and all movies
n_M = 200
m_M = 87585
M = np.zeros((n_M, m_M))
for i in range(n_df):
    if df[i, 0] > n_M:  
        break
    if df[i, 0] <= n_M and df[i, 1] <= m_M:
        M[df[i, 0] - 1, df[i, 1] - 1] = df[i, 2]

# sort movies by how many ratings they have among our subset of users
ratings = np.count_nonzero(M, axis=0)
idx_sort = np.argsort(-ratings)
M = M[:, idx_sort]

# restrict M to the 100 most rated movies to keep problem size manageable
m_M = 100
M = M[:, 0:m_M]

# remove users with no data
M = M[~np.all(M == 0, axis=1)]
M = M.astype(int)
(n_M, _) = M.shape

# modify df to contain same data as M
df = np.zeros((np.count_nonzero(M), 3))
row = 0
for i in range(n_M):
    for j in range(m_M):
        if M[i, j] != 0:
            df[row, 0] = i + 1
            df[row, 1] = j + 1
            df[row, 2] = M[i, j]
            row += 1
df = df[~np.all(df == 0, axis=1)]
df = df.astype(int)

# split data into training set and test set
df_train, df_test = train_test_split(df, test_size = 0.2, random_state=42)
(n_train, _) = df_train.shape
M_train = np.zeros((n_M, m_M))
for i in range(n_train):
    M_train[df_train[i, 0] - 1, df_train[i, 1] - 1] = df_train[i, 2]
(n_test, _) = df_test.shape
M_test = np.zeros((n_M, m_M))
for i in range(n_test):
    M_test[df_test[i, 0] - 1, df_test[i, 1] - 1] = df_test[i, 2]

# write necessary M and df to csv files
np.savetxt('M_train.csv', M_train, delimiter=',', fmt='%d')
np.savetxt('df_train.csv', df_train, delimiter=',', fmt='%d')
np.savetxt('M_test.csv', M_test, delimiter=',', fmt='%d')