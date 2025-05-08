import numpy as np
import pandas as pd

# read in M_train and df_train
M_train = pd.read_csv('M_train.csv',  header = None)
M_train = M_train.to_numpy()
M_train = M_train.astype(int)
df_train = pd.read_csv('df_train.csv',  header = None)
df_train = df_train.to_numpy()
df_train = df_train.astype(int)
(users, movies) = M_train.shape
(ratings, _) = df_train.shape

# define the loss function
def f(X, M):
    square_diff = np.square(X - M)
    omega = np.where(M != 0, 1, 0)
    return np.trace(omega.T @ square_diff)

# assign global average rating to unobserved values
X_1 = M_train.astype(np.float64)
global_average = np.mean(M_train[M_train != 0])
X_1 = np.where(X_1 == 0, global_average, X_1)

# assign user's average rating to unobserved values
X_2 = M_train.astype(np.float64)
user_averages = np.mean(M_train, axis = 1, where = M_train != 0)
for i in range(users):
    zero_cols = np.where(X_2[i, :] == 0)[0]
    X_2[i, zero_cols] = user_averages[i]

# evaluate test error
M_test = pd.read_csv('M_test.csv',  header = None)
M_test = M_test.to_numpy()
M_test = M_test.astype(int)
print(f"Using the global average, we achieve test error {f(X_1, M_test)}.")
print(f"Using the user averages, we achieve test error {f(X_2, M_test)}.")
