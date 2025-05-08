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

# define the matrix factorization U^T @ V = X
d = 1
U = np.random.randn(d, users)
V = np.random.randn(d, movies)
X = U.T @ V

# execute alternating minimization
alpha = 0.01
T = 100
ratings_order = np.arange(ratings)
for t in range(T):
    np.random.shuffle(ratings_order)
    for d in range(ratings):
        i = int(df_train[ratings_order[d], 0] - 1)
        j = int(df_train[ratings_order[d], 1] - 1)
        U[:, i] = U[:, i] - alpha * (np.dot(U[:, i], V[:, j]) - df_train[ratings_order[d], 2]) * V[:, j]
        V[:, j] = V[:, j] - alpha * (np.dot(U[:, i], V[:, j]) - df_train[ratings_order[d], 2]) * U[:, i]

# evaluate test error
M_test = pd.read_csv('M_test.csv',  header = None)
M_test = M_test.to_numpy()
M_test = M_test.astype(int)
print(f"Alternating minimization results in test error {f(U.T @ V, M_test)}.")