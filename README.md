# movielens-predictor
This folder contains my code demonstrating matrix completion techniques on MovieLens data.
- ratings.csv is a data file obtained from the ml_32m dataset.
  This was in my project folder, but I omit it here due to size (877 MB).
  A download link for this data is available at https://grouplens.org/datasets/movielens/
- preprocess_data.py transforms ratings.csv into a usable form for this project.
  It writes the following csv files required by the matrix completion algorithms:
	- df_train.csv: a list of 4096 ratings to train on
	  Each rating includes user ID, movie ID, and score respectively.
	- M_train.csv: a 198 x 100 matrix reflecting observations from the training set
	- M_test.csv: a 198 x 100 matrix reflecting observations from the test set
- From here, you can run any of the three matrix completion methods.
  Each file displays test error when run.
	- baseline.py: execute the baseline methods (section 2.1)
	- alternating_minimization.py: execute the alternating minimization method (section 2.2)
	- frank_wolfe.m: execute the Frank-Wolfe method to solve convex relaxation (section 2.3)
- report.pdf describes my findings.

All of the Python files require the numpy and pandas packages.
preprocess_data.py additionally requires the scikit package.
frank_wolfe.m does not require any external libraries and should run on any MATLAB installation.
