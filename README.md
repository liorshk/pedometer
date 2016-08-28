Pedometer
---------------------

#### Prerequisites

 - numpy
 - pandas
 - scipy
 - sklearn

#### Directory Structure

	|-- Code
	   |-- step_detector.py - Helper functions
	   |-- train.py - Used to evaluate the best model for detecting the steps and distance
	   |-- test.py - Reads the TestFiles directory and prints the step_count and distance
	|-- TestFiles
	   |-- Test_i.csv
	|-- TrainFiles
	   |-- Train_i.csv
	   |-- SessionsSummary.csv 

#### Train

    python train.py

	Output: classifier.pkl
	
Grid Search Results:

	GridSearchCV(cv=10, error_score='raise',
       estimator=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'kernel': ['linear'], 'C': array([  0.1,   0.2, ...,  49.8,  49.9]), 'epsilon': array([ 0.01,  0.06,  0.11,  0.16,  0.21,  0.26,  0.31,  0.36,  0.41,
        0.46,  0.51,  0.56,  0.61,  0.66,  0.71,  0.76,  0.81,  0.86,
        0.91,  0.96])},
       pre_dispatch='2*n_jobs', refit=True, scoring='mean_squared_error',
       verbose=0)
{'epsilon': 0.96000000000000008, 'C': 1.4000000000000001, 'kernel': 'linear'}

	 MSE:
	14.57755

	Best parameters set found for: 
	GridSearchCV(cv=10, error_score='raise',
		   estimator=Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
	   normalize=False, positive=False, precompute=False, random_state=None,
	   selection='cyclic', tol=0.0001, warm_start=False),
		   fit_params={}, iid=True, n_jobs=1,
		   param_grid={'alpha': array([  0.1,   0.2, ...,  49.8,  49.9])},
		   pre_dispatch='2*n_jobs', refit=True, scoring='mean_squared_error',
		   verbose=0)
	{'alpha': 0.5}

	 MSE:
	14.4082100964

	Best parameters set found for: 
	GridSearchCV(cv=10, error_score='raise',
		   estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
	   normalize=False, random_state=None, solver='auto', tol=0.001),
		   fit_params={}, iid=True, n_jobs=1,
		   param_grid={'alpha': array([  0.1,   0.2, ...,  49.8,  49.9])},
		   pre_dispatch='2*n_jobs', refit=True, scoring='mean_squared_error',
		   verbose=0)
	{'alpha': 8.5}

	 MSE:
	14.4199991899

	Best parameters set found for: 
	GridSearchCV(cv=10, error_score='raise',
		   estimator=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False),
		   fit_params={}, iid=True, n_jobs=1,
		   param_grid={'normalize': [True, False]}, pre_dispatch='2*n_jobs',
		   refit=True, scoring='mean_squared_error', verbose=0)
	{'normalize': False}

	 MSE:
	14.2564045671
	
#### Test

    python test.py