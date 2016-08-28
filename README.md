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

	Best parameters set found for SVR: 
	
	{'epsilon': 0.96000000000000008, 'C': 1.4000000000000001, 'kernel': 'linear'}

	 MSE:
	14.57755

	Best parameters set found for Lasso: 
	
	{'alpha': 0.5}

	 MSE:
	14.4082100964

	Best parameters set found for Ridge: 
	{'alpha': 8.5}

	 MSE:
	14.4199991899

	Best parameters set found for LinearRegression: 
	{'normalize': False}

	 MSE:
	14.2564045671
	
#### Test

    python test.py
