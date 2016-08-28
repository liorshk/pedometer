import pandas as pd
import os
import os.path
import math
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy.signal import argrelmax,find_peaks_cwt

from step_detector import *


#Global variables
dir = '../TrainingFiles'

summarySession = pd.read_csv(os.path.join(dir,"SessionsSummary.csv"))
true_steps_count= summarySession["StepsCounts"]
duplicateTrainFiles = range(14,27)

NUM_OF_FILES = 30
NUM_OF_SEC_BEFORE = 11.2
NUM_OF_SEC_AFTER = 0.2

displayGraph = False

##### Detect Step Count ######

total_errors = []
step_counts = []
for i in range(1,NUM_OF_FILES+1):

    # Read the file
    df = pd.read_csv(os.path.join(dir,'Train_' + str(i) + '.csv'))

    # Change column names, parse dates and create magnitude column
    df['x']=df["load.txt.data.x"]
    df['y']=df["load.txt.data.y"]
    df['z']=df["load.txt.data.z"]

    df['magnitude'] = df.apply(lambda row: math.sqrt((row.x)**2 + (row.y)**2 + (row.z)**2),axis=1)
    df['time']= pd.to_datetime(df['epoch'], format='%Y-%m-%d %H:%M:%S.%f')

    # Drop duplicates
    if i in duplicateTrainFiles:
        df = df.drop_duplicates(subset=['x','y','z']).reset_index(drop=True)

    # Get the number of steps
    # Calculate the samples per second in the dataframe
    samples_per_sec = calculate_num_of_sampling_per_sec(df)

    # Get only the relevant data points out of the data frame
    filtered_df = filter_noise(df,samples_per_sec)

    # Butterworth filter
    cutoff = 1.33
    b, a = butter_lowpass(cutoff, samples_per_sec)

    # Low filter
    magnitude_smoothed = lfilter(b, a, filtered_df.magnitude)

    # Get peaks
    peaks = argrelmax(magnitude_smoothed)[0]

    # Get peak values
    peak_values = magnitude_smoothed[peaks]

    t = np.linspace(0, 15.0, magnitude_smoothed.size, endpoint=False)

    error = ((len(peaks)) - (true_steps_count[i - 1])) ** 2

    total_errors.append(error)
    step_counts.append(len(peaks))
    print("{}: True number of steps: {}, Predicted: {}, Error Squared: {}".format(i, true_steps_count[i - 1], len(peaks), error))

    if displayGraph:
        plt.figure(figsize=(20,20))
        plt.plot(t, magnitude_smoothed, 'b-', linewidth=2)

        plt.plot(t[peaks], magnitude_smoothed[peaks], 'ro', linewidth=2)

        plt.show()


print("\t MSE for Step Count: {}".format(np.average(total_errors)))


##### Detect Walking Distance ######
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.metrics import mean_squared_error


data = np.zeros((len(summarySession),1))
# we tried adding more variables - but they don't help much

# Data = predicted step counts from before
data[:,0] = np.array(step_counts)

# Target = distance covered
target = summarySession.DistanceCovered

print("Performing Grid Search in order to find the best model and best parameters")

# Splitting the data into train and test
# The train set will be used for cross validation
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=1)

# Doing Grid search for Support Vector Regression over C and epsilon parameters
svr = GridSearchCV(SVR(), cv=10,
                   param_grid={"kernel":['linear'],
                               "C": np.arange(0.1,50,0.1),
                               "epsilon":np.arange(0.01,1,0.05)}
                   ,scoring="mean_squared_error")
				   
# Doing Grid search for Lasso Regression over alpha parameters
l = GridSearchCV(Lasso(), cv=10,
                  param_grid={"alpha": np.arange(0.1,50,0.1)}
                  ,scoring="mean_squared_error")
				  
# Doing Grid search for Ridge Regression over alpha parameters
r = GridSearchCV(Ridge(), cv=10,
                  param_grid={"alpha": np.arange(0.1,50,0.1)}
                  ,scoring="mean_squared_error")
				  
# Doing Grid search for Linear Regression over normalized / not normalized
lr = GridSearchCV(LinearRegression(), cv=10,
                  param_grid={"normalize":[True,False]}
                  ,scoring="mean_squared_error")


clfs = [svr,l,r,lr]

bestClf = {}
bestMse = 10000
for clf in clfs:
    clf.fit(X_train, y_train)

    print("\nBest parameters set found for: \n" + str(clf))

    print(clf.best_params_)

    print("\n MSE:")
    y_true, y_pred = y_test, clf.predict(X_test)

    mse = mean_squared_error(y_true,y_pred)
    print(mse)
    if mse < bestMse:
        bestClf = clf

from sklearn.externals import joblib	
joblib.dump(bestClf, 'classifier.pkl')
	
	