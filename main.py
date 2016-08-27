%matplotlib inline
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from scipy.signal import argrelmax,find_peaks_cwt

def butter_lowpass(cutoff, sample_freq, order=6):
    """

    :param cutoff: Cutoff that was found after grid search
    :param sample_freq: Samples per second
    :return: Numerator (b) and denominator (a) polynomials of the IIR filter
    """
    
    #Nyquist frequency = half the sampling frequency
    nyq = 0.5 * sample_freq
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def calculate_num_of_sampling_per_sec(df):
    """  Gets the number of samples per second
    :param df: input data frame
    :return:  Return the total samples divided by the number of seconds in the files
    """

    # Get only the seconds out of the time column and then count the number of unique seconds
    # (Which is to count the number of seconds in the file)
    num_of_seconds = df.time.map(lambda t: t.second).nunique()

    # Return the total samples divided by the number of seconds in the files
    return df.time.count() / num_of_seconds

def distance_from_mean(series,divide_param):
    """  Gets the mean + x*std
    :param series: an array of values for specific column
    :param divide_param:  how far we want the samples to be relative to the mean
    :return: mean + std/divide_param
    """
    res = series.mean() + (series.std()/divide_param)
    return res

# TODO: Comment and explain
def return_end_of_walking_index(df,len_of_peace,max_space_between_peaks,starting_index,cur_max,curr_end):
    """
    :param df:
    :param len_of_peace:
    :param max_space_between_peaks:
    :param starting_index:
    :param cur_max:
    :param curr_end:
    :return:
    """
    start_index = starting_index
    end_index = starting_index + len_of_peace
    counter = 0
    last_peak = starting_index

    if len(df)< starting_index + len_of_peace:
        return curr_end

    for i in range (starting_index,starting_index + len_of_peace+1):
        counter = counter + df.is_peak[i]
        if df.is_peak[i]==1:
            #print i
            if (i - last_peak) > max_space_between_peaks:
                #print str(i) + ' split'
                return return_end_of_walking_index(df,len_of_peace,max_space_between_peaks,i-1,counter,last_peak)  #was instead of cur_end - last peak
            last_peak=i

    if counter > cur_max:
        cur_max = counter

    #print cur_max
    #print last_peak

    for i in range(starting_index+len_of_peace+2,len(df)):
        if df.is_peak[i]==1:
            if (i - last_peak) > max_space_between_peaks:
                #print str(i) + 'split2, curr_end = ' + str(end_index)
                #print 'max = ' + str(cur_max)
                return return_end_of_walking_index(df,len_of_peace,max_space_between_peaks,i-1,cur_max,end_index)
            last_peak = i

        counter = counter - df.is_peak[i-len_of_peace] + df.is_peak[i]
        if counter > cur_max:
            #print i
            cur_max = counter
            end_index = i
            start_index = i-len_of_peace

    return end_index


def filter_noise(df,samples_per_sec):
    """
    :param df: Full Dataframe
    :param samples_per_sec: Samples per second in dataframe
    :return: Gets only the relevant 10 seconds
    """

    peaks = argrelmax(np.array(df.y))[0]

    df['is_peak'] = 0 
    
    for i in range (0,len(peaks)):
        if df.y[peaks[i]] > df.y[peaks].mean():
            df.loc[peaks[i],'is_peak']=1

    index = return_end_of_walking_index(df,samples_per_sec*10,130,0,0,501)
    
    df_filtered = df[int(index - (samples_per_sec*NUM_OF_SEC_BEFORE)): int(index + samples_per_sec* NUM_OF_SEC_AFTER)]

    return df_filtered

#Global variables
dir = 'C:\Workspace\University\pedometer\TrainingFiles'
cutoff = 1.33

summarySession = pd.read_csv(os.path.join(dir,"SessionsSummary.csv"))
true_steps_count= summarySession["StepsCounts"]
duplicateTrainFiles = range(14,27)

NUM_OF_FILES = 30
NUM_OF_SEC_BEFORE = 11.2
NUM_OF_SEC_AFTER = 0.2


if __name__ == "__main__":
   
    mse = 0.0
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

        # Calculate the samples per second in the dataframe
        samples_per_sec = calculate_num_of_sampling_per_sec(df)
        
        # Get only the relevant data points out of the data frame
        filtered_df = filter_noise(df,samples_per_sec)

        # Butterworth filter
        b, a = butter_lowpass(cutoff, samples_per_sec)
        
        # Low filter
        magnitude_smoothed = lfilter(b, a, filtered_df.magnitude) 
        
        # Get peaks
        peaks = argrelmax(magnitude_smoothed)[0]  
        
        # Get peak values
        peak_values = magnitude_smoothed[peaks]        

        t = np.linspace(0, 15.0, magnitude_smoothed.size, endpoint=False) 

        error = ((len(peaks)) - (true_steps_count[i - 1])) ** 2
        
        mse += error
        print("{}: True number of steps: {}, Predicted: {}, Error Squared: {}".format(i, true_steps_count[i - 1], len(peaks), error))
        
        if error > 1:

            plt.figure(figsize=(20,20))
            plt.plot(t, magnitude_smoothed, 'b-', linewidth=2)

            plt.plot(t[peaks], magnitude_smoothed[peaks], 'ro', linewidth=2)

            plt.show()
        

    print("\t MSE: {}".format(mse / float(NUM_OF_FILES)))

