import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import os.path
from sklearn.metrics import mean_squared_error
import lowpass as lp


#Global variables
dir = 'TrainingFiles'
order = 6
fs =119.0       # sample rate, Hz
cutoff = 3.667
num = []
peaklist = []
sdev= [1,1.2,1.5,1.8,2,3,4,5,6,7,8,9]
summarySession = pd.read_csv(os.path.join(dir,"SessionsSummary.csv"))
true_steps_count= summarySession["StepsCounts"]
duplicateTrainFiles = range(14,27)

NUM_OF_FILES = 30
NUM_OF_SEC_BEFORE = 11.2
NUM_OF_SEC_AFTER = 0.2

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
def define_peaks(df,range_to_check,s_t):
    """
    :param df:
    :param range_to_check:
    :param s_t:
    :return:
    """
    df['is_peak'] = 0
    for i in range(range_to_check+1,len(df)-range_to_check):
        counter = 0
        if float(df.y[i])>s_t:
            for k in range(i-range_to_check,i+range_to_check):
                if float(df.y[k])<float(df.y[i]):
                    counter = counter + 1
            if counter >=range_to_check*2-2:
                df.loc[i,'is_peak'] = 1

    return df

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


# TODO: Comment
def filter_noise(df):
    """

    :param df:
    :return:
    """
    rangeC = 3  # number of left and right min values to find a peak

    samples_per_sec = calculate_num_of_sampling_per_sec(df)

    # TODO: Why did you do only >0 and < 100 ??
    y_samples = df.y[df.y >0][df.y < 100]
    mean = y_samples.mean()
    s_t = distance_from_mean(y_samples,1)

    # TODO: Please explain the following part
    if mean + 3 <= s_t:
        s_t = distance_from_mean(y_samples,4)

    df = define_peaks(df,rangeC,s_t)

    #between peaks ideal = 90
    index = return_end_of_walking_index(df,samples_per_sec*10,130,0,0,501)

    df_filtered = df[int(index - (samples_per_sec*NUM_OF_SEC_BEFORE)): int(index + samples_per_sec* NUM_OF_SEC_AFTER)]

    return df_filtered


# TODO: Comment
def peak_accel_threshold(data, timestamps, threshold):
    """

    :param data:
    :param timestamps:
    :param threshold:
    :return:
    """
    last_state = 'below'
    # below - less than threshold
    # above - above the threshold
    crest_troughs = 0
    crossings = []

    for i, datum in enumerate(data):

        current_state = last_state
        if datum < threshold:
            current_state = 'below'
        elif datum > threshold:
            current_state = 'above'

        if current_state is not last_state:
            if current_state is 'above':
                crossing = [timestamps[i], threshold]
                crossings.append(crossing)
            else:
                crossing = [timestamps[i], threshold]
                crossings.append(crossing)

            crest_troughs = crest_troughs + 1
        last_state = current_state

    return np.array(crossings)


def find_best_thres(crossings,sdev,k,num,r):
    """  Finds the best threshold for each Train set.
    :param crossings: the number of time the graphs crosses the threshold.
    :param sdev: an array containing the different options for the sd size
    :param k: an index starting from 0
    :param num: array
    :param r: the data after lowpass filter
    :return: the best threshold
    """
    tempMin = ((len(crossings)/2) - true_steps_count[k])**2
    Finalmin = tempMin
    index = 1
    for j in sdev[1:]:
        tmp_st = distance_from_mean(r,j)
        t = lp.show_filter(r,cutoff,fs,order)
        crossings = peak_accel_threshold(r, t, tmp_st)
        tempMin = ((len(crossings)/2) - true_steps_count[k])**2
        
        if tempMin < Finalmin:
            Finalmin = tempMin
            index = j
    num.append(index)
    tmp_st = distance_from_mean(r,index)
    crossings = peak_accel_threshold(r, t, tmp_st)
    return crossings


# TODO: Comment
def get_peak_thres(df, k):
    """

    :param df:
    :param k:
    :return:
    """
    #peak threshold
    r = lp.butter_lowpass_filter(df.magnitude, cutoff, fs, order)
    tmp_st = distance_from_mean(r,1)
    t = lp.show_filter(r,cutoff,fs,order)
    crossings = peak_accel_threshold(r, t, tmp_st)
    cross = find_best_thres(crossings,sdev,k,num,r)
    return (len(cross)/2)
   
    '''
    print "Peak Acceleration Threshold Steps:", len(cross)/2 -1
    plt.plot(t, r, 'b-', linewidth=2)
    plt.plot(cross.T[0], cross.T[1], 'ro', linewidth=2)

    plt.title("trial" + " - Peak Acceleration Threshold")
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    for i in range(1,NUM_OF_FILES+1):

        # Read the file
        df = pd.read_csv(os.path.join(dir,'Train_' + str(i) + '.csv'))

        # Change column names, parse dates and create magnitude column
        df['x']= df["load.txt.data.x"]
        df['y']= df["load.txt.data.y"]
        df['z']= df["load.txt.data.z"]
        df['magnitude'] = df.apply(lambda row: math.sqrt((row.x)**2 + (row.y)**2 + (row.z)**2),axis=1)
        df['time']= pd.to_datetime(df['epoch'], format='%Y-%m-%d %H:%M:%S.%f')
        df['timedelta'] = df['time'] - df['time'][0]

        # Drop duplicates
        if i in duplicateTrainFiles:
            df = df.drop_duplicates(subset=['x','y','z']).reset_index(drop=True)

        # Get only the relevant data points out of the data frame
        filtered_df = filter_noise(df)

        peak_thres = get_peak_thres(filtered_df,i-1)
        print("{}: True number of steps: {}, Predicted: {}, Error Squared: {}".format(i, true_steps_count[i - 1], peak_thres, (true_steps_count[i - 1] - peak_thres) ** 2))

    #print("MSE: {}".format(mean_squared_error(s,peaklist)))



