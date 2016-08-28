import pandas as pd
import os
import os.path
import math
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from scipy.signal import argrelmax

NUM_OF_SEC_BEFORE = 11.2
NUM_OF_SEC_AFTER = 0.2

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
	
