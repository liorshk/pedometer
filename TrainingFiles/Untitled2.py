import sys
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import os
import os.path
from sklearn.metrics import mean_squared_error


#change to your own directory
royeedir = 'C:\Workspace\University\pedometer\TrainingFiles'
os.chdir(royeedir)

import lowpass as lp
import timestamp as ts
import datetime
import time
import omri as omri
import remove_doubles as rd
import rename as rn

#Global variables
order = 6
fs =119.0       # sample rate, Hz
cutoff = 3.667
num = []
peaklist = []
sdev= [1,1.2,1.5,1.8,2,3,4,5,6,7,8,9]
summarySession = pd.read_csv(royeedir+"\\" + "SessionsSummary.csv")
sc_original= summarySession["StepsCounts"]
s = pd.np.array(sc_original)


#Pull data from csv files
def pull_data(dir_name, file_name):
    f = pd.read_csv(dir_name + "\\" + file_name + ".csv")
    xs = []
    ys = []
    zs = []
    rs = []
    timestamps = []
    timeS = f["epoch"]
    x = f["load.txt.data.x"]
    y = f["load.txt.data.y"]
    z = f["load.txt.data.z"]
    xs.append(x)
    ys.append(y)
    zs.append(z)
    #timestamps.append(tstamps)
    
    i=0
    r=0
    mag = []
    while i<len(f.axes[0]):
        r = math.sqrt((xs[0][i])**2 + (ys[0][i])**2 + (zs[0][i])**2)
        tstamp = timeS[i].split(' ')[1].split('.')[0]
        x = time.strptime(tstamp.split(',')[0],'%H:%M:%S')
        j = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
        #timestamps as unix time
        #times = ts.main_function()
        timestamps.append(j)
        rs.append(r)
        i+=1
        
    return np.array(xs), np.array(ys), np.array(zs), np.array(rs), np.array(timestamps)


omri.main_function()

##compute standard deviation
def standard_deviation_function(arr,mean,devide_param):
    summ = 0
    tmp = 0
    var = 0
    counter = 0
    for i in range(1,len(arr)):
        tmp = float(arr[i])-mean
        tmp = tmp*tmp
        summ = summ + tmp
        counter = counter + 1
    var = summ / counter
    s_t =  mean +  (var**(0.5)/devide_param)
    return s_t

#First attempt: 'peak accel threshold'
def peak_accel_threshold(data, timestamps, threshold):

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


#finds the best threshold for each Train set.
# params: crossings - the number of time the graphs crosses the threshold.
#         sdev - an array containing the different options for the sd size
#         k = an index starting from 0
#         num - array
#         r - the data after lowpass filter

def find_best_thres(crossings,sdev,k,num,r):
    tempMin = ((len(crossings)/2) - s[k])**2
    Finalmin = tempMin
    index = 1
    for j in sdev[1:]:
        tmp_st = standard_deviation_function(r,np.average(r),j)
        t = lp.show_filter(r,cutoff,fs,order)
        crossings = peak_accel_threshold(r, t, tmp_st)
        tempMin = ((len(crossings)/2) - s[k])**2
        
        if tempMin < Finalmin:
            Finalmin = tempMin
            index = j
    num.append(index)
    tmp_st = standard_deviation_function(r,np.average(r),index)
    crossings = peak_accel_threshold(r, t, tmp_st)
    return crossings


def do_peak_thres(filenameToRead, k):
    x_data, y_data, z_data, r_data, timestamps = pull_data(royeedir, filenameToRead)
    #peak threshold
    r = lp.butter_lowpass_filter(r_data, cutoff, fs, order)
    tmp_st = standard_deviation_function(r,np.average(r),1)
    t = lp.show_filter(r,cutoff,fs,order)
    crossings = peak_accel_threshold(r, t, tmp_st)
    cross = find_best_thres(crossings,sdev,k,num,r)
    peaklist.append((len(cross)/2))

    
   
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


for i in range (1,31):

    do_peak_thres('Train_'+ str(i) +'_filtered',i-1)


for j in range(0,30):
    print j+1
    print ("orig is: " + str(sc_original[j]))
    print ("mine is " + str(peaklist[j] ))


print "the mse is" + str(mean_squared_error(s,peaklist))


