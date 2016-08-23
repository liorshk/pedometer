import math
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

# TODO: Comment
def butter_lowpass(cutoff, fs, order=5):
    """

    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# TODO: Comment
def butter_lowpass_filter(data, cutoff, fs, order=5):
    """

    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# TODO: Comment
def show_filter(r_data,cutoff,fs,order):
    """

    :param r_data:
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """
    b, a = butter_lowpass(cutoff, fs, order)
    # Plot the frequency response.

    ''' w, h = freqz(b, a, worN=8000)

    plt.subplot(2, 1, 1)
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
    plt.axvline(cutoff, color='k')
    plt.xlim(0, 0.5*fs)
    plt.title("Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.grid()
    '''
    T = 15.0 # seconds
    t = np.linspace(0, T, r_data.size, endpoint=False)

    # Filter the data, and plot both the original and filtered signals.
    y = butter_lowpass_filter(r_data, cutoff, fs, order)
    '''
    plt.subplot(2, 1, 2)
    plt.plot(t, r_data, 'b-',label='unfiltered')
    plt.plot(t, y, 'g-', linewidth=2,label='filtered')
    plt.xlabel('Time [sec]')
    plt.grid()
    plt.legend()

    plt.subplots_adjust(hspace=0.35)
    plt.show()
    '''
    return t

