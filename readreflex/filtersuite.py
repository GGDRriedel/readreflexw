# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:17:37 2019

@author: Riedel
"""

from scipy.signal import butter, lfilter
import numpy as np
import pandas as pd
import copy
import scipy
from .utils import normalize
import tqdm
   
def butter_bandpass(lowcut,highcut,fs,order):
#Taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# Necessary Helper Funkction for filtering
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
    
def butter_bandpass_filter(radarobject,lowcut, highcut, order=5):
    ''' Applies a Butterworth Lowcut Filter to the data within the object
        Lowcut: Low cutoff freq. IN relative numbers! Example: lowcut=0.1
        Highcut: Uper cutoff freq. IN relative numbers! Example: highcut=0.4, needs to be <0.5
    '''
    #turn the data
    copyrad=copy.copy(radarobject)
    data=radarobject.traces.T
    fs=1.0/radarobject.header["timeincrement"]
    print("Sampling Frequency 1/dt = {:f} GHz".format(fs/1E9) )
    lowcut=lowcut*fs
    print("Lowcut Frequency = {:f} GHz".format(lowcut/1E9))
    highcut=highcut*fs
    print("Highcut Frequency = {:f} GHz".format(highcut/1E9))
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data,axis=0)
    copyrad.traces=y.T
    return copyrad
    
def trace_average_removal(radarobject,q=2,window_form='boxcar'):
    '''removes an average of the last 2*q and the next 2*q traces from each trace
    so the window witdth for the average is always 2*q+1
    
    Warning: cyclical indexing! the first q traces will be calculated with 
    the average from the last ones
    
    Keyword arguments:
    radarobject -- the container for the radargram
    q -- Window index, window length= 2*q+1
    window -- window form , standard: 'boxcar'
    
    Window forms coming from Pandas Module. 
    See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
    
    RETURNS: [result, removed field]
    
    
    
    '''
    #taken from  https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    
    def running_mean(x, N):
        cumsum = np.cumsum(x,axis=1) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

    copyrad=copy.deepcopy(radarobject)
    filteredrad=copy.deepcopy(radarobject)
    data=pd.DataFrame(copyrad.traces)
    data=data.rolling(window=q*2+1,axis=0,center=True,win_type=window_form,min_periods=1).mean()
    copyrad.traces=copyrad.traces-data.values
    filteredrad.traces=data.values
    return copyrad, filteredrad
    
    
    
def trace_full_average_removal(radarobject):
    '''removes the complete average
    
    RETURNS: [result, removed field]
    
    
    
    '''
    #taken from  https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    
    copyrad=copy.deepcopy(radarobject)
    
    data=pd.DataFrame(copyrad.traces)
    datamean=data.mean(axis=0).values
    copyrad.traces=copyrad.traces-datamean
    
    return copyrad, datamean
    
def gain_agc_instant(radarobject,targetrms,q=5):
    ''' Instantaneous gain control after 
    https://wiki.seg.org/wiki/Instantaneous_AGC
    '''
    
   # def trace_instagc(trace,RMS): 
        
        
        
    rad=copy.deepcopy(radarobject)   
    data=pd.DataFrame(np.abs(rad.traces))
    data=radarobject.header["tracenumber"]*targetrms/data.rolling(window=q*2+1,axis=1,center=True,min_periods=1).sum()
    rad.traces=data.values*rad.traces
    
    return rad
        
def gain_exp(radarobject,a=1.0,k=-0.1,b=1):
    '''
    returns radargram object with an exponentially gained signal
    exponential curve is fitted to the abs mean of all traces, 
    start values need to be given for the exponential fit
    f(x)=a*e^k +b
    Standard: a=1.0,k=-0.1,b=1
    '''
    
     #copy 
    copyrad=copy.deepcopy(radarobject)
    #calculate the mean of all traces: 
    meantrace=np.mean(np.abs(radarobject.traces),axis=0)
    import matplotlib.pyplot as plt
    #plt.plot(meantrace)
    #normalize it to one
    
    #time vector for fitting function
    time=range(radarobject.header["samplenumber"])
    #define exponential function
    def exponential(x, a, k, b):
        return a*np.exp(x*k) + b
    #optimizer use non-linear least squares to fit a function, exponential, to data.
    popt_exponential, pcov_exponential = scipy.optimize.curve_fit(
            exponential, time, meantrace,
            p0=[a,k,b])
    #print(popt_exponential,pcov_exponential)
    
    #plt.plot(time,exponential(time,popt_exponential))
    
    #multiply with the inverse of the exponential
    
    copyrad.traces=(normalize(1.0/exponential(time,*popt_exponential)))*copyrad.traces
    
    #plt.plot(exponential(time,*popt_exponential))
    #copyrad.radarplot()
    return copyrad


    
    