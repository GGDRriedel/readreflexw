# -*- coding: utf-8 -*-
import numpy as np

def nullstelle(x1,x2,y1,y2):
    worked=1
    if x2<x1: 
        print('Nullstellenfehler, x1 > x2')
        print(x1,x2)
        worked=0
    elif y2<y1:
        print('y2 kleiner y1')
        worked=0
    
        
    #return (x2*y1-x1*y2)/(y1-y2)
    result= -1*y1/((y2-y1)/(x2-x1))+x1
    return result,worked

def standardize(data):
    '''
    Careful! Returns an array of dtype float64 in every case!
    '''
    import matplotlib.pyplot as plt
    data=data.astype('float64')
    [X,Y]=data.shape
    mean=np.mean(data,axis=1)
    #plt.plot(mean)
    std=np.std(data,axis=1)
    #plt.plot(std)
    standardized=data.copy()
    for tracenumber in range(0,X-1):
       standardized[tracenumber,:]=((data[tracenumber,:] - mean[tracenumber]) / std[tracenumber])
    return standardized
        
def shift(arr, num, fill_value=0):
    num=int(num)
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
        
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
       
    else:
        result = arr
    return result
   
def mean_confidence_interval(array,confidence=0.95):
    '''
    Returns the confidence interval parameters of an array
    
    IN:
    array: Numpy Array
    confidence: confidence in p
    
    Out:
        
    mean,mean-confidence,mean+confidence
    '''
    import scipy.stats
    assert type(array)==np.ndarray,'Not a Numpy Array!'
    a = 1.0 * array
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def find_first_prom_sig(data,promquote=3,smax=None):
    '''
    
    Parameters
    ----------
    data : 2d np-array
    promquote: integer - What nth part of the maximum of the signal is prominent
                enough. Standard: 3-> 1/3rd of max(abs( amplidute))
        
    Finds the first arrival maximum 

    Returns
    -------
    firstmax : index of first arrival maximum np array
    
    note: fail-poof for when it just finds the max

    '''
    from scipy.signal import find_peaks
    if smax:
        data=data[:,:smax]
    [X,Y]=data.shape
    prominencelimit=np.abs(np.abs(data)).max()/promquote
    allprompeaks=[]
    for tracenumber in range(0,X):
        prom_peaks,properties=find_peaks(data[tracenumber,:],prominence=(prominencelimit,None))
        #if prom_peaks[0]>50:
         #   print('what')
        if prom_peaks.size>0:
            #append the index of argmax
            allprompeaks.append(prom_peaks[properties['prominences'].argmax()])
        else:
            allprompeaks.append(np.nan)
        
    return np.array(allprompeaks)





def find_prom_sig(data,promquote=3,smax=None,option='first'):
    '''
    
    Parameters
    ----------
    data : 2d np-array
    promquote: integer - What nth part of the maximum of the signal is prominent
                enough. Standard: 3-> 1/3rd of max(abs( amplidute))
    smax: integer - limiting index for signal length
    option: string - defaults to "first", gives first prominent peak
                                "most", gives most prominent peak as per promquote
        
    Finds the first arrival maximum 

    Returns
    -------
    firstmax : index of first arrival maximum np array
    
    note: fail-poof for when it just finds the max

    '''
    from scipy.signal import find_peaks
    if smax:
        data=data[:,:smax]
    [X,Y]=data.shape
    prominencelimit=np.abs(np.abs(data)).max()/promquote
    allprompeaks=[]
    for tracenumber in range(0,X):
        prom_peaks,properties=find_peaks(data[tracenumber,:],prominence=(prominencelimit,None))
        #if prom_peaks[0]>50:
         #   print('what')
        if prom_peaks.size>0:
            #append the index of argmax
            if option=='first':
                allprompeaks.append(prom_peaks[0])    
            if option=='most':
                allprompeaks.append(prom_peaks[properties['prominences'].argmax()])
        else:
            allprompeaks.append(np.nan)
        
    return np.array(allprompeaks)



def findmaxv2(data,directmaxbuffer=200, direct_min=70,
              direct_max=112,
              smin=None,
              smax=None):
    '''
    Finds the peaks based on the peak of the first 
        
    data: 2D Array of dtype float
    directmaxbuffer: minimum distance from directmax to look from
    smin: minimum sample number
    smax: maximum sample number
    
    '''
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    import tqdm
    
    
    allpeaksf=[]
    minpeaksf=[]
    nullstellen=[]
    sigzeronullstellen=[]
    sigzero=[]
    baseline=[]
    
    idxadder=0
    
    if smin:
        data=data[:,smin:]
        idxadder=smin
    if smax and smin:
        #data=data[:,:(smax-smin)]
        data=data[:,smin:smax]
    elif smax:
        data=data[:,:smax]
    [X,Y]=data.shape
    
    for tracenumber in tqdm.tqdm(range(0,X),'Finding Peaks and Zeros'):
        #sanitize the trace: 
        if not np.any(data[tracenumber]): 
            data[tracenumber]=data[tracenumber-1]
        #find the direct signal peak and it's height
        
        maxdirectpeak,properties= find_peaks(-1*data[tracenumber,direct_min:direct_max],distance=Y)
        #sometimes noise makes the peak detect go crazy, we scratch the negative
        if maxdirectpeak.shape==(0,):
            maxdirectpeak,properties= find_peaks(data[tracenumber,direct_min:direct_max],distance=Y)
            
            
        
        #print(maxdirectpeak[0]+direct_min)
        #get its height
        height=np.abs(data[tracenumber,maxdirectpeak[0]+direct_min])
        directmaxpeakindex=maxdirectpeak[0]+direct_min
        #print(height)
        #now we use that as the minimum height. 
        # added 1 unit, all peaks that are higher than that are found. The first one of those needs to be our
        #ground reflection signal
        #print(find_peaks(data[tracenumber,:],height=height+1))
        larger_peaks=find_peaks(data[tracenumber,directmaxpeakindex+directmaxbuffer:],height=height+1)
        #no peaks found is usually due to the max depth set too shallow
        #for now we take the full length as the index. not ideal
        #but zero can still be found. correction height will be wrong
        while len(larger_peaks)==0 or larger_peaks[0].shape==() or larger_peaks[0].shape==(0,): 
            #is there a value after DC that is larger than our height?
            if data[tracenumber,directmaxpeakindex+1:].max()>height:
                
                larger_peaks=(np.array(data[tracenumber,directmaxpeakindex+directmaxbuffer:].argmax(),ndmin=1),'dummy')
                height=data[tracenumber,directmaxpeakindex+directmaxbuffer+larger_peaks[0][0]]
            #if not, there is no strong reflection and we need to lower the barrier
        
            else: 
                height=height*0.9
                larger_peaks=find_peaks(data[tracenumber,directmaxpeakindex+directmaxbuffer:],height=height+1)
        
        reflectmaxpeakindex=larger_peaks[0][0]+directmaxpeakindex+directmaxbuffer
       
    
        #find the minimum like in V1
        #print(tracenumber,directmaxpeakindex,reflectmaxpeakindex)
        subsignal=data[tracenumber,directmaxpeakindex:reflectmaxpeakindex]
        minpeaks=find_peaks(subsignal*-1)
        
        
        #check if minimum within what we are working is lower than the very last we found
        # this is in case the signal from max to min contains a local min
        
        #take last one
        candidatevalue=subsignal[minpeaks[0][-1]]
        peakpicker=minpeaks[0][-1]
        #go through it backwards
        for minpeakcandidate in minpeaks[0][::-1]: 
            if subsignal[minpeakcandidate]<candidatevalue:
                candidatevalue=subsignal[minpeakcandidate]
                peakpicker=minpeakcandidate
            else:
                break
            
                    
        #while subsignal[minpeaks[0][peakpicker]]>subsignal[minpeaks[0][peakpicker-1]]:
        #    if np.abs(peakpicker)>=minpeaks[0].shape[0]:
        #        break
        #    peakpicker=peakpicker-1
            
        reflectminpeakindex=peakpicker+directmaxpeakindex

        #the zero is the signal between the reflect min and max ansolute, negative and maxed
        reflectionramp=data[tracenumber,reflectminpeakindex:reflectmaxpeakindex]
        #print(reflectionramp)
        #print(np.abs(reflectionramp*-1))
        #print(subsignal)
       
        # value closest to zero of the abs should be the zero crossing

        #since we cut at the min peak we add that index back up
        zeromin=np.abs(reflectionramp).argmin() +    reflectminpeakindex            

       # plt.plot(data[tracenumber])
       # plt.scatter(directmaxpeakindex,data[tracenumber,directmaxpeakindex])
       # plt.scatter(reflectmaxpeakindex,data[tracenumber,reflectmaxpeakindex])
       # plt.scatter(reflectminpeakindex,data[tracenumber,reflectminpeakindex])
       # plt.scatter(zeromin,data[tracenumber,zeromin])
       # if tracenumber in [71258]:
       #     print("pause")
       #     plt.plot(data[tracenumber])
       #     plt.scatter(directmaxpeakindex,data[tracenumber,directmaxpeakindex])
       #     plt.scatter(reflectmaxpeakindex,data[tracenumber,reflectmaxpeakindex])
       #     plt.scatter(reflectminpeakindex,data[tracenumber,reflectminpeakindex])
       #     plt.scatter(zeromin,data[tracenumber,zeromin])
       #     plt.title(zeromin)
       #     plt.show()
       #     plt.pause(0.1)
        
        
        
        allpeaksf.append(reflectmaxpeakindex)
        minpeaksf.append(reflectminpeakindex)
        nullstellen.append(zeromin)
        sigzeronullstellen.append(zeromin)
        sigzero.append(zeromin)
        baseline.append(directmaxpeakindex)
        
        
        
        
    allpeaksf=np.array(allpeaksf)
    minpeaksf=np.array(minpeaksf)
    nullstellen=np.array(nullstellen)
    baseline=np.array(baseline)
    sigzero=np.array(sigzero)
    sigzeronullstellen=np.array(sigzeronullstellen)
   
  
    
            
    return allpeaksf+idxadder,minpeaksf+idxadder,nullstellen+idxadder,sigzero+idxadder,sigzeronullstellen+idxadder,baseline+idxadder


def findmax(data,smin=None,smax=None):
    '''
    Finds the Peaks necessary for the catalogue detection
    
    data: 2D Array of dtype float
    smin: minimum sample number
    smax: maximum sample number
    
    Return: 
    allpeaksf+idxadder :            highest peak in each trace
    minpeaksf+idxadder :            lowest peak in each trace
    nullstellen+idxadder:           analytical zero crossing
    sigzero+idxadder:               closest sample to zero between peaks
    sigzeronullstellen+idxadder:    analytical zero between value before and after crossing zero
    baseline+idxadder:              first peak with 1/n-th height of maximum of trace(THIS SHOULD BE BETTER)
    
    idxadder:                       just adds the smin in case it was given
    
    
    NOTICE: Just realized it could be faster by making distance larger than half of signal length
    '''
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt
    import tqdm
    
    [X,Y]=data.shape
    allpeaksf=[]
    minpeaksf=[]
    nullstellen=[]
    sigzeronullstellen=[]
    sigzero=[]
    baseline=[]
    idxadder=0
    
    if smin:
        data=data[:,smin:]
        idxadder=smin
    if smax and smin:
        data=data[:,:(smax-smin)]
    elif smax:
        data=data[:,:smax]
    [X,Y]=data.shape
        
    for tracenumber in tqdm.tqdm(range(0,X),'Finding Peaks and Zeros'):
         # delivers all peaks
        maxpeak= find_peaks(data[tracenumber,:],distance=Y)
        minpeak= find_peaks(-1*data[tracenumber,:],distance=Y)
        peakind=maxpeak[0]
       
        # could be empty for empty traces, lets make sure it's not
        if peakind.size>0:    
            allpeaksf.append(peakind[0])
            height=data[tracenumber,:].max()/5
            baseindex=find_peaks(data[tracenumber,:],height=height)[0][0]
          
            # find the last minimum in the signal before the global max
            subsignal=data[tracenumber,0:peakind[0]]
            minpeaks=find_peaks(subsignal*-1)
            # print(subsignal)
            #peaks are unreliable for local minima so we see if they are at least SOMEWHAT
            #near zero
            peakpicker=-1 #last peak index
            #check if minimum within what we are working is lower than the very last we found
            while np.abs(subsignal[minpeaks[0][peakpicker]])<subsignal[minpeaks[0][peakpicker-1]]:
                peakpicker=peakpicker-1
            #find_peaks returns dict, take first element and the last element of that array
            minpeaksf.append(minpeaks[0][peakpicker])
            #function call to the linear interpolated zero

            
            zero,worked=nullstelle(minpeaks[0][-1],
                                          peakind[0],
                                          data[tracenumber,minpeaks[0][peakpicker]],
                                          data[tracenumber,peakind[0]])

            #for plotting the line between min and max
            #plt.plot([minpeaks[0][peakpicker],peakind[0]],[data[tracenumber,minpeaks[0][peakpicker]],data[tracenumber,peakind[0]]])
            if worked==0:
                print('trace ',tracenumber,' did not work')
            #this used to be an int:
            nullstellen.append(zero)
            # Zero of signal inbetwenn min and max must be 
            # the last min of its absolute when min and max are crossing zero
            inbetween=np.abs(subsignal)
            #prominence needed because peaks in noisy data are not always clear, sometimes it finds
            #them  on plateaus
            minzeropeaks=find_peaks(inbetween*-1,prominence=inbetween.max()/100)
            
            zeropeakpicker=-1 #last peak index
            #check if minimum within what we are working is lower than the very last we found
            #needs to be as close to zero in the minzeropeaks but not left of the minpeak(negative lobe of reflect)
            while (np.abs(subsignal[minzeropeaks[0][zeropeakpicker]])>np.abs(subsignal[minzeropeaks[0][zeropeakpicker-1]])) and (minzeropeaks[0][zeropeakpicker-1]>minpeaks[0][peakpicker]):
                                zeropeakpicker=zeropeakpicker-1
            sigzero.append(minzeropeaks[0][zeropeakpicker])
            #now if we have this minumum, we could calculate a better zero
            #between the value BEFORE and the value AFTER that minimum to be more accurate
            before_zero=minzeropeaks[0][zeropeakpicker]-1
            after_zero=minzeropeaks[0][zeropeakpicker]+1
            before_zero_y= data[tracenumber,before_zero]
            after_zero_y= data[tracenumber,after_zero]
            accurate_zero,worked=nullstelle(before_zero,after_zero,before_zero_y,after_zero_y) 
            if worked==0:
                print('trace ',tracenumber,
                               ' did not work with values xmin {} xmax{} ymin{} ymax{}'.format(before_zero,after_zero,before_zero_y,after_zero_y))
            
            #we need to check if the baseindex is not super wrong. if it is, we reduce the "height" and try again
            while baseindex > minzeropeaks[0][zeropeakpicker]: 
                print('Detection Problem at trace {}, trying with lower baseline threshold'.format(tracenumber))
                height=height*2/3
                baseindex=find_peaks(data[tracenumber,:],height=height)[0][0]
            sigzeronullstellen.append(accurate_zero)
            baseline.append(baseindex)
           # if tracenumber == 19760: 
           #     print("pause")
           #     plt.plot(data[tracenumber,:])
           #     plt.plot( inbetween)
           #     plt.scatter(minpeaks[0],data[tracenumber,minpeaks[0]])
           #     plt.scatter(minzeropeaks[0],data[tracenumber,minzeropeaks[0]])
           #     plt.plot([minpeaks[0][peakpicker],peakind[0]],[data[tracenumber,minpeaks[0][peakpicker]],data[tracenumber,peakind[0]]])
        # but for empty traces we keep a Nan for now
        else:
            allpeaksf.append(np.nan)
            minpeaksf.append(np.nan)
            nullstellen.append(np.nan)
            sigzero.append(np.nan)
            sigzeronullstellen.append(np.nan)
            baseline.append(np.nan)
    allpeaksf=np.array(allpeaksf)
    minpeaksf=np.array(minpeaksf)
    nullstellen=np.array(nullstellen)
    baseline=np.array(baseline)
    sigzero=np.array(sigzero)
    sigzeronullstellen=np.array(sigzeronullstellen)
    # plt.scatter(np.arange(0,len(np.unique(baseline))),np.unique(baseline))
    # plt.scatter(np.arange(0,len(np.unique(sigzero))),np.unique(sigzero))
    
            
    return allpeaksf+idxadder,minpeaksf+idxadder,nullstellen+idxadder,sigzero+idxadder,sigzeronullstellen+idxadder,baseline+idxadder

def obspytriggers(data,count):
    
    from obspy.signal.trigger import recursive_sta_lta,trigger_onset
    
    triggerpicks=[]
    for i in range(0,count):
        cft = recursive_sta_lta(data[i,:], int(10), int(20))
        triggers=trigger_onset(cft,1.5,0.5)
        #print(triggers)
        if len(triggers)>1:
            triggerpicks.append(triggers[1,0])
        else:
            triggerpicks.append(np.nan)
    return triggerpicks

def energynorm(target,correction):
    '''
    Takes the energy of Target and scales it to match the 
    energy of correction
    uses the square root of that scale to scale the target signal
    
    energy = sum over all L2norms
    '''
    target=target.astype('float64')
    correction=correction.astype('float64')
    
    correction_energy=np.sum(correction ** 2)
    target_energy=np.sum(target ** 2)
    
    energy_quot=target_energy/correction_energy 
    factor=np.sqrt(energy_quot)
    print(factor)
    return target*factor

def normalize(data,maximum=None,minimum=None):
    '''normalizes a given numpy data array'''
    if maximum==None: 
        maximum=data.max()
    if minimum==None: 
        minimum=data.min()
    return (data-data.min())/(data.max()-data.min())
    
    
    
if __name__ == '__main__':

  pass
    
    