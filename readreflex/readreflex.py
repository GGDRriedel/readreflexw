# -*- coding: utf-8 -*-
"""
readreflex.py

firstly intended as a library to read and export ReflexW-Data-Formats
Reads .PAR and .DAT formats written by that program
https://www.sandmeier-geo.de/reflexw.html


"""

import pandas
import struct 
import numpy as np
import h5py
import csv
import matplotlib.pyplot as plt
#for reading seismic formats: 
from obspy import read as obread
from scipy.signal import resample
import copy
import segyio


#specific scipy packages for heritage to filters: 
from scipy.signal import butter, lfilter, spectrogram, welch,windows,hilbert
#from librosa.core import reassigned_spectrogram
#from librosa.core import reassigned_spectrogram as ifgram
from tqdm import tqdm

    
## radargram reader-class
       
   
class radargram():
    
    #initialization to set up overtake of fileformats
    #for later bookkeeping, unused yet
    def __init__(self,fileformat=None):
        print("Initialized Radargram object")
        #this is unfinished and not used
#        if fileformat is not None and fileformat.lower() in {"**r", "**t"}:
#            self.fileformat = "**R"
        
    
#        else:
#            print('No file format was given, assuming .par')
            
    # get some functions for copying itself correctly 
    def copy(self): 
        return copy.copy(self)
    def deepcopy(self):
        return copy.deepcopy(self)
            
   # Unfortunately there are some undocumented changes in the byteorders somewhere along the upgrade-path 
   # so there are different byte-positions of information that are not really clear
    def read_header_file_v8(self,filepath): 
        ''' reads OLD .**R file of the reflexw format and returns a dictionary of the chosen parmeters
        It doesn't matter if you input .**R file or .**T file but **T would be prefered to keep consistency
        
        Keyword arguments: 
            filepath -- str path to the .DAT file
        
        '''
        
        print("reading file: " + filepath)
        
        #check if we are dealing with raw .at and .par files 
        stringlist= filepath.split('.')
        if stringlist[-1]=="DAT":
             parfilefull=stringlist[0]+'.'+"PAR"
        else:
        # get file ending number wich according to documentation can only be 2 digits of numbers + the rest T of the ".dat"
            procnumber=stringlist[-1][-3:-1]
            parfile=stringlist[-2]
    
            parfilefull="."+parfile+'.'+procnumber+'R'
        #parfilefull=parfile+'.PAR'
    
        with open(parfilefull, "rb") as f:
       # Read the whole file at once
            data = f.read()
       #print(data)

        distdict={'METER':1,'CM':1E-2,'MM':1E-3}
        distdimension=data[253:258].decode()
        distdimension_float=distdict[distdimension]
        
         #translate time dimension into numbers
        timedict={'ns':1E-9,'ms':1E-3,'s':1}
        timedimension= data[343:345].decode()
        timedimension_float=timedict[timedimension]
        
        samplenumber=int.from_bytes(data[420:422],byteorder='little')
    
        tracenumber=int.from_bytes(data[452:456],byteorder='little') 
    
        formatcode=int.from_bytes(data[436:440],byteorder='little') 
    
      
        traceincrement=struct.unpack('f', data[460:464])[0]
        #print(traceincrement)
        timeincrement=struct.unpack('f', data[464:468])[0]*timedimension_float
        #print(timeincrement)
        
        timebegin=struct.unpack('f', data[472:476])[0]*timedimension_float
        
        #print(timebegin)
        #span up a vecotor of timesteps
        timevec=np.arange(0,samplenumber*timeincrement,timeincrement)
        
        zerosample=int(timebegin/timeincrement)
        
        #description of the header goes in here
        description= 'Samplenumber: Number of samples in a trace; Zerosample: Sample of the first reflection point; tracenumber: Number of Traces in the Radargram, formatcode: 2 - 16 bit small integer 3 - 32 bit float                     traceincrement: distance increment                     timeincrement: sampling time 1/f                     timebegin: Set from processing, time when material in radargram begins                     timevec: Vector of timesteps                     xoffset: X-Profile offset, assumed to be only dimension for now                    description: this text'
        
        while True:  # Loop until valid input is received
            try:
                xoffset = float(input("V8 offset and coordinates are not documented and readable, please input a numeric in meters:"))
                break  # Exit the loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a whole number for your age.")


        
        header={"samplenumber":samplenumber, "zerosample":zerosample, "tracenumber":tracenumber,"formatcode":formatcode,"traceincrement":traceincrement,
            "timeincrement":timeincrement,"timebegin":timebegin,"timedimension":timedimension,"time":timevec,"xoffset":xoffset,"description": description}
        self.header=header    
        
    def read_header_file_v9(self,filepath): 
        
        '''  reads .**T file of the reflexw format and returns a dictionary of the chosen parmeters
        Parameters: 
            filepath -- path to the file(str)
        '''
        print("reading file: " + filepath)
        
        ## NEEDS UNIX/WINDOWS PATH HANDLING
    
        stringlist= filepath.split('.')
        
        #check if we are dealing with raw .at and .par files 
        if stringlist[-1]=="DAT":
            #check if relative path: 
            if stringlist[0]=='':
                parfilefull='.'+stringlist[1]+'.'+"PAR"
            else:
                parfilefull=stringlist[0]+'.'+"PAR"
        else:
            # get file ending number wich according to documentation can only be 2 digits of numbers + the rest T of the ".dat"
            procnumber=stringlist[-1][-3:-1]
            parfile=stringlist[-2]
    
            parfilefull=parfile+'.'+procnumber+'R'
        #parfilefull=parfile+'.PAR'
    
        with open(parfilefull, "rb") as f:
       # Read the whole file at once
            data = f.read()
      
       # Reading header data is unfortunately quite inconvenient, as the documentation suggests that lengths of the used parameter strings
       # is fixed. upon inspection of the bytes it turns out it's not for all. so while a few fields DO possess the suggested 20 byte length, 
       # others do not
       # so parameters are read here to the best of my ability but are not guaranteed to deliver constant or even usable results
       # for example the first 400 bytes are suggested to be an array of 20 strings of 20 bytes per entry, yet the positioning within that array is off
       # or worse: filled with random uninitialized memory at the time of program execution
       # much sanitization probably needs to be done here
       
        #translate dist dimension into numbers        
        distdict={'METER':1,'CM':1E-2,'MM':1E-3}
        distdimension=data[253:258].decode()
        distdimension_float=distdict[distdimension]
        
        #translate time dimension into numbers
        timedict={'ns':1E-9,'ms':1E-3,'s':1}
        timedimension= data[343:345].decode()
        timedimension_float=timedict[timedimension]
        
        samplenumber=int.from_bytes(data[420:424],byteorder='little')
    
        tracenumber=int.from_bytes(data[424:428],byteorder='little') 
        
        #unfortunately there is some undocumented occurence where when the trace-count is >64000 this value is set to 64000 and
        #the REAL count is written at 484:488
        
        if tracenumber==64000: 
            print('Tracenumber appears to be 64000 \n')
            print("This could be due to numerical issues with the file formats. ")
            print("Try other data position, possible correct value? Y/N")
            answer=input()
            if answer=='y':
                tracenumber=int.from_bytes(data[484:488],byteorder='little',signed=True)
            
    
        formatcode=int.from_bytes(data[452:456],byteorder='little') 
        #formatcode  doesn't work sometimes as it doesn't seem to be set correctly sometimes, error catching: 
        if formatcode==1: 
            print('Error. The formatcode of reflex read as 1, however it can only be 2(new int)\n or 3(new float). Please enter:\n')
            newcode=int(input('Enter an integer 2 or 3:'))
            formatcode=newcode
          #these all change because v9 uses double
        traceincrement=struct.unpack('d', data[492:500])
        print('Trace increment',traceincrement[0]/distdimension_float , " Meters")
        timeincrement=struct.unpack('d', data[500:508])[0]*timedimension_float
        print('Time increment: ',timeincrement, " Seconds" )
        timebegin=struct.unpack('d', data[516:524])[0]*timedimension_float
        print('Time start: ',timebegin)
        
        x_start=struct.unpack('d', data[548:556])[0]
        print("X-coord Start:", x_start)
        y_start=struct.unpack('d', data[556:564])[0]
        print("Y-coord Start:", y_start)
        z_start=struct.unpack('d', data[564:572])[0]
        x_end=struct.unpack('d', data[572:580])[0]
        print("X-coord End:", x_end)
        y_end=struct.unpack('d', data[580:588])[0]
        print("Y-coord End:", y_end)
        
        zerosample=int(timebegin/timeincrement)
        
        xoffset=np.abs(x_end-x_start)
        xoffset=x_start
        
        timevec=np.arange(0,samplenumber*timeincrement,timeincrement)
        description= 'Samplenumber: Number of samples in a trace; Zerosample: Sample of the first reflection point; tracenumber: Number of Traces in the Radargram, formatcode: 2 - 16 bit small integer 3 - 32 bit float                     traceincrement: distance increment                     timeincrement: sampling time 1/f                     timebegin: Set from processing, time when material in radargram begins                     timevec: Vector of timesteps                     xoffset: X-Profile offset, assumed to be only dimension for now                    description: this text'
                        
        
        header={"samplenumber":samplenumber, "zerosample":zerosample, "tracenumber":tracenumber,"formatcode":formatcode,"traceincrement":traceincrement[0],
            "timeincrement":timeincrement,"timebegin":timebegin,"timedimension":timedimension,"time":timevec,"xoffset":xoffset,"description": description}
        self.header=header    
         
    
    def read_data_file(self,filepath,version=8):
        ''' 
         reads hole binary file as a bytes object, reads the header file
         converts the bytedata to an array
     
     
         If version=8, old formats
         version=9: new formats'''
         
        with open(filepath, "rb") as f:
            # Read the whole file at once
            datdata = f.read()
            #print(data)
            self.bytedata =datdata
            if version==9:
                self.read_header_file_v9(filepath=filepath)
            else:
                self.read_header_file_v8(filepath=filepath)
        self.__convert_to_array()
           
            
    def __readtrace_newformat(self,byteobject):
    # reads a trace in the given byteobject 
    # byteobject should be of Formatcode 3 (32bit floating point int)
    
    
        self._TraceNo=int.from_bytes(byteobject[0:4],byteorder='little')
        self.NoOfSamples=int.from_bytes(byteobject[4:8],byteorder='little')
        
        
        #error catching in case header is broke which for some reason happens super often
        if self.NoOfSamples==0 or self.NoOfSamples>1024:
            self.NoOfSamples=self.header['samplenumber']
        tracedata=np.empty(self.NoOfSamples)
    #header takes 158 bytes, always! (at least it should)
        if self.header["formatcode"]==3:
            bytelength=4
            for j,i in enumerate(np.arange(158,158+self.NoOfSamples*bytelength,bytelength)):
                    tracedata[j]=struct.unpack('f', byteobject[i:i+bytelength])[0]
                    #print(tracedata[j])
            return tracedata
        else:
            bytelength=2
            for j,i in enumerate(np.arange(156,156+self.NoOfSamples*bytelength,bytelength)):
                    tracedata[j]=struct.unpack('h', byteobject[i:i+bytelength])[0]
                    #tracedata[j]=struct.unpack('h', byteobject[i:i+bytelength])
                    #print(tracedata[j])
            return tracedata
   
    def __convert_to_array(self):
        if self.header["formatcode"]==3:
            _bytetracesize=158+self.header['samplenumber']*4
        else:
            _bytetracesize=156+self.header['samplenumber']*2
            
        self.traces=np.empty([self.header["tracenumber"],self.header["samplenumber"]])
        for i,j in enumerate(range(self.header["tracenumber"])):
            self.traces[i,:]=self.__readtrace_newformat(self.bytedata[i*_bytetracesize:(i+1)*_bytetracesize])
            #print([i,j])
    def save(self,filepath):
        #old h5file=filepath.split(".")[0]+'.hdf5'
        h5file=filepath+'.hdf5'
        with  h5py.File(h5file, 'w') as f:
            dset = f.create_dataset("radargram", data=self.traces, dtype='f',compression="gzip", compression_opts=9)
            for name, value in self.header.items():
                #print(name)
                dset.attrs[name]=value

    def load_seismics(self,filepath): 
        ''' Reads all kinds of seismic file formats inherited from \\
    the obspy library
    Formatcode is set to 3 automatically for legacy purpose
    Traceincrement is set to 1 automatically for legacy purpose
    Timebegin is set to 0 automatically for legacy purpose
    
     Keyword arguments:
         filepath-- the file to read
    '''
        # read
        traceobject=obread(filepath)
        #create array of Dimensions (samples, tracenumber)
        data=np.empty((len(traceobject.traces),(traceobject[0].stats.npts)))
        #get the data from the traces
        for i in range(len(traceobject.traces)):
            data[i,:]=traceobject.traces[i].data
        self.traces=data
        _time=np.arange(0,traceobject[0].stats.npts*traceobject.traces[0].stats.delta*1E9,traceobject.traces[0].stats.delta*1E9)
        header={"samplenumber": traceobject[0].stats.npts, "tracenumber":len(traceobject.traces),"formatcode": 3,"traceincrement": 1,
                        "timeincrement":traceobject.traces[0].stats.delta,"timebegin": 0,"time":_time,"description": "Loaded from Seismic Format"}
        self.header=header 
        
    def load_hdf5(self,filepath):       
        ''' Loads HDF5 files'''
        #h5file=filepath.split(".")[0]+'.hdf5'
        h5file=filepath+'.hdf5'
        print("Looking for "+h5file)
        try:
            with  h5py.File(h5file, 'r+') as f:
                # get the raw number data
                dataset=f.get('radargram')
                self.traces= np.array(dataset)
                #get the data for the dictionary
            
                _samplenumber=dataset.attrs['samplenumber']
                _tracenumber=dataset.attrs['tracenumber']
                _formatcode=dataset.attrs['formatcode']
                _traceincrement=dataset.attrs['traceincrement']
                _timeincrement=dataset.attrs['timeincrement']
                _timebegin=dataset.attrs['timebegin']
                _time=dataset.attrs['time']
                _description=dataset.attrs['description']
                #offset does not always exist so we need to make sure to set it
                #by hand
                while True:
                    try: 
                        _xoffset=dataset.attrs['xoffset']
                        break
                    except KeyError:
                        while True:
                            try:
                                dataset.attrs['xoffset']=int(input("Found no X-Offset in Dataset, please enter the offset in Meters: "))
                                break
                            except ValueError:
                                print("Error! This doesn't seem to be an Integer Number, try again!" )
            #timevec=np.arange(0,headerdata["samplenumber"]*headerdata["traceincrement"],headerdata["timeincrement"])
                self.header={"samplenumber":_samplenumber, "tracenumber":_tracenumber,"formatcode":_formatcode,"traceincrement":_traceincrement,
                        "timeincrement":_timeincrement,"timebegin":_timebegin,"time":_time,"description":_description,"xoffset":_xoffset}
                print("Found and loaded the file!")
        except:
            print("Seems like there is no HDF5 file present or structure is not according to what's expected, check speeling please")
          
    def get_output_data(self,filename, rxnumber, rxcomponent,xstep):
        '''   
        reads the output data of GPRmax-Simulations
        ''' 
    # Open GPRMax  output file and read some attributes
        f = h5py.File(filename, 'r')
        nrx = f.attrs['nrx']
        dt = f.attrs['dt']

    # Check there are any receivers
        if nrx == 0:
            raise CmdInputError('No receivers found in {}'.format(filename))

        path = '/rxs/rx' + str(rxnumber) + '/'
        availableoutputs = list(f[path].keys())

     # Check if requested output is in file
        if rxcomponent not in availableoutputs:
            raise CmdInputError('{} output requested to plot, but the available output for receiver 1 is {}'.format(rxcomponent, ', '.join(availableoutputs)))

        outputdata = f[path + '/' + rxcomponent]
        outputdata = np.array(outputdata)
        f.close()
        
        self.traces= outputdata.T
        _samplenumber=outputdata.shape[0]
        _tracenumber=outputdata.shape[1]
        _traceincrement=xstep
        _timeincrement=dt
        _timebegin=0
        _time =np.arange(dt,(_samplenumber)*dt,dt)
        _description='Dataset derived from gprMax .out File'
        _xoffset=0
        self.header={"samplenumber":_samplenumber, "tracenumber":_tracenumber,"formatcode":0,"traceincrement":_traceincrement,
                        "timeincrement":_timeincrement,"timebegin":_timebegin,"time":_time,"description":_description,"xoffset":_xoffset}
        
        return outputdata, dt
    
    def radarplot(self,contrast=4,short=True,distanceplot=False,perc=98):
        ''' Plots the Radargram
        contrast(default 4) -- sets the max/min values of the colorbar to 1/contrast 
        short(default True) -- only displays the timeframe from the beginning of the starttime parameter
        distanceplot(default False) -- plots radardgra as distance
        per(default 98) -- Percentile for plot colourbar
        '''
        minmax=np.percentile(self.traces,perc)
        
        minimum=-1.*minmax
        maximum=minmax
        # find the next index in the time vector 
        #https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        if short==True: 
            
            topindex=find_nearest(self.header["time"],np.abs(self.header["timebegin"]))
            starttime=topindex*self.header["timeincrement"]
           # print(self.header["timebegin"])
            #print(starttime)
        else:
            topindex=0
            starttime=0
        
        tmax=self.header["timeincrement"]*self.header["samplenumber"]
        fig=plt.figure()
        if distanceplot==False:
            image=plt.imshow(self.traces.T[topindex:-1,:],aspect='auto',extent=[0,self.header["tracenumber"],tmax,starttime],interpolation='none',vmin=minimum/contrast,vmax=maximum/contrast)
            plt.xlabel("Tracenumber")
        else:
            xmin=self.header['xoffset']
            xmax=self.header['tracenumber']*self.header['traceincrement']
            image=plt.imshow(self.traces.T[topindex:-1,:],aspect='auto',extent=[xmin,xmax,tmax,starttime],interpolation='none',vmin=minimum/contrast,vmax=maximum/contrast)
            plt.xlabel("Distance [m]")
        plt.ylabel("t")
        
        cbar=plt.colorbar()
        cbar.set_label("Amplitude")
        ax = plt.gca()
        #ax.set_ylabel([self.header["timeincrement"]*self.header["samplenumber"],0])
        
        # Picking and plotting
        
        def onpick1(event):
             print(event.artist)
        fig.canvas.mpl_connect('pick_event', onpick1)
        return fig,ax
        
    def traceplot(self,tracenumber,*args,**kwargs): 
        '''
         Plots one or more traces into the same graph
        tracenumber - scalar or array of ints , needs to be a list or np array
        '''
        
        if isinstance(tracenumber,(list,np.ndarray))==False:
            print("Error, please submit a list or an (Numpy)array of trace numbers!")
            return
            
        plt.figure()
        for traceno in tracenumber: 
            plt.plot(self.header["time"],self.traces[traceno,:],linewidth=3,*args,**kwargs)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
    
    def spectroplot(self,tracenumber,nperseg=32):
        fs=1.0/self.header["timeincrement"]
        
        
        signal=self.traces[tracenumber,:]
        time=self.header['time']
        time_to_append=np.arange((int(nperseg/2)+1)*self.header["timeincrement"]*-1,time[0],self.header["timeincrement"])
        time=np.append(time_to_append,time)
        time=time+np.abs(time[0])
        signal=np.append(np.zeros(int(nperseg/2)+1),signal)
        m=len(signal)
        nfft=m
        print(m)
        
        f, t, Sxx = spectrogram(signal, fs,nperseg=nperseg,nfft=nfft,noverlap=nperseg-2,scaling='spectrum')
        # find the maximuns
        maxima=Sxx.argmax(axis=0)
        maxima=f[maxima]
        #plt.pcolormesh(t, f, Sxx)
        fig,(ax_signal,ax_spectro)=plt.subplots(2,1,sharex=True)
        ax_signal.plot(time,signal)
        spec=ax_spectro.pcolormesh(t,f,np.log(Sxx),shading='auto')
        ax_spectro.scatter(t,maxima)
        ax_spectro.set_ylabel("Frequency in 1/dt Hz")
        ax_spectro.set_xlabel("Time in sample dimension")
        ax_spectro.set(title="Mean maximum frequency: "+ str(np.mean(maxima)))
        #ax_spectro.set_xlim([t[0],t[-1]])
        
        
       # fig.colorbar(spec, orientation='vertical',label='Log10[Spec]')
       # plt.figure()
       # plt.pcolormesh(t, f, Sxx)
        #plt.show()
       # plt.figure()
        #for i in range(int(m/32)):
         #   windowsig=self.traces[tracenumber,i*32:(i+1)*32]
          #  windowwelch=welch(windowsig,fs=fs,nperseg=16,noverlap=15)
           # plt.plot(windowwelch)
           

    def freq_times_powersum(self,tracenumber,nperseg=64):
        
        '''
         Gives Back the frequency with the maximum amplitude along the time axis multiplied by the cumsum of the power of the trace
         Input:   Tracenumber -- Int Desired trace to treat
                   nperseg    -- Int Sample window for the fft 
        '''
        #Sampling frequency 
        fs=1.0/self.header["timeincrement"]
                 
        signal=self.traces[tracenumber,:]
        
        #we append some values to avoid hard corers at edges of the signal
        time=self.header['time']
        time_to_append=np.arange((int(nperseg*2))*self.header["timeincrement"]*-2,time[0],self.header["timeincrement"])
        time=np.append(time_to_append,time)
        time=time+np.abs(time[0])

        signal=np.append(np.zeros(int(nperseg*2)),signal)
        m=len(signal)
        # nfft window of 10 times signal length turned out good
        nfft=10*m
        #print(m)
        
        f, t, Sxx = spectrogram(signal, fs,nperseg=nperseg,nfft=nfft,noverlap=nperseg-1,scaling='spectrum')
        # find the maxima
        arg_spec_max=Sxx.argmax(axis=0)
        max_frequ_at_each_timestep=f[arg_spec_max]
        # last value of cumsum function is sum over all
        # might be an error here, do we use the LAST value or the one where the maximum is
        cumsumpower=np.cumsum(np.square(Sxx),axis=0)
        power_at_frequ=np.zeros(max_frequ_at_each_timestep.shape)
        # get the cumsums at the max frequency indexes for each timestep: 
        cumsum_at_max_frequ=np.take(cumsumpower,arg_spec_max[0],axis=0) 
        cumsumpower_at_frequ_max_times_frequmax= cumsum_at_max_frequ* max_frequ_at_each_timestep
        #written as a loop:
        #for i in range(len(max_frequ_at_each_timestep)):
        #    power_at_frequ[i]=max_frequ_at_each_timestep[i]*cumsumpower[arg_spec_max[i],i]
            
        #plt.pcolormesh(t, f, Sxx)
        # fig,(ax_signal,ax_spectro)=plt.subplots(2,1,sharex=True)
        #ax_signal.plot(time,signal)
        #spec=ax_spectro.pcolormesh(t,f,np.log(Sxx))
        #ax_spectro.scatter(t,maxima)
        
        #fig.colorbar(spec, orientation='horizontal',label='Log10[Spec]')
        return  cumsumpower_at_frequ_max_times_frequmax[nperseg+1:]
# =============================================================================
#     
#     def spectrosum(self,tracenumber,f_max,f_min,nperseg=64):
#         
#         '''
#          Gives Back the powersum up until fmax divided by the powersum up to fmin of each trace 
#          Input:     Tracenumber -- Int              -- Trace to calculate at 
#                     fmax        -- Int              -- Maximum Frequency
#                     fmin        -- Int              -- Minimum Frequency
#                     nperseg     -- Int Default 64   -- FFT Segments
#                     
#                     All Freqs at 1/sampletime Frequencies please.
#                     Example: t_s= 1s -> f at n Hz please
#         '''
#         #sanity check: 
#         
#         if f_max< f_min: 
#             temp=f_max
#             f_max=f_min
#             f_min=temp
#             print('You swapped Min and Max, this was corrected')
#             
#         def find_nearest(array, value):
#             array = np.asarray(array)
#             idx = (np.abs(array - value)).argmin()
#             return array[idx],idx
#          
#         fs=1.0/self.header["timeincrement"]
# #        nperseg=256
#         
#         signal=self.traces[tracenumber,:]
#         
#         
#         time=self.header['time']
#         time_to_append=np.arange((int(nperseg*2))*self.header["timeincrement"]*-2,time[0],self.header["timeincrement"])
#         time=np.append(time_to_append,time)
#         time=time+np.abs(time[0])
#         signal=np.append(np.zeros(int(nperseg*2)),signal)
#         m=len(signal)
#         nfft=10*m
#         #print(m)
#         
#         f, t, Sxx = spectrogram(signal, fs,nperseg=nperseg,nfft=nfft,noverlap=nperseg-1,scaling='spectrum')
#         
#         # find where to sum to
#         freq_min_idx=find_nearest(f,f_min)
#         freq_max_idx=find_nearest(f,f_max)
# 
#         powerfmax=np.cumsum(np.square(Sxx[0:freq_max_idx[1],:]),axis=0)[-1,:]
#         powerfmin=np.cumsum(np.square(Sxx[0:freq_min_idx[1],:]),axis=0)[-1,:]
#        # plt.pcolormesh(t, f, Sxx)
#         #plt.title('Fmin:{} Fmax:{}'.format(f[freq_min_idx[1]],f[freq_max_idx[1]]))
#        # fig,(ax_signal,ax_spectro)=plt.subplots(2,1,sharex=True)
#         #ax_signal.plot(time,signal)
#         #spec=ax_spectro.pcolormesh(t,f,np.log(Sxx))
#         #ax_spectro.scatter(t,maxima)
#         
#         #fig.colorbar(spec, orientation='horizontal',label='Log10[Spec]')
#         #return np.divide((powerfmaSx-powerfmin),powerfmax)
#         division=np.divide(powerfmax,powerfmin)
#         division=np.nan_to_num(division)
#         return division
#         
#     
#     def powersum(self, tracenumber,end=0):
#         ''' Gives Back the sum of the squared values of the trace'''
#         if end==0: end=self.header["samplenumber"]
#         signal=self.traces[tracenumber,0:end]
#         signalsq=np.square(signal)
#         sqsum=np.sum(signalsq)
#         return sqsum
#         
#         
#     def powerfuncs(self,tracenumber,refspec,powerwindow=64,nperseg=64):
#         
#         ''' gives back a normed value for the spectral content of the trace'''
#         
#         fs=1.0/self.header["timeincrement"]
#         #segment length in which to look at the spectrum 
#         #nperseg=128
#         
#      #   dfsignal=pandas.DataFrame(data=self.traces)
#      #   dfsignal.mean(axis=1)
#         signal=self.traces[tracenumber,:]
#         
#         
#         time=self.header['time']
#         
#         # append some zeros in the beginning to padd the fft
#         
#         time_to_append=np.arange((int(nperseg*2))*self.header["timeincrement"]*-2,time[0],self.header["timeincrement"])
#         time=np.append(time_to_append,time)
#         time=time+np.abs(time[0])
#         signal=np.append(np.zeros(int(nperseg*2)),signal)
#         m=len(signal)
#         nfft=m
#         #print(m)
#         
#         f, t, Sxx = spectrogram(signal, fs=fs,nperseg=nperseg,nfft=nfft,noverlap=nperseg-1,scaling='spectrum')
#         print("Summing up to f={} Ghz".format(f[powerwindow]/1E9))
#         # find the maximuns
#         maxima_ind=Sxx.argmax(axis=0)
#         dfSxx=pandas.DataFrame(data=Sxx)
#         power=np.cumsum(np.square(Sxx[0:powerwindow,:]),axis=0)[-1,:]
#         powerall=np.cumsum(np.square(Sxx),axis=0)[-1,:]
#         powerref=np.cumsum(np.square(refspec),axis=0)[-1,:]
#         
#         maxima=f[maxima_ind]
#        # if tracenumber%200==0 and tracenumber!=0:
#       #  +  #plt.figure()
#            #plt.pcolormesh(t, f, Sxx)
#         #    fig,(ax_signal,ax_spectro,ax_power)=plt.subplots(3,1,sharex=True)
#          #   ax_signal.plot(time,signal)
#            # spec=ax_spectro.pcolormesh(t,f,np.log(Sxx))
#            # ax_spectro.scatter(t,maxima)
#            # ax_power.scatter(t,np.divide(power,powerref))
#            # plt.title('Test Trace{}'.format(tracenumber))
#         
#         #fig.colorbar(spec, orientation='horizontal',label='Log10[Spec]')
#         
#         return np.nan_to_num(np.divide(power,powerref))
#     
#     def weighted_frequency(self,tracenumber,nperseg=64)    :
#         '''
#         Calculates the Weighted Instantaneous Signal
#         https://ieeexplore.ieee.org/document/721370
#         
#         returns array of seize of radargram data
#         '''
#         
#         fs=1.0/self.header["timeincrement"]
#         signal=self.traces[tracenumber,:]
#         time=self.header['time']
#         time_to_append=np.arange((int(nperseg*2))*self.header["timeincrement"]*-2,time[0],self.header["timeincrement"])
#         time=np.append(time_to_append,time)
#         time=time+np.abs(time[0])
#         signal=np.append(np.zeros(int(nperseg*2)),signal)
#         
#         
# # =============================================================================
# #         #
# #         analytic_signal = hilbert(signal)
# #         amplitude_envelope = np.abs(analytic_signal)
# #         instantaneous_phase = np.unwrap(np.angle(analytic_signal))
# #         instantaneous_frequency = (np.diff(instantaneous_phase) /(2.0*np.pi) * fs)
# #         instantaneous_frequency=np.append(instantaneous_frequency,instantaneous_frequency[-1])
# #         asq=amplitude_envelope*amplitude_envelope
# #         asq_f=amplitude_envelope*instantaneous_frequency
# # =============================================================================
#         
#         m=len(signal)
#        # return(np.cumsum(asq_f)/asq.sum())
# 
#         nfft=m
#         f, t, mag = spectrogram(signal, fs=fs,nperseg=nperseg,nfft=nfft,noverlap=nperseg-1,scaling='spectrum',mode='magnitude',window=('gaussian',20))
#         #sum over magsquared times instant frequency
#         
#         magsq=mag*mag # this was squared before but spectrogram already gives the squared
#         magsqsum=magsq.sum(axis=0)
#         magsum=(magsq.T*f).T.sum(axis=0)
#         wif=np.divide(magsum,magsqsum)
#         return np.nan_to_num(wif)
# 
#         #return instantaneous_frequency
#     
#     def ifgram_frequency(self,tracenumber,nperseg=64)    :
#         '''
#         Calculates the Ifgram Weighted Instantaneous Signal
#         https://ieeexplore.ieee.org/document/721370
#         
#         returns array of seize of radargram data
#         '''
#         
#         fs=1.0/self.header["timeincrement"]
#         signal=self.traces[tracenumber,:]
#         time=self.header['time']
#         time_to_append=np.arange((int(nperseg*2))*self.header["timeincrement"]*-2,time[0],self.header["timeincrement"])
#         time=np.append(time_to_append,time)
#         time=time+np.abs(time[0])
#         signal=np.append(np.zeros(int(nperseg*2)),signal)
#         m=len(signal)
#         nfft=m
#         ifg,S,_=reassigned_spectrogram(signal,sr=fs,n_fft=2*nfft,win_length=nperseg-1)
#         return ifg
# 
#         #return instantaneous_frequency        
# =============================================================================
    def shortening(self,begin=0,end=200):
        '''shorts itself to traces begin - end
        begin --- start trace of shortened radargram
        defend --- default end trace
        Optional: end: overwrites shortend
        '''
        if end > self.header["tracenumber"]:
            end=self.header["tracenumber"]
            print('End trace larger than radargram trace-no., changing nothing')
            return
        else: 
            self.header["tracenumber"]=end
            self.traces=self.traces[begin:end,:]
    
    def time_resampling(self,target):
        '''
            Resamples the contained traces to "target"
            number of samples
        '''
        
        self.traces=resample(self.traces,target,axis=1)
        #need to change all time parameters in the header
        newtincrement=self.header["samplenumber"]*self.header["timeincrement"]/target
        self.header["samplenumber"]=target
        self.header["time"]=np.arange(self.header["time"][0],target*newtincrement+self.header["time"][0],newtincrement)
        self.header["timeincrement"]=newtincrement
        
    def time_shortening(self,start:int,end:int): 
        '''
            shortens the radargram data to the sample range start end
        '''
        n,m=self.traces.shape
        if start<0 or start >= end or start >m or end>m: 
            print('Something wrong withs start and end')
            return
        
        #print(n,m)
        self.traces=self.traces[:,start:end]
        self.header["samplenumber"]=end-start
        self.header["time"]=self.header["time"][start:end]
   
    def add_radargram(self,new_radargram): 
        '''
        adds another radargram to the end
        useful for fusing calibration data
        Input: object of type radargram 
        '''
        try:
            type(new_radargram)=='readreflex.readreflex.radargram'
        except:
            print('error, wrong file type, please provide radargram')

        #get dimensions of the data to add
        tracecount,samplecount = new_radargram.traces.shape   
        #check if they match up with base radargram sampling
        try:
            self.traces.shape[1]==samplecount
        except:
            print('Input Radargram has a different sample count')
            
        #fuse the data
        else:
            self.traces=np.concatenate((self.traces,new_radargram.traces),axis=0)
            self.header['tracenumber']=self.traces.shape[0]
        
    def return_envelope(self,*args):
        ''' Returns the envelope of the contained traces
        through uses of scipy analytic signal (Hilbert Transform)
        and its absolute
        checks for existence of traces in the object
        Check for single trace or all traces needs to implemented
        '''
                
        try: 
            assert self.traces.shape, "Traces not initialized, try again"
        except AssertionError as error: 
            print(error)  
        else:
            envelope=np.zeros_like(self.traces)
            for i,trace in tqdm(enumerate(self.traces)): 
                analytic_signal = hilbert(trace)
                envelope[i,:]=np.abs(analytic_signal)
                
            return envelope

    def pad(self,beginning:int,ending:int,method='edge',inplace=True): 
        ''' Pads the traces with their edge, wraps np.pad'''
        try: 
            assert self.traces.shape, "Traces not initialized, try again"
        except AssertionError as error: 
            print(error)  
        else:
            if inplace==True:
                self.traces=np.pad(self.traces,((0,0),(beginning,ending)),mode=method)
                self.header["samplenumber"]=self.header["samplenumber"]+beginning+ending
                self.header["time"]=np.arange(self.header["time"][0],self.header["samplenumber"]*self.header["timeincrement"],self.header["timeincrement"])
            else:
                copy=self.deepcopy()
                copy.traces=np.pad(copy.traces,((0,0),(beginning,ending)),mode=method)
                copy.header["samplenumber"]=copy.header["samplenumber"]+beginning+ending
                copy.header["time"]=np.arange(copy.header["time"][0],copy.header["samplenumber"]*copy.header["timeincrement"],copy.header["timeincrement"])
                return copy
            
    def gssi_fir_filter(self,filterlength,filtertime,filterfile)    :
        ''' Applies a custom fir filter  , filterfile should be provided by user
        
        can be used for some GSSI systems
        
        filterlength:   n
        fltertime:      Time equivalent of n filterparameters in filterfile
        flterfile:      filterparameters to apply
        '''
        old_length=self.header['samplenumber']
        custom_filter=np.genfromtxt(filterfile,skip_header=13,delimiter=',')[:,0]
        targetlength=np.round(self.header["time"][-1]*filterlength/filtertime)
        self.time_resampling(int(targetlength))
        old_resample_length=self.header['samplenumber']
        sample_rate = 1/self.header["timeincrement"]
        nsamples = self.header["samplenumber"]

        t = self.header["time"]
        
        addnumber=len(custom_filter)
        target_nsamples=nsamples+addnumber

        self.pad(beginning=addnumber,ending=0,method='edge')

        t=np.arange(0,target_nsamples*self.header["timeincrement"],self.header["timeincrement"])    
        N=len(custom_filter)
        taps = custom_filter
        self.traces = lfilter(taps, 1, self.traces,axis=1)
        self.time_shortening(N,N+old_resample_length)
        
    def export_csv(self,exportpath='export.csv'): 
        np.savetxt(fname=exportpath,X=self.traces,delimiter=',')
        pieces=exportpath.split('.')
        metafilename=pieces[0]+'_meta.'+pieces[1]
        with open(metafilename, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in self.header.items():
                writer.writerow([key, value])
    
    def export_sgy(self,exportpath='default.sgy'):
        #reference: https://github.com/equinor/segyio/blob/master/python/examples/make-file.py
        spec = segyio.spec()
        filename = exportpath
        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 2
        spec.format = 5
        spec.samples = range(0,self.header['samplenumber'])
        spec.ilines = range(0,self.header['tracenumber'])
        spec.xlines = [0]
        tr = 0
        with segyio.create(filename, spec) as f:
            for il in spec.ilines:
                for xl in spec.xlines:
                    f.header[tr] = {
                        segyio.su.offset : 1,
                        segyio.su.iline  : il,
                        segyio.su.xline  : xl,
                        segyio.su.dt     : int(self.header['timeincrement'])
                        }
                    f.trace[tr] = self.traces[il,:]
                    tr += 1
            f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
            #set sample time
            f.bin.update(hdt=int(self.header['timeincrement']))

