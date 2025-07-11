# -*- coding: utf-8 -*-
"""
readreflex.py

firstly intended as a library to read and export ReflexW-Data-Formats
Reads .PAR and .DAT formats written by that program
https://www.sandmeier-geo.de/reflexw.html


"""

import pandas as pd
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
import sys 
from pathlib import Path

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
#specific scipy packages for heritage to filters: 
from scipy.signal import butter, lfilter, spectrogram, welch,windows,hilbert
from scipy.ndimage import uniform_filter, median_filter
#from librosa.core import reassigned_spectrogram
#from librosa.core import reassigned_spectrogram as ifgram
from tqdm import tqdm
tqdm.pandas()

    
## radargram reader-class
 
class GPRPlotterWidget(QtWidgets.QMainWindow):
    """Standalone plotting widget that can be embedded or used independently"""
    
    def __init__(self, data, title="GPR Array Plotter", parent=None):
        super().__init__(parent)
        self.data = data
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        
    def initUI(self):
        # Central widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        controls = QtWidgets.QHBoxLayout()
        
        # Color scale controls
        self.vmin_input = QtWidgets.QDoubleSpinBox()
        self.vmin_input.setRange(-2000, 2000)
        self.vmin_input.setValue(self.data.min())
        self.vmin_input.setSingleStep(0.1)
        self.vmin_input.valueChanged.connect(self.update_colormap)
        
        self.vmax_input = QtWidgets.QDoubleSpinBox()
        self.vmax_input.setRange(-2000, 2000)
        self.vmax_input.setValue(self.data.max())
        self.vmax_input.setSingleStep(0.1)
        self.vmax_input.valueChanged.connect(self.update_colormap)
        
        # Colormap selection
        self.colormap_combo = QtWidgets.QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis','greys'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        
        # Auto-scale button
        auto_scale_btn = QtWidgets.QPushButton('Auto Scale')
        auto_scale_btn.clicked.connect(self.auto_scale)
        
        controls.addWidget(QtWidgets.QLabel('Min:'))
        controls.addWidget(self.vmin_input)
        controls.addWidget(QtWidgets.QLabel('Max:'))
        controls.addWidget(self.vmax_input)
        controls.addWidget(QtWidgets.QLabel('Colormap:'))
        controls.addWidget(self.colormap_combo)
        controls.addWidget(auto_scale_btn)
        controls.addStretch()
        
        layout.addLayout(controls)
        
        # Plot widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # Image item
        self.plot = self.plot_widget.addPlot()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        
        # Colorbar
        self.colorbar = pg.ColorBarItem(
            values=(self.data.min(), self.data.max()),
            colorMap='viridis'
        )
        self.plot_widget.addItem(self.colorbar)
        
        # Set up the image
        self.update_image()
        
        # Enable mouse interaction
        self.plot.setLabel('left', 'Depth/Time')
        self.plot.setLabel('bottom', 'Trace Number')
        
    def update_image(self):
        """Update the displayed image with current data and scaling"""
        display_data = self.data.T
        self.img.setImage(display_data, autoLevels=False)
        self.img.setLevels([self.vmin_input.value(), self.vmax_input.value()])
        
    def update_colormap(self):
        """Update colormap and color scaling"""
        colormap_name = self.colormap_combo.currentText()
        colormap = pg.colormap.get(colormap_name)
        self.img.setColorMap(colormap)
        self.img.setLevels([self.vmin_input.value(), self.vmax_input.value()])
        self.colorbar.setColorMap(colormap)
        self.colorbar.setLevels((self.vmin_input.value(), self.vmax_input.value()))
        
    def auto_scale(self):
        """Auto-scale to data min/max"""
        vmin, vmax = self.data.min(), self.data.max()
        self.vmin_input.setValue(vmin)
        self.vmax_input.setValue(vmax)
        self.update_colormap()
        
    def update_data(self, new_data):
        """Update the data being displayed"""
        self.data = new_data
        self.vmin_input.setValue(self.data.min())
        self.vmax_input.setValue(self.data.max())
        self.colorbar.setLevels((self.data.min(), self.data.max()))
        self.update_image()
      
   
class radargram():
    
    #initialization to set up overtake of fileformats
    #for later bookkeeping, unused yet
    def __init__(self,fileformat=None):
        print("Initialized Radargram object")
        #some plotting setups: 
        self.plotter = None
        self._app = None
        #this is unfinished and not used
#        if fileformat is not None and fileformat.lower() in {"**r", "**t"}:
#            self.fileformat = "**R"
        
    
#        else:
#            print('No file format was given, assuming .par')
            
   
    def plot_interactive(self,  title="GPR Data"):
        
      """Create interactive plot of GPR data"""
      data=np.flipud(self.traces.T)
      plot_data = data if data is not None else self.data
      if plot_data is None:
          raise ValueError("No data to plot")
          
      # Ensure Qt application exists
      app = QtWidgets.QApplication.instance()
      if app is None:
          app = QtWidgets.QApplication(sys.argv)
      self.plotter = GPRPlotterWidget(plot_data, title=title)
      self.plotter.show()
      return app, self.plotter
  
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
        
        
        #turn path into object
        object_path=Path(filepath)
        #get suffix without care for relative or absolute
        current_suffix = object_path.suffix
        
        #create new suffix
        if current_suffix==".DAT":
            target_suffix=".PAR"
        else:
            target_suffix=current_suffix[0:3]+'R'
        
        #create new file path and reolace parfilefull for opening
        
        parfilefull= object_path.with_suffix(target_suffix)
        
            
        
    
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
        timeincrement=struct.unpack('f', data[464:468])[0]#*timedimension_float
        #print(timeincrement)
        #time for inner working is kept on time dimension
        timebegin=struct.unpack('f', data[472:476])[0]#*timedimension_float
        #print(timebegin)
        #span up a vecotor of timesteps
        timevec=np.arange(0,samplenumber*timeincrement,timeincrement)
        zerosample=int(timebegin/timeincrement)
        #description of the header goes in here
        description= 'Samplenumber: Number of samples in a trace; Zerosample: Sample of the first reflection point ;\n\
                    tracenumber: Number of Traces in the Radargram; \n \
                    formatcode: 2 - 16 bit small integer 3 - 32 bit float ; \n\
                    traceincrement: distance increment; \n\
                    timeincrement: sampling time 1/f;\n\
                    timedimension: the unit of the time; \n\
                    timebegin: Set from processing, time when material in radargram begins ;\n\
                    timevec: Vector of timesteps ;\n\
                    xoffset: X-Profile offset, assumed to be only dimension for now;\n\
                    description: this text'
        
        while True:  # Loop until valid input is received
            try:
                xoffset = float(input("V8 offset and coordinates are not documented and readable, please input a numeric in meters:"))
                break  # Exit the loop if input is valid
            except ValueError:
                print("Invalid input. Please enter a whole number for your offset.")


        
        header={"samplenumber":samplenumber,
                "zerosample":zerosample,
                "tracenumber":tracenumber,
                "formatcode":formatcode,
                "traceincrement":traceincrement,
                "timeincrement":timeincrement,
                "timebegin":timebegin,
                "timedimension":timedimension,
                "time":timevec,
                "xoffset":xoffset,
                "description": description}
        self.header=header    
        
    def read_header_file_v9(self,filepath): 
        
        '''  reads .**T file of the reflexw format and returns a dictionary of the chosen parmeters
        Parameters: 
            filepath -- path to the file(str)
        '''
        print("reading file: " + filepath)
        
        ## NEEDS UNIX/WINDOWS PATH HANDLING
    
        
        #turn path into object
        object_path=Path(filepath)
        #get suffix without care for relative or absolute
        current_suffix = object_path.suffix
        
        #create new suffix
        if current_suffix==".DAT":
            target_suffix=".PAR"
        else:
            target_suffix=current_suffix[0:3]+'R'
        
        #create new file path and reolace parfilefull for opening
        
        parfilefull= object_path.with_suffix(target_suffix)
        
    
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
           # print('Tracenumber appears to be 64000 \n')
           # print("This could be due to numerical issues with the file formats. ")
           # print("Try other data position, possible correct value? Y/N")
           # answer=input()
           # if answer=='y':
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
        timeincrement=struct.unpack('d', data[500:508])[0]#*timedimension_float
        print('Time increment: ',timeincrement, " Seconds" )
        timebegin=struct.unpack('d', data[516:524])[0]#*timedimension_float
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
        description= 'Samplenumber: Number of samples in a trace; Zerosample: Sample of the first reflection point ;\n\
                    tracenumber: Number of Traces in the Radargram; \n \
                    formatcode: 2 - 16 bit small integer 3 - 32 bit float ; \n\
                    traceincrement: distance increment; \n\
                    timeincrement: sampling time 1/f;\n\
                    timedimension: the unit of the time; \n\
                    timebegin: Set from processing, time when material in radargram begins ;\n\
                    timevec: Vector of timesteps ;\n\
                    xoffset: X-Profile offset, assumed to be only dimension for now;\n\
                    description: this text'
                        
        
        header={"samplenumber":samplenumber,
                "zerosample":zerosample,
                "tracenumber":tracenumber,
                "formatcode":formatcode,
                "traceincrement":traceincrement[0],
                "timeincrement":timeincrement,
                "timebegin":timebegin,
                "timedimension":timedimension,
                "time":timevec,
                "xoffset":xoffset,
                "description": description}
        
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
            print("Reading File {}".format(filepath))
            self.bytedata =datdata
            if version==9:
                self.read_header_file_v9(filepath=filepath)
            else:
                self.read_header_file_v8(filepath=filepath)
        self.__convert_to_array()
           
            
    def __readtrace_newformat(self,byteobject):
    # reads a trace in the given byteobject 
    # byteobject should be of Formatcode 3 (32bit floating point int)
        ''' returns array of trace amplitudes and a time ( if formatcode is 3)'''
    
    
        self._TraceNo=int.from_bytes(byteobject[0:4],byteorder='little')
        self.NoOfSamples=int.from_bytes(byteobject[4:8],byteorder='little')
        
        
        #error catching in case header is broke which for some reason happens super often
        if self.NoOfSamples==0 or self.NoOfSamples>1024:
            self.NoOfSamples=self.header['samplenumber']
        tracedata=np.empty(self.NoOfSamples)
    #header takes 158 bytes, always! (at least it should)
        if self.header["formatcode"]==3:
            bytelength=4
            timecollect=struct.unpack('d', byteobject[114:122])[0]
            #test approach: create c type string that represents all of data 
            # instead of looping over byte object
            ctype=self.NoOfSamples*'f'
            tracedata=struct.unpack(ctype, byteobject[158:158+self.NoOfSamples*bytelength])
            t_y,t_x=struct.unpack('2d', byteobject[54:70])
            t_z=struct.unpack('d', byteobject[38:46])[0]
# =============================================================================
#             for j,i in enumerate(np.arange(158,158+self.NoOfSamples*bytelength,bytelength)):
#                     tracedata[j]=struct.unpack('f', byteobject[i:i+bytelength])[0]
#                     #print(tracedata[j])
# =============================================================================
            return tracedata,timecollect,t_x,t_y,t_z
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
        self.timecollect=np.empty(self.header["tracenumber"])
        self.shot_x=np.empty(self.header["tracenumber"])
        self.shot_y=np.empty(self.header["tracenumber"])
        self.shot_z=np.empty(self.header["tracenumber"])
        #NOTE:THIS COULD BE PARALLELIZED FOR FASTER READING
        for i,j in tqdm(enumerate(range(self.header["tracenumber"])),total=self.header["tracenumber"]):
            self.traces[i,:],self.timecollect[i],self.shot_x[i],self.shot_y[i],self.shot_z[i]=self.__readtrace_newformat(self.bytedata[i*_bytetracesize:(i+1)*_bytetracesize])
            #print([i,j])
        dataframe=pd.DataFrame(data={'X':self.shot_x,'Y':self.shot_y,'Z':self.shot_z,'timecollect':self.timecollect})
        self.set_metadata(dataframe)
  
    def save(self,filepath):
        #old h5file=filepath.split(".")[0]+'.hdf5'
        h5file=filepath+'.hdf5'
        with  h5py.File(h5file, 'w') as f:
            dset = f.create_dataset("radargram", data=self.traces, dtype='f',compression="gzip", compression_opts=9)
            for name, value in self.header.items():
                #print(name)
                dset.attrs[name]=value
        #save the metadata                
        if hasattr(self, 'metadataframe'):
                self.metadataframe.to_hdf(filepath+'.hdf5',key="metadataframe")
            

    def set_metadata(self,metadataframe:pd.DataFrame,coordinatenames=["X","Y","Z"]): 
        ''' this sets a metadata frame attached to the radargram
        metadataframe:          A pandas Dataframe
        coordinatenames:        ["X,Y,Z"] Two strings for the coordinates'''
        self.metadataframe=metadataframe
        if(coordinatenames[0] in metadataframe and coordinatenames[1] in metadataframe): 
            print("Found Coordinates in Metaframe")
            if coordinatenames[2] in metadataframe:
               self.coordinates=metadataframe[coordinatenames[0:3]]    
            else:
               self.coordinates=metadataframe[coordinatenames[0:2]]    
        else: 
            print("No coordinates provided")
            
    def set_time(self,sampling_time, sampling_dimension): 
        '''
        

        Parameters
        ----------
        sampling_time : float
            desired sampling interval
        sampling_dimension: str
            sampling dimenstion of sampling interal e.g. s ms ns
        Returns
        -------
        None.

        '''
        #recalculate header time
        self.header['time']=np.linspace(0, self.header["samplenumber"]*sampling_time,self.header["samplenumber"])
        self.header["timedimension"]=sampling_dimension
        self.header["timeincrement"]=sampling_time
        

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
        path=Path(filepath)
        if path.suffix!='.hdf5':
            print("File ending is not .hdf5 , trying to append ending")
            h5file=path.as_posix()+'.hdf5'
        else: 
            h5file=filepath
        
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
                try:
                    _timedimension=dataset.attrs["timedimension"]
                except:
                    print("HDF has no timedmensoon attribute, you might want to check your file version")
                    _timedimension="Not set"
                        
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
                self.header={"samplenumber":_samplenumber,
                             "tracenumber":_tracenumber,
                             "formatcode":_formatcode,
                             "traceincrement":_traceincrement,
                             "timeincrement":_timeincrement,
                             "timedimension":_timedimension,
                             "timebegin":_timebegin,
                             "time":_time,"description":_description,
                             "xoffset":_xoffset}
                print("Found and loaded the file!")
        except:
            print("Seems like there is no HDF5 file present or structure is not according to what's expected, check speeling please")
        #load possible metadata, pandas dataframe
        try:
            self.metadataframe=pd.read_hdf(h5file)
        except: 
            print("Could not read any metadata frame")  
            
            
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
        
    def traceplot(self,tracenumber,timeplot=True,*args,**kwargs): 
        '''
         Plots one or more traces into the same graph
        tracenumber - scalar or array of ints , needs to be a list or np array
        timeplot - Whether the x axis is in time or samples
        '''
        
        if isinstance(tracenumber,(list,np.ndarray))==False:
            print("Error, please submit a list or an (Numpy)array of trace numbers!")
            return
            
        plt.figure()
        for traceno in tracenumber: 
            if timeplot==True:
                plt.plot(self.header["time"],self.traces[traceno,:],linewidth=3,*args,**kwargs)
            else:
                plt.plot(self.traces[traceno,:],linewidth=3,*args,**kwargs)
        if timeplot==True:        
            plt.xlabel("Time")
        else: 
            plt.xlabel("Traces")
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
            self.header["tracenumber"]=end-begin
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
    
        
    def background_removal(self,inplace=False): 
        ''' 
        Subtracts the total horizontal mean from the data
        ''' 
        data=pd.DataFrame(data=self.traces)
        if inplace==True:
            self.traces=self.traces-data.mean(axis=0).values
        else:
            return self.traces-data.mean(axis=0).values
        
    def running_background_removal(self,tracewindow=10,inplace=False):
        ''' 
        Averages the tracewindow (symmetrically) around each trace and subtracts it
        from the traces
        
        '''
        
        
        data=pd.DataFrame(data=self.traces)
        #debugplot
        
        rollingmean=data.rolling(window=tracewindow,min_periods=1,center=True,axis=0).mean()
        if inplace==True:
            self.traces=self.traces-rollingmean.values
        else: 
            return self.traces-rollingmean.values
        
    

    def apply_agc(self, window_size=1,inplace=False):
        """
        Apply a Quick and Dirty   Automatic Gain Control (AGC) filter to GPR data.
        Gain ist based on Hilbert Envelope convolution
        
        Parameters:
            window_size (int): Window size for the AGC (in samples).
            
        Returns:
            numpy.ndarray: AGC filtered data.
        """
        from scipy.signal import hilbert
        data=self.traces
        # Initialize the output array
        agc_data = np.zeros_like(data)
        
        # Loop over each trace
        for trace in range(data.shape[0]):
            # Calculate the amplitude envelope of the trace
            #envelope = np.abs(data[trace,:])
            h_envelope=np.abs(hilbert(data[trace,:]))
            # Apply a moving average to the envelope to get the gain factor
            gain = np.convolve(h_envelope, np.ones(window_size) / window_size, mode='same')
            
            # Avoid division by zero
            gain[gain == 0] = 1e-10
            
            # Apply the gain to the trace
            agc_data[ trace,:] = data[trace,:] / gain
        
        if inplace==True: 
            self.traces=agc_data
        else:
            
            return agc_data
            
    def apply_linear_gain(self, breakpoint_sample=51, slope=0.0672, intercept=1.0, inplace=True):
        '''
        Applies a piecewise linear gain to the GPR traces.
    
        For samples < breakpoint_sample:
            gain = intercept
        For samples >= breakpoint_sample:
            gain = intercept + slope * (sample_index - breakpoint_sample)
    
        Parameters:
            breakpoint_sample (int): The sample index at which the gain begins to increase.
            slope (float): Slope of the linear gain after the breakpoint.
            intercept (float): Gain value before the breakpoint.
            inplace (bool): If True, modifies self.traces in-place. If False, returns a gained copy.
    
        Returns:
            If inplace is False, returns a new NumPy array with gained traces.
        '''
        if not hasattr(self, 'traces'):
            raise AttributeError("Object must have a 'traces' attribute (2D NumPy array).")
        
        n_samples = self.traces.shape[1]
    
        # Build gain vector
        gain = np.full(n_samples, intercept)
        gain[breakpoint_sample:] = intercept + slope * (np.arange(breakpoint_sample, n_samples) - breakpoint_sample)
    
        # Apply gain to each trace
        if inplace:
            self.traces *= gain
        else:
            return self.traces * gain
   
    def apply_2d_average_filter(self, window_size=(3, 3), inplace=True):
        '''
        Applies a 2D moving average filter to the GPR data using a square window.
        
        This smooths both in the time (samples) and trace directions.
    
        Parameters:
            window_size (tuple): Size of the averaging window (default is (3, 3)).
                                 Format is (trace_window, time_sample_window).
            inplace (bool): If True, modifies self.traces in-place. If False, returns a filtered copy.
    
        Returns:
            If inplace is False, returns the filtered array.
        '''
        if not hasattr(self, 'traces'):
            raise AttributeError("Object must have a 'traces' attribute (2D NumPy array).")
    
        # Apply uniform filter (mode='nearest' handles edges without zero-padding)
        filtered = uniform_filter(self.traces, size=window_size, mode='nearest')
    
        if inplace:
            self.traces = filtered
        else:
            return filtered


    
    

    
    def apply_horizontal_filter(
        self, sample_start=136, sample_end=512,
        window_radius=100, use_median=True,
        taper_length=40, inplace=True
    ):
        '''
        Applies a horizontal filter that subtracts the median (or mean) over a running window
        of 201 traces (±100), with a soft taper at the beginning.
        
        Parameters:
            sample_start (int): First sample index to apply filtering (e.g., corresponds to 10 ns).
            sample_end (int): Last sample index to apply filtering (e.g., 54.1 ns).
            window_radius (int): Radius of the trace window (default 100 → total width = 201).
            use_median (bool): If True, use median filter. If False, use mean filter.
            taper_length (int): Number of samples over which to blend in the filter at the start.
            inplace (bool): If True, modifies self.traces in-place. If False, returns filtered copy.
        
        Returns:
            If inplace is False, returns filtered data. Otherwise, modifies self.traces in-place.
        '''
        if not hasattr(self, 'traces'):
            raise AttributeError("Object must have a 'traces' attribute (2D NumPy array).")
    
        data = self.traces
        n_traces, n_samples = data.shape
    
        sample_start = max(0, sample_start)
        sample_end = min(n_samples, sample_end)
        taper_length = min(taper_length, sample_start)  # prevent out-of-bound
    
        output = data.copy() if not inplace else data
    
        # Precompute taper weights using a Hanning window
        if taper_length > 0:
            taper_weights = np.hanning(taper_length * 2)[:taper_length]  # ramp from 0 to 1
        else:
            taper_weights = []
    
        for s in range(sample_start - taper_length, sample_end):
            if s < 0 or s >= n_samples:
                continue
    
            trace_column = data[:, s]
    
            if use_median:
                filtered = median_filter(trace_column, size=2 * window_radius + 1, mode='nearest')
            else:
                window_size = 2 * window_radius + 1
                cumsum = np.cumsum(np.insert(trace_column, 0, 0))
                mean_filtered = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
                pad_left = window_radius
                pad_right = len(trace_column) - len(mean_filtered) - pad_left
                filtered = np.pad(mean_filtered, (pad_left, pad_right), mode='edge')
    
            # Blend using taper if in taper region
            if s < sample_start:
                taper_idx = s-(sample_start-taper_length)
                weight = taper_weights[taper_idx]
                #blend into the data
                output[:,s]=output[:,s]-(weight *filtered)
                #output[:, s] = (1 - weight) * output[:, s] + weight * filtered
            else:
                output[:, s] = output[:,s]-filtered
    
        if not inplace:
            return output

        

    
    def export_csv(self,exportpath='export.csv'): 
        np.savetxt(fname=exportpath,X=self.traces,delimiter=',')
        pieces=exportpath.split('.')
        metafilename=pieces[0]+'_meta.'+pieces[1]
        with open(metafilename, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in self.header.items():
                writer.writerow([key, value])
    
    def export_sgy(self,exportpath='default.sgy',maxtrace='default'):
        #reference: https://github.com/equinor/segyio/blob/master/python/examples/make-file.py
        spec = segyio.spec()
        filename = exportpath
        if maxtrace=='default':
            maxtrace=1000
        else:
            maxtrace=int(self.header['tracenumber'])
        # to create a file from nothing, we need to tell segyio about the structure of
        # the file, i.e. its inline numbers, crossline numbers, etc. You can also add
        # more structural information, but offsets etc. have sensible defautls. This is
        # the absolute minimal specification for a N-by-M volume
        spec.sorting = 2
        spec.format = 5
        spec.samples = range(0,self.header['samplenumber'])
        spec.ilines = range(0,maxtrace)
        spec.xlines = [0]
        tr = 0
        if self.header['timedimension'] == 'ns':
            temp_dt=int(self.header['timeincrement']*1E6)
        elif self.header['timedimension'] == 'ms':
            temp_dt=1
        print("Setting time to ", temp_dt)
        print("Saving {} Traces".format(maxtrace))
        with segyio.create(filename, spec) as f:
            for il in range(maxtrace):
                for xl in spec.xlines:
                    f.header[tr] = {
                        segyio.su.offset : 1,
                        segyio.su.iline  : il,
                        segyio.su.xline  : xl,
                        segyio.su.dt     : temp_dt
                        }
                    f.trace[tr] = self.traces[tr,:]
                    tr += 1
            print("Tr reached ",tr)
            f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
        #set sample time
            f.bin.update(hdt=temp_dt)
        print("Wrote {}".format(exportpath))
    
    def export_sgy(self,exportpath='default.sgy',keyX="X",keyY="Y"):
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
                     if hasattr(self,"metadataframe"): 
                         f.header[tr] = {
                         segyio.su.offset : 1,
                         segyio.su.iline  : il,
                         segyio.su.xline  : xl,
                         segyio.su.dt     : int(self.header['timeincrement']*1000),
                         segyio.su.cdpx: int(self.metadataframe[keyX][tr]*100),
                         segyio.su.cdpy: int(self.metadataframe[keyY][tr]*100),
                         }
                     else:
                         f.header[tr] = {
                         segyio.su.offset : 1,
                         segyio.su.iline  : il,
                         segyio.su.xline  : xl,
                         segyio.su.dt     : int(self.header['timeincrement']*1000)
                         }
                     f.trace[tr] = self.traces[il,:]
                     tr += 1
             f.bin.update(tsort=segyio.TraceSortingFormat.INLINE_SORTING)
             #set sample time
             f.bin.update(hdt=int(self.header['timeincrement']*1000))



