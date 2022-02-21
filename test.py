# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 13:37:25 2020

@author: Riedel
"""




import sys
from readreflex.readreflex import radargram
import numpy as np
import matplotlib.pyplot as plt

#for reading seismic formats: 
import os




filepath=r'exampledata\testdata.DAT'
calibdata=radargram()
if os.path.isfile(filepath+'.hdf5'):
    calibdata.load_hdf5(filepath)
else:
    calibdata.read_data_file(filepath,version=9)
    calibdata.save(filepath)
    
    
    calibdata.header['xoffset']=0
    
    
#copy the data    
calibcopy=calibdata.copy()
#upsample it
calibcopy.time_resampling(16*512)
#plot it
calibcopy.radarplot()



