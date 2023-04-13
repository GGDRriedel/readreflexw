# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:36:39 2020

@author: Riedel

Helper-Functions to Read and Write meta files like .pck 
"""


import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def readpicks(filelocation,nrows=None):

    filename, file_extension = os.path.splitext(filelocation)
    #print(file_extension)
# =============================================================================
#     if file_extension=='.dat':
#         df =pd.read_csv(filelocation,sep='\t',nrows=nrows)
#         return df
#     elif file_extension.lower()=='.pck':
#         #need to check if file is binary or ascii
#         try:
#             with open(filelocation, "r") as f:
#                 for l in f:
#                     print(l)
#         except UnicodeDecodeError:
#             print('Not a text file')
#             pass # Fond non-text data
# =============================================================================
    try: 
        df=pd.read_csv(filelocation,sep='\s+',nrows=nrows,header=None)
        return df
    except:
        print('Not an ASCII file')
    
def readdata(filelocation,dt=0.0585937,start=0,nrows=None):
    
    df=pd.read_csv(filelocation,sep='\s+',header=None,nrows=nrows,engine='python')
    x,y=df.shape
    time=np.linspace(0,(y-1)*dt,y)
    df.columns=time
    return df
    
def readlabelfile(filename): 
    nameparts=filename.split("\\")
    names=[]
    xmins=[]
    ymins=[]
    xmaxs=[]
    ymaxs=[]
    #open the xml
    root = ET.parse(filename).getroot()
    #get the file name
    for name in root.findall("./object/name"):
        names.append(name.text)
    #each bounding box should have coordinates
    for bndbox in root.findall("./object/bndbox"):
        xmins.append(int(bndbox[0].text))
        ymins.append(int(bndbox[1].text))
        xmaxs.append(int(bndbox[2].text))
        ymaxs.append(int(bndbox[3].text))
    #turns out
    filecolumn=len(xmaxs)*[filename]
    if "part" in nameparts[-1]:
        filenoext=os.path.splitext(nameparts[-1])[0]
        fileparams=filenoext.split("_")
        begin=int(fileparams[-2])
        end=int(fileparams[-1])
        print(begin, end)
        xmins=np.array(xmins)
        xmaxs=np.array(xmaxs)
        xmins=xmins+begin
        xmaxs=xmaxs+begin
    
    #data=array([filecolumn,xmins,ymins,xmaxs,ymaxs])
    classifications=pd.DataFrame(data=[filecolumn,xmins,ymins,xmaxs,ymaxs,names])
    classifications=classifications.transpose()
    return classifications
    

def read_dst(dstfilepath):
    data=pd.read_csv(filepath_or_buffer=dstfilepath,delim_whitespace=True,
                        header=None,
                        names=['id','dist','shot_x','shot_y','rec_x','rec_y','shot_z','rec_z'])
    return data
    
   
    
    
    
    
    