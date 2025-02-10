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

import geopandas as gpd
from shapely.geometry import Point
import re, glob

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
    
   
    
def parse_ifi_file(ifi_folder,dzx_file_path,lookup_file):
    """
   Parses an IFI file associated with a given DZX file and returns a GeoDataFrame of extracted data.
   
   The function:
   1. Reads the lookup Excel file to find the corresponding IFI file for the given DZX file.
   2. Extracts the numeric part of the DZX filename to locate its matching entry in the lookup data.
   3. Reads the IFI file and extracts relevant metadata and numerical data.
   4. Constructs a GeoDataFrame containing IMU and GPS data.

   Parameters:
   ----------
   ifi_folder : str
       Path to the folder containing IFI files.
   dzx_file_path : str
       Full path to the DZX file, used to find the corresponding IFI file.
   lookup_file : str
       Path to an Excel file containing a lookup table mapping DZX files to IFI files.

   Returns:
   -------
   gdf : geopandas.GeoDataFrame
       A GeoDataFrame containing the parsed data, with columns:
       - 'flag': Flag indicator from the file.
       - 'time_since_imu_start': Time (float).
       - 'roll': IMU roll angle (float).
       - 'pitch': IMU pitch angle (float).
       - 'yaw': IMU yaw angle (float).
       - 'distance': Distance measurement (float).
       - 'gps_lat': Latitude (float).
       - 'gps_long': Longitude (float).
       - 'second_of_week': GPS time in seconds of the week (float).
       - 'hdop': Horizontal dilution of precision (float).
       - 'gps_z': GPS altitude (float).
       - 'geometry': Shapely Point geometry (longitude, latitude).

   Notes:
   ------
   - Assumes the lookup Excel file contains a column 'GSSI' with DZX filenames.
   - Uses EPSG:4326 (WGS 84) coordinate reference system for spatial data.
   - Skips lines starting with ';' except for extracting specific metadata fields.

   Example:
   --------
   >>> gdf = parse_ifi_file("path/to/ifi_folder", "path/to/FILE____042.DZG", "path/to/lookup.xlsx")
   >>> print(gdf.head())
   """
    
    
    
    lookup_data=pd.read_excel(lookup_file)
    dzxfile=dzx_file_path.split('\\')[-1]
    # find the dzg number in the lookup df
    file_number = dzxfile.split('FILE____')[1].split('.DZG')[0]
    index = lookup_data[lookup_data['GSSI'].str.contains(f"FILE____{file_number}\.DZG")].index
    
    if not index.empty:
        print(f"Index of match: {index[0]}")
    else:
        print("No match found.")
        
        
    file_path=ifi_folder+r'\\'+str(lookup_data.iloc[index[0],1])
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    metadata = {}
    
    for line in lines:
        line = line.strip()
        
        if line.startswith(';Datum:'):
            metadata['date'] = line.split(': ')[1]
        elif line.startswith(';Timecode='):
            metadata['timecode'] = line.split('=')[1]
        elif line.startswith(';'):
            continue  # Skip other metadata lines
        else:
            parts = re.split(r'\s+', line)
            if len(parts) == 11:
                flag, time, roll, pitch, yaw, dist, lon, lat, sec_week, hdop, gps_z = parts
                data.append([
                    flag, float(time), float(roll), float(pitch), float(yaw), float(dist), 
                    float(lat), float(lon), float(sec_week), float(hdop), float(gps_z)
                ])
    
    df = pd.DataFrame(data, columns=[
        'flag', 'time_since_imu_start', 'roll', 'pitch', 'yaw', 'distance',
        'gps_lat', 'gps_long', 'second_of_week', 'hdop', 'gps_z'
    ])
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df['gps_long'], df['gps_lat'])], crs='EPSG:4326')
    
    return gdf
    
    
    
    