# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:12:07 2025

@author: Riedel
"""


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import re, glob

def parse_ifi_file(ifi_folder,dzx_file_path,lookup_file):
    
    
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
