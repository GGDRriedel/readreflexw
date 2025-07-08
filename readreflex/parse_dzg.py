# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 13:54:03 2025

@author: Riedel
"""


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.interpolate import CubicSpline,interp1d

from readreflexw.readreflex.utils import apply_kalman_filter


def parse_gps_data(file_path):
    """
    Parses GPS data from a text file with $GNGGA lines.

    Args:
        file_path (str): Path to the text file containing GPS data.

    Returns:
        pd.DataFrame: A pandas dataframe of parsed GPS information.
    """
    gps_data = []

    with open(file_path, 'r') as file:
        previous_line = None
        for line in file:
            if previous_line and previous_line.startswith("$GSSIS"):
                previous_fields = previous_line.split(',')
                fields = line.split(',')
                
                gps_entry = {
                    "trace_no": int(previous_fields[1]),
                    "time_utc": fields[1],
                    "latitude": convert_to_decimal(fields[2], fields[3]),
                    "longitude": convert_to_decimal(fields[4], fields[5]),
                    "fix_quality": int(fields[6]),
                    "num_satellites": int(fields[7]),
                    "hdop": float(fields[8]) if fields[8] else None,
                    "altitude": float(fields[9]),
                    "altitude_units": fields[10],
                    "geoid_separation": float(fields[11]),
                    "geoid_units": fields[12],
                    "dgps_age": fields[13].strip(),
                }
                gps_data.append(gps_entry)
            previous_line = line
    df = pd.DataFrame(gps_data)
    df['time_utc_decimal'] = df['time_utc'].apply(convert_time_to_decimal_seconds)
    return df


def convert_time_to_decimal_seconds(time_str):
    """
    Converts time in HHMMSS.SS format to decimal seconds.
    
    Args:
        time_str (str): Time in HHMMSS.SS format (e.g., '082121.60')
        
    Returns:
        float: Time in decimal seconds.
    """
    hours = int(time_str[:2])
    minutes = int(time_str[2:4])
    seconds = float(time_str[4:])
    return hours * 3600 + minutes * 60 + seconds


def convert_to_decimal(coord, direction):
    """
    Converts latitude/longitude in NMEA format to decimal degrees.

    Args:
        coord (str): Coordinate in NMEA format (e.g., 4925.3748032).
        direction (str): Direction (N/S/E/W).

    Returns:
        float: Coordinate in decimal degrees.
    """
    if not coord or not direction:
        return None

    degrees = int(coord[:2]) if direction in ['N', 'S'] else int(coord[:3])
    minutes = float(coord[2:]) if direction in ['N', 'S'] else float(coord[3:])
    decimal = degrees + (minutes / 60.0)

    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal


def interpolate_gps_data(file_path,tracenumber):
    """
    Parses GPS data, projects it to UTM, and interpolates the path.

    Args:
        file_path (str): Path to the text file containing GPS data.
        tracenumber (int): Number of traces to interpolate coordinates for 

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with interpolated points in UTM coordinates.
    """
    df = parse_gps_data(file_path)

    # Project to UTM
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.set_crs("EPSG:4326", inplace=True)
    gdf_utm = gdf.to_crs(epsg=32632)
    
    gdf_utm["distance"] = gdf_utm.geometry.distance(gdf_utm.geometry.shift())

    # Interpolate the distances along the path every TRACE
    
    #target_trace_no = np.arange(0, gdf_utm['trace_no'].iloc[-1], 1)
    target_trace_no = np.arange(0, tracenumber, 1)
    spline_x = CubicSpline(gdf_utm['trace_no'], gdf_utm.geometry.x,extrapolate=True)
    spline_y = CubicSpline(gdf_utm['trace_no'], gdf_utm.geometry.y,extrapolate=True)
    linear_t = interp1d(gdf_utm['trace_no'], gdf_utm.time_utc_decimal,fill_value="extrapolate")
    
    # Get interpolated coordinates
    interpolated_x = spline_x(target_trace_no)
    interpolated_y = spline_y(target_trace_no)
    interpolated_t = linear_t(target_trace_no)

    # Create a new GeoDataFrame for the interpolated points
    interpolated_points = [Point(x, y) for x, y in zip(interpolated_x, interpolated_y)]
    gdf_utm_interpolated = gpd.GeoDataFrame(geometry=interpolated_points, crs=gdf_utm.crs)
    gdf_utm_interpolated["distance"] = gdf_utm_interpolated.geometry.distance(gdf_utm_interpolated.geometry.shift())
    gdf_utm_interpolated["trace_no"] = target_trace_no
    gdf_utm_interpolated["time_utc_decimal"] = interpolated_t

    # Save the outputs
    gdf.to_file(file_path + "_utm_coords.gpkg", driver="GPKG")
    gdf_utm_interpolated.to_file(file_path + "_interpolated_utm_coords.gpkg", driver="GPKG")

    return gdf_utm_interpolated,gdf_utm



def interpolate_ifi_data(gdf, traces, steplength, offset=0):
    """
    Interpolates IFI ZEB location data along given trace numbers using cubic splines.

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        A GeoDataFrame containing IFI data with columns:
        - 'distance': Distance along the path.
        - 'geometry': Geometrical point locations.
        - 'gps_z': Elevation values.
        - 'roll', 'pitch', 'yaw': Orientation angles.
        - 'time_since_imu_start': Time since IMU start.

    traces : array-like
        The trace numbers at which to interpolate the IFI data.

    steplength : float
        The spatial step length used to compute trace numbers from distance.

    offset : float, optional (default=0)
        An optional offset for trace number calculation.

    Returns:
    --------
    gdf_ifi_interpolated : geopandas.GeoDataFrame
        A new GeoDataFrame with interpolated values, containing:
        - 'geometry': Interpolated spatial points.
        - 'ifi_z': Interpolated elevation values.
        - 'trace_no': Corresponding trace numbers.
        - 'roll', 'pitch', 'yaw': Interpolated orientation angles.
        - 'time_since_imu_start': Interpolated timestamps.
    """
    
    # Compute trace numbers based on distance
    trace_no_from_dist = ((gdf["distance"] - gdf["distance"].min()) / steplength).to_numpy()

    # Extract values for interpolation
    coords = np.array([(pt.x, pt.y) for pt in gdf.geometry])
    gps_z = gdf["gps_z"].to_numpy()
    roll, pitch, yaw = gdf["roll"].to_numpy(), gdf["pitch"].to_numpy(), gdf["yaw"].to_numpy()
    time_since_imu_start = gdf["time_since_imu_start"].to_numpy()

    # Create cubic splines for interpolation
    spline_x = CubicSpline(trace_no_from_dist, coords[:, 0], extrapolate=True)
    spline_y = CubicSpline(trace_no_from_dist, coords[:, 1], extrapolate=True)
    spline_z = CubicSpline(trace_no_from_dist, gps_z, extrapolate=True)
    spline_roll = CubicSpline(trace_no_from_dist, roll, extrapolate=True)
    spline_pitch = CubicSpline(trace_no_from_dist, pitch, extrapolate=True)
    spline_yaw = CubicSpline(trace_no_from_dist, yaw, extrapolate=True)
    spline_time = CubicSpline(trace_no_from_dist, time_since_imu_start, extrapolate=True)

    # Perform interpolation
    interpolated_x = spline_x(traces)
    interpolated_y = spline_y(traces)
    interpolated_z = spline_z(traces)
    interpolated_roll = spline_roll(traces)
    interpolated_pitch = spline_pitch(traces)
    interpolated_yaw = spline_yaw(traces)
    interpolated_time = spline_time(traces)

    # Construct interpolated GeoDataFrame
    gdf_ifi_interpolated = gpd.GeoDataFrame(
        {
            "geometry": [Point(x, y) for x, y in zip(interpolated_x, interpolated_y)],
            "ifi_z": interpolated_z,
            "trace_no": traces,
            "roll": interpolated_roll,
            "pitch": interpolated_pitch,
            "yaw": interpolated_yaw,
            "time_since_imu_start": interpolated_time,
        },
        crs=gdf.crs,
    )

    return gdf_ifi_interpolated


# Example usage as a script or module
if __name__ == "__main__":
    file_path = r"FILE____042.DZG"  # Replace with your file path
    result = interpolate_gps_data(file_path)
