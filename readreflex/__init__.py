# -*- coding: utf-8 -*-
"""
readreflexw
===========
Python library to read and process data written with ReflexW
(Sandmeier Geophysical Research).

Public API
----------
Core:
    radargram          — Main container class for GPR data (I/O, filtering,
                         export, visualization)
    GPRNoiseAnalyzer   — Advanced noise characterisation (systematic, coherent,
                         periodic, RFI, impulse detection)

Filters (stateless helpers, operate on radargram instances):
    butter_bandpass_filter
    trace_average_removal
    trace_full_average_removal
    gain_agc_instant
    gain_exp

Metadata / GPS:
    parse_gps_data        — Parse $GNGGA NMEA sentences from DZG files
    interpolate_gps_data  — UTM projection + cubic spline interpolation
    interpolate_ifi_data  — Interpolate IFI distance/orientation to traces
    parse_ifi_file        — Parse IFI ZEB GPS/IMU metadata files

Utilities:
    find_prom_sig         — Flexible prominent peak finder
    findmax               — Classic peak / zero-crossing detection
    apply_kalman_filter   — 6-state Kalman smoother for trajectory data
    normalize             — Min-max normalisation
"""

from .readreflex import radargram, GPRPlotterWidget
from .noise_analyzer import GPRNoiseAnalyzer
from .filtersuite import (
    butter_bandpass_filter,
    trace_average_removal,
    trace_full_average_removal,
    gain_agc_instant,
    gain_exp,
)
from .parse_dzg import (
    parse_gps_data,
    interpolate_gps_data,
    interpolate_ifi_data,
)
from .meta_files import parse_ifi_file
from .utils import (
    find_prom_sig,
    findmax,
    apply_kalman_filter,
    normalize,
)

__all__ = [
    "radargram",
    "GPRPlotterWidget",
    "GPRNoiseAnalyzer",
    "butter_bandpass_filter",
    "trace_average_removal",
    "trace_full_average_removal",
    "gain_agc_instant",
    "gain_exp",
    "parse_gps_data",
    "interpolate_gps_data",
    "interpolate_ifi_data",
    "parse_ifi_file",
    "find_prom_sig",
    "findmax",
    "apply_kalman_filter",
    "normalize",
]


