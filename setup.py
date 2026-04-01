# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:50:22 2022

@author: Riedel
"""

import setuptools
from readreflex import _version

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="readreflexw",
    version=_version.version,
    author="Paul-Benjamin Riedel",
    author_email="riedel@geophysik-ggd.com",
    license='MIT',
    description="Python library to read and handle data written with the Software reflexW by Sandmeier Geophysical Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GGDRriedel/readreflexw",
    packages=setuptools.find_packages(),
    install_requires=[
        'segyio', 'tqdm', 'numpy', 'obspy', 'matplotlib',
        'pandas', 'h5py', 'scipy', 'geopandas', 'shapely',
        'scikit-learn', 'pyqtgraph', 'PyQt5',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
    ],
)
