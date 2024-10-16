# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:16:55 2024

@author: Riedel
"""


from readreflex import readreflex
import timeit

testradargram = readreflex.radargram()

testradargram.read_data_file(r"exampledata/testdata.DAT",version=9)

#print(testradargram.header)

#the standardplot 
testradargram.radarplot()

#note how it plots from the Beginning time set in the file

#%% Timing tests

testradargram.shortening(0,256)



#1D Numpy
start_time = timeit.default_timer()
for i in range(testradargram.header["tracenumber"]): 
    testradargram.apply_agc_pandas(tracenumber=i,inplace=False)
elapsed = timeit.default_timer() - start_time
print(elapsed) 

#1D Pandas
start_time = timeit.default_timer()
for i in range(testradargram.header["tracenumber"]): 
    testradargram.apply_agc(tracenumber=i,inplace=False)
elapsed = timeit.default_timer() - start_time
print(elapsed)

#2D Numpy
start_time = timeit.default_timer()
testradargram.apply_agc2D(inplace=False)
elapsed = timeit.default_timer() - start_time
print(elapsed)