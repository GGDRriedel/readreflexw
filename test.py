# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:16:55 2024

@author: Riedel
"""


from readreflex import readreflex


testradargram = readreflex.radargram()

testradargram.read_data_file(r"exampledata/testdata.DAT",version=9)

print(testradargram.header)

testradargram.radarplot()
