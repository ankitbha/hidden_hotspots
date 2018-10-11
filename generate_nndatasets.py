# ********************************************************************
# 
# Generate datasets for neural network training, validation and
# testing, using panel csv or dta data. Using time series data over a
# period of several days for multiple locations, we will generate a
# 2-D panel dataset, where first axis is spatial and second axis is
# temporal.
#
# Author: Shiva R. Iyer
#
# Date: Aug 15, 2018
#
# ********************************************************************

import os
import numpy as np
import pandas as pd
import torch

infilepath = '../dtaFiles/Aug_2018_Kaiterra/pilot2_2018_panel_5min__9_Aug_2018.dta'
savedir = 'datasets/'

def generate(sensor_id, neighborlist=None):
    
