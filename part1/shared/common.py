#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
common utilities functions

@author: teddy
"""

import numpy as np
import operator as op
import pandas as pd

# return a sorted dictionary
def dicSorted(items, desc = True):
    return sorted(items, key=op.itemgetter(1), reverse=desc)

# load a numpy array from txt file
def loadTable(file):
    return np.array(pd.read_table(file,header=None))

