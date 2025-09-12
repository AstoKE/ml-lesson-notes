# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 20:35:19 2025

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as pls

#kodlar
#veriyukleme

veriler = pd.read_csv('veriler.csv')

print(veriler)

# veri on isleme

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy', 'kilo']]
print(boykilo)

