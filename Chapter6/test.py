# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:11:54 2020

@author: SGQ
"""

import numpy as np

seasons = [3,6]
a = list(enumerate(seasons))
b = np.arange(6)
c = b + 1
p = np.zeros([10, 10, 10], dtype = np.float32)
p[:, 9, 9] = 1
p[1, 9, 9] = 0.5

for d in range(0, 101):
    print(d)