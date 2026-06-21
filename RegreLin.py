""" Copyright 2026 Esteban Alfaro Sabogal 

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# RegreLin

import numpy as np

# RegreLin: given two variables (x) and (y), exist a straight line defined as y = mx + b, that reduce at minimum the Euclidean distance to all cartesian coordinates (x,y). 
# The function return the slope (m) and the y-axis intercept (b) of the straight line. The determination coefficient (r2) is also returned by the function.

def RegreLin(x, y): 
# input: cartesian x- and y- coordinates in the plane
# output: slope (pend), intercept (inter), determination coefficient (r2)

    meanx = np.mean(x)
    meany = np.mean(y)
    sumy = np.sum(y)
    
    sum_num_pend = 0
    sum_den_pend = 0
    sum_y_sq = 0
    sum_xy = 0

    nDatos = x.shape[0]

    for i in range(nDatos):
        sum_num_pend = ((x[i] - meanx) * (y[i] - meany)) + sum_num_pend
        sum_den_pend = pow((x[i] - meanx), 2) + sum_den_pend
        sum_y_sq = pow(y[i], 2) + sum_y_sq
        sum_xy = (x[i] * y[i]) + sum_xy

    pend = sum_num_pend / sum_den_pend # slope (m)
    inter = meany - (pend * meanx) # y-axis intercept (b)
    sse = sum_y_sq - (inter * sumy) - (pend * sum_xy) # sum of total residuals
    sst = sum_y_sq - (pow(sumy, 2) / nDatos) # total variation
    r2 = 1 - (sse / sst) # determination coefficient (r2)
    
    return pend, inter, r2
