import sys
import gymnasium as gym
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Set plotting options
# %matplotlib inline

sample = [0.11, 0.43]
grid = [[0, 0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.2, 0.4, 0.6, 0.8, 1.0]]

a = zip(sample, grid)
print(tuple(a))

for s, g in zip(sample, grid):
    print(np.digitize(s, g))

b = [[1, 2, 3] for i in range(3)]
print(b)

t1 = (0, 1)
t2 = (2, )
print(t1)
t3 = t1 + t2
print(t3)

arr = np.array([[[1.1, 1.2, 1.3],[2.1,2.2,2.3],[3.1,3.2,3.3]], [[4.1,4.2,4.3],[5.1,5.2,5.3],[6.1,6.2,6.3]], [[7.1,7.2,7.3],[8.1,8.2,8.3],[9.1,9.2,9.3]]])
print(arr[0,1,2])
print(arr[t3])
print(arr[0][1][2])
print(arr[[0,1,2]])