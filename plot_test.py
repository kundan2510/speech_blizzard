import matplotlib
matplotlib.use('Agg')


import numpy
import matplotlib.pyplot as plt
import os

plt.plot([1,2],[3,4], 'b.')

plt.savefig('abc.png')