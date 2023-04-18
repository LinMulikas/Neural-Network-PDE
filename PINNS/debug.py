from HeatEq import *
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

best_model = HeatEq([0, 1], [0, 1], 1000)

best_model.train(100)