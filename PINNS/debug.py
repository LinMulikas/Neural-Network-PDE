from HeatEq import *
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

net = PDENN(input_size=2, output_size=1, hidden_depth=10, hidden_size=6, lr = 1e-2)
best_model = HeatEq(net, [0, 1], [0, 1], 1000, 1000, False)

best_model.train(100)