import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))
from NeuralGalerkin import *
from Heat import *
from Drawer import Drawer

heat = Heat((0, 1), (0, 1), 5000)

best = 'Neural-Network-PDE/NeuralGalerkin/models/best.pt'
nil = ''

net = Net(pde_size = (2, 1), 
          shape = (6, 10), 
          data = heat.data_generator(),
          loadFile = nil, lr = 1e-3)


heat.setNet(net)
NG = NeuralGalerkin(heat)
theta = NG.pde.net.paramVec()
NG.pde.net.updateParameters(theta)
theta_ = NG.pde.net.paramVec()
print(theta == theta_)