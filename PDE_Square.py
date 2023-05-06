import numpy as np
import torch as tc

from torch.autograd import grad
from torch import Tensor
from torch.nn import MSELoss
from Net import Net

from pyDOE import lhs as LHS
from typing import Tuple
from torch.types import Number

class PDE_Square:
    net: Net
    NAME = "NONE"
    t: Tuple[float, float]
    x: Tuple[float, float]
    N: int
    
    def __init__(self, t: Tuple[float, float], x: Tuple[float, float], N: int) -> None:
        device = tc.device('cuda') if(tc.cuda.is_available()) else tc.device('cpu')
        tc.set_default_device(device)
        self.NAME = self.__class__.__name__
        self.t = t
        self.x = x
        self.N = N
        
    def train(self, epoch):
        self.net.train(epoch, self.loss)
        
        
    def setNet(self, net: Net):
        self.net = net
        self.net.PDENAME = self.NAME
        
  
    def data_generator(self):
        t = self.t
        x = self.x 
        N = self.N
        
        #? X: LHS
        X = Tensor(LHS(2, N))
        X[:, 0] = t[0] + (t[1] - t[0]) * X[:, 0]
        X[:, 1] = x[0] + (x[1] - x[0]) * X[:, 1]
        
        #? X: Density
        # t_1 = tc.arange(0, 1/3, 1/(3*N/10)).reshape((-1, 1))
        # t_2 = tc.arange(1/3, 2/3, 1/(3*N/50)).reshape((-1, 1))
        # t_3 = tc.arange(2/3, 1, 1/(3*N/100)).reshape((-1, 1))
        
        # t_density = tc.cat([t_1, t_2, t_3]).reshape((-1,))
        
        # x_line = tc.arange(0, 1, 1/len(t_density)).reshape((-1,))
        
        # X = tc.stack(tc.meshgrid(t_density, x_line)).reshape((2, -1)).T
        
        X.requires_grad_()
        
        #? IC
        # x_line = tc.arange(x[0], x[1], 10/N).reshape((-1, 1))
        # x_ic = Tensor(np.sort(x[0] + (x[1] - x[0]) * np.array(LHS(1, int(N/5))), axis=0)).reshape((-1, 1))
        
        x_ic = tc.arange(x[0], x[1], 5/N).reshape((-1, 1))
        x_ic = x_ic[1:]
        x_ic = x_ic[:-1]
        # x_ic = tc.cat((x_line, x_ic))
                                                                        
                                                                        
        IC = tc.hstack((
            t[0] * tc.ones_like(x_ic),
            x_ic
            )).reshape((-1, 2))
        
        #? BC
        
        # t_bc_lhs = tc.cat((t_line, t_bc_lhs))
        
        #? t: uniform
        
        t_line = tc.arange(t[0], t[1], 5/N).reshape((-1, 1))
        
        #? t: LHS
        
        # t_line = Tensor(np.sort(t[0] + (t[1] - t[0]) * np.array(LHS(1, int(N/5))), axis=0)).reshape((-1, 1))
        # t_bc_rhs = tc.cat((t_line, t_bc_rhs))
        
        bc_lhs = tc.hstack((
            t_line[1:],
            x[0] * tc.ones_like(t_line[1:])
        )).reshape((-1, 2))
        
        
        bc_rhs = tc.hstack((
            t_line[1:],
            x[1] * tc.ones_like(t_line[1:])
        )).reshape((-1, 2))
        
        BC = tc.cat([bc_lhs, bc_rhs])
        
        return X, IC, BC
     
     

     
    def realSolution(self):
        raise(KeyError("No instance of method."))
      
      
    def loss(self):
        raise(KeyError("No instance of Method."))