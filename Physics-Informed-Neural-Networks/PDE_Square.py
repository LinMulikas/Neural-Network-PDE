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
    t: Tuple[int, int]
    x: Tuple[int, int]
    N: int
    
    def __init__(self, t: Tuple[int, int], x: Tuple[int, int], N: int) -> None:
        self.NAME = self.__class__.__name__
        self.t = t
        self.x = x
        self.N = N
        
    def setNet(self, net: Net):
        self.net = net
        self.net.PDENAME = self.NAME
        
  
    def data_generator(self):
        t = self.t
        x = self.x 
        N = self.N
        
        X = Tensor(LHS(2, N))
        X[:, 0] = t[0] + (t[1] - t[0]) * X[:, 0]
        X[:, 1] = x[0] + (x[1] - x[0]) * X[:, 1]
      
        X = X
        X.requires_grad_()
        
        #? IC
        x_line = tc.arange(x[0], x[1], 10/N).reshape((-1, 1))
        x_ic = Tensor(
            np.sort(
                x[0] + (x[1] - x[0]) * np.array(LHS(1, int(N/10))), 
                axis=0)).reshape((-1, 1))
        # x_ic = tc.cat((x_line, x_ic))
                                                                        
                                                                        
        IC = tc.hstack((
            t[0] * tc.ones_like(x_ic),
            x_ic
            )).reshape((-1, 2))
        
        #? BC
        t_line = tc.arange(t[0], t[1], 10/N).reshape((-1, 1))

        t_bc_lhs = Tensor(
                np.sort(t[0] + (t[1] - t[0]) * np.array(LHS(1, int(N/10))), axis=0)).reshape((-1, 1))
        
        # t_bc_lhs = tc.cat((t_line, t_bc_lhs))
        
        
        t_bc_rhs = Tensor(
                np.sort(t[0] + (t[1] - t[0]) * np.array(LHS(1, int(N/10))), axis=0)).reshape((-1, 1))
        # t_bc_rhs = tc.cat((t_line, t_bc_rhs))
        
        bc_lhs = tc.hstack((
            t_bc_lhs,
            x[0] * tc.ones_like(t_bc_lhs)
        )).reshape((-1, 2))
        
        
        bc_rhs = tc.hstack((
            t_bc_rhs,
            x[1] * tc.ones_like(t_bc_rhs)
        )).reshape((-1, 2))
        
        BC = tc.cat([bc_lhs, bc_rhs])
   
        return X, IC, BC
     
     
     
    def realSolution(self):
        raise(KeyError("No instance of method."))
      
      
    def loss(self):
        raise(KeyError("No instance of Method."))