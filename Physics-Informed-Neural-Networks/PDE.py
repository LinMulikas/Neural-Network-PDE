import numpy as np
import torch as tc

from torch import Tensor
from Net import Net

from pyDOE import lhs as LHS
from typing import Tuple
from torch.types import Number

from mpl_toolkits.mplot3d import 

class PDE:
    net: Net
    NAME = "NONE"
    
    def __init__(self, net, 
                 t: Tuple[int, int], x: Tuple[int, int], N: int) -> None:
        
        self.net = net
        self.NAME = self.__class__.__name__
        self.net.PDENAME = self.NAME
        self.t = t
        self.x = x
        self.N = N
        
        X_sample = np.array(LHS(2, 4*N))
        X_sample[:, 0] = self.t[0] + (self.t[1] - self.t[0]) * X_sample[:, 0]
        X_sample[:, 1] = self.x[0] + (self.x[1] - self.x[0]) * X_sample[:, 1]
        
        self.X = Tensor(X_sample)
        self.X.requires_grad_()
        
        #? IC
        t_ic = Tensor(self.t[0] + (self.t[1] - self.t[0]) * np.array(LHS(1, N))).reshape((-1, 1))
        self.IC = tc.hstack(
            [t_ic, tc.zeros_like(t_ic)]).reshape((-1, 2))
        
        #? BC
        x_bc_lhs = Tensor(
            self.x[0] + (self.x[1] - self.x[0]) * np.array(LHS(1, self.N)))
        
        bc_lhs = tc.hstack(
            [tc.zeros_like(x_bc_lhs)]
        ).reshape((-1, 2))
        
        x_bc_rhs = Tensor(
            self.x[0] + (self.x[1] - self.x[0]) * np.array(LHS(1, self.N))
        )
        bc_rhs = tc.hstack(
            [tc.ones_like(x_bc_rhs)]
        ).reshape((-1, 2))
        
        self.BC = tc.cat([bc_lhs, bc_rhs])
        
        
    def realSolution(self):
        raise(KeyError("No instance of method."))
        
        
    def loss(self):
        