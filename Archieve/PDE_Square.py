import numpy as np
import torch as th

from torch.autograd import grad
from torch import Tensor
from torch.nn import MSELoss
from Net import Net

from pyDOE import lhs as LHS
from typing import Tuple, Literal
from torch.types import Number

class PDE_Square:
    net: Net
    NAME = "NONE"
    t: Tuple[float, float]
    x: Tuple[float, float]
    N_sample: int
    
    def __init__(self, 
                 t: Tuple[float, float], x: Tuple[float, float], N_sample: int) -> None:
        
        device = th.device('cuda') if(th.cuda.is_available()) else th.device('cpu')
        th.set_default_device(device)
        self.NAME = self.__class__.__name__
        self.t = t
        self.x = x
        self.N_sample = N_sample
        
        
    def train(self, epoch):
        self.net.train(epoch, self.loss)
    
    def loss(self):
        raise(RuntimeError("No implement of method."))
    
    def sampling(self, method: Literal['lhs', 'uni'] = 'lhs') -> Tensor:
        x_lhs = self.x[0]
        x_rhs = self.x[1]
        t_lhs = self.t[0]
        t_rhs = self.t[1]
        
        if(method == 'uni'):
            x_ic = x_lhs + (x_rhs - x_lhs) * th.arange(
                    x_lhs, x_rhs, (1.0 * x_rhs - x_lhs)/self.N_sample
                ).reshape((-1, 1))
            
            t_ic = t_lhs * th.ones_like(x_ic)
            X_ic = th.hstack((t_ic, x_ic))
            
            t_bc = t_lhs + (t_rhs - t_lhs) * th.arange(
                    t_lhs, t_rhs, (1.0 * t_rhs - t_lhs)/self.N_sample
                ).reshape((-1, 1))
            x_bc_lhs = x_lhs * th.ones_like(t_bc)
            x_bc_rhs = x_rhs * th.ones_like(t_bc)
            X_bc_lhs = th.hstack((t_bc, x_bc_lhs))
            X_bc_rhs = th.hstack((t_bc, x_bc_rhs))
            X_bc = th.cat((x_bc_lhs, x_bc_rhs))
            
            