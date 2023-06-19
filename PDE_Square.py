import numpy as np
import torch as th

from torch.autograd import grad
from torch import Tensor
from torch.nn import MSELoss

from pyDOE import lhs as LHS
from typing import Tuple, Literal
from torch.types import Number


class PDE_Square:
    NAME = "NONE"
    t: Tuple[float, float]
    x: Tuple[float, float]
    N_sample: int
    
    def __init__(self, 
                 t: Tuple[float, float], x: Tuple[float, float], 
                 u0,
                 N_sample: int) -> None:
        
        device = th.device('cuda') if(th.cuda.is_available()) else th.device('cpu')
        th.set_default_device(device)
        self.NAME = self.__class__.__name__
        self.t = t
        self.x = x
        self.u0 = u0
        self.N_sample = N_sample
    
    
    def u0(self, X: Tensor) -> Tensor:
        raise(RuntimeError("No implement of method."))
        
    
    
    def sampling(self, lhs: float, rhs: float, method: Literal['lhs', 'uni']) -> Tensor:
        data = None
        if(method == 'lhs'):
            data = lhs + (rhs - lhs) * th.arange(lhs, rhs, (1.0 * rhs - lhs)/self.N_sample).reshape((-1, 1))
        elif(method == 'uni'):
            data = lhs + (rhs - lhs) * Tensor(np.array(LHS(1, self.N_sample))).reshape((-1, 1))
        
        
        return data
      
      
        
    def loss(self, X: Tuple[Tensor, Tensor, Tensor], fn):
        raise(RuntimeError("No implement of method."))
    

    def sampling_data(self, method: Literal['lhs', 'uni'] = 'lhs') -> Tuple[Tensor, Tensor, Tensor]:
        x_lhs = self.x[0]
        x_rhs = self.x[1]
        t_lhs = self.t[0]
        t_rhs = self.t[1]
        
        x_ic = self.sampling(x_lhs, x_rhs, method)
        
        t_ic = t_lhs * th.ones_like(x_ic)
        X_ic = th.hstack((t_ic, x_ic))
        
        t_bc = self.sampling(t_lhs, t_rhs, method)
        
        x_bc_lhs = x_lhs * th.ones_like(t_bc)
        x_bc_rhs = x_rhs * th.ones_like(t_bc)
        X_bc_lhs = th.hstack((t_bc, x_bc_lhs))
        X_bc_rhs = th.hstack((t_bc, x_bc_rhs))
        X_bc = th.cat((X_bc_lhs, X_bc_rhs))
        
        
        x_line = self.sampling(x_lhs, x_rhs, method).reshape((-1,))
        t_line = self.sampling(t_lhs, t_rhs, method).reshape((-1,))
        T, X = th.meshgrid(t_line, x_line)
        X_region = th.stack((T, X)).reshape((-1, 2))
        
            
        return (X_region, X_ic, X_bc)
