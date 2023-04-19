import sys
from os import path
import pyDOE
import numpy as np
from torch import Tensor
from torch.autograd import grad

sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))

from PDE2D import *

class HeatEq(PDE2D):
    
    def __init__(self, t: Tuple[int, int], x: Tuple[int, int], N: int, net: PDENN, load_best=False, auto_lr=True) -> None:
        super().__init__(net, load_best, auto_lr)
        
        self.t = t
        self.x = x
        self.N = N
        
        self.X = Tensor(x[0] + (x[1] - x[0]) * np.array(pyDOE.lhs(2, N)))
        self.X.requires_grad_()
        
        self.U = self.net(self.X)
        self.dU = grad(
            self.U, self.X, torch.ones_like(self.U), True, True
            )[0]
        
        self.pt = self.dU[:, 0]
        self.px = self.dU[:, 1]
        
        self.dU2 = grad(
            self.dU, self.X, torch.ones_like(self.dU), True, True
            )[0]
        
        self.pt2 = self.dU2[:, 0]
        self.px2 = self.dU2[:, 1]
        
        self.IC = torch.cat((
            torch.hstack((
                torch.arange(t[0], t[1], 1.0/N).reshape((-1, 1)),
                torch.zeros((N, 1))
                )), 
            torch.hstack((
                Tensor(t[0] + (t[1] - t[0]) * np.array(pyDOE.lhs(1, N))),
                torch.zeros((N, 1))
                ))
            ))
        
        
        #TODO: To write
        
        self.BC = torch.cat((BC_lhs, BC_rhs))
    
    """
        The version of concacted input, with X = (t, x).
        
        X = [   (t_0, x_0), (t_0, x_1), ..., 
                (t_1, x_0), (t_1, x_1), ...,
                ...
                (t_N_t, x_0), ...]
        ##TODO: The input with high dimension x.
    """
    def loss(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        #? Calculate the Differential
        self.net.optim.zero_grad()
        
        self.U = self.net(self.X)
        self.dX = torch.autograd.grad(self.U, self.X, torch.ones_like(self.U), create_graph=True, retain_graph=True)[0]
        self.dX2 = torch.autograd.grad(self.dX, self.X, torch.ones_like(self.dX), create_graph=True, retain_graph=True)[0]
        
        self.pt = self.dX[:, 0]
        self.px = self.dX[:, 1]
        
        self.pt2 = self.dX2[:, 0]
        self.px2 = self.dX2[:, 1]
        
        eq_pde = self.pt - self.px2
        loss_pde = self.net.loss_criterion(eq_pde, torch.zeros_like(eq_pde))
        
        eq_bc = self.net(self.BC).reshape((-1, 1))
        loss_bc = self.net.loss_criterion(eq_bc, torch.zeros_like(eq_bc))
        
        eq_ic = self.net(self.IC).reshape((-1, 1)) - torch.sin(torch.pi * self.IC[:, 1].reshape((-1, 1)))
        loss_ic = self.net.loss_criterion(eq_ic, torch.zeros_like(eq_ic))
    
        #? Calculate the Loss
        loss = loss_pde + loss_bc + loss_ic
        loss.backward()
                
        self.net.loss_tensor = loss.clone()
        self.net.cnt_Epoch = self.net.cnt_Epoch + 1 
        self.net.loss_value = self.net.loss_tensor.item()
        self.net.loss_history.append(self.net.loss_value)
        
        #? Backward
        return self.net.loss_tensor
    
    
    def realSolution(self, X: torch.Tensor):
        t = X[:, 0]
        x = X[:, 1]
        return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)