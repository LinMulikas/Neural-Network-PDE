import sys
from os import path
import pyDOE

sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))

from PDE2D import *

class HeatEq(PDE2D):

    def calculateLoss(self) -> torch.Tensor:
        return self.loss_PDE() + self.loss_BC() +  self.loss_IC()
    
    def loss_PDE(self) -> torch.Tensor:
        eq = self.pt - self.px2
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    
    def loss_BC(self) -> torch.Tensor:
        t_lhs = torch.Tensor(self.t[0] + (self.t[1] - self.t[0]) * pyDOE.lhs(1, self.N))
        t_lhs = torch.Tensor(self.t[0] + (self.t[1] - self.t[0]) * pyDOE.lhs(1, self.N))
        
        lhs = torch.stack([t_lhs, self.x[0] * torch.ones_like(t_lhs)]).reshape((2, -1)).T
        rhs = torch.stack([t_lhs, self.x[1] * torch.ones_like(t_lhs)]).reshape((2, -1)).T
        
        bc = torch.vstack([lhs ,rhs])
        eq = self.net(bc)
        return self.net.loss_criterion(eq, torch.zeros_like(eq))

    
    def loss_IC(self) -> torch.Tensor:
        x_line = torch.Tensor(self.x[0] + (self.x[1] - self.x[0])*pyDOE.lhs(1, self.N))
        ic = torch.stack([self.t[0] * torch.ones_like(x_line), x_line]).reshape((2, -1)).T
        
        eq = self.net(ic).reshape((-1, 1)) - torch.sin(torch.pi * x_line.reshape((-1, 1)))
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    def realSolution(self, X: torch.Tensor):
        t = X[:, 0]
        x = X[:, 1]
        return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)