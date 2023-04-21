import sys
from os import path
import pyDOE

sys.path.append(path.dirname(path.dirname(path.abspath('__file__'))))

from PDE2D import *

class Poisson(PDE2D):
    
    
    def realSolution(self, X: torch.Tensor):
        x = X[:, 0]
        y = X[:, 1]
        return (torch.sin(torch.pi * x) * torch.sin(torch.pi * y))/(2*torch.pi**2)
    
    
    def calculateLoss(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        return self.loss_PDE() + self.loss_BC()
        
        
    def loss_PDE(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        x = self.X[:, 0]
        y = self.X[:, 1]
        eq = torch.square(self.pt) + torch.square(self.px) + torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
       
    def loss_BC(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        t_lhs = torch.Tensor(
            self.t[0] + (self.t[1] - self.t[0]) * pyDOE.lhs(self.N))
        
        t_rhs = torch.Tensor(
            self.t[0] + (self.t[1] - self.t[0]) * pyDOE.lhs(self.N))
        
        x_lhs = torch.Tensor(
            self.x[0] + (self.x[1] - self.x[0]) * pyDOE.lhs(self.N))
        
        x_rhs = torch.Tensor(
            self.x[0] + (self.x[1] - self.x[0]) * pyDOE.lhs(self.N))
        
        self.bc_left = torch.stack((t_lhs, torch.zeros_like(t_lhs)))
        self.bc_right = torch.stack((t_rhs, torch.ones_like(t_rhs)))
        self.bc_down = torch.stack((torch.zeros_like(x_lhs), x_lhs))
        self.bc_up = torch.stack((torch.ones_like(x_rhs), x_rhs))
        
        self.bc = torch.stack((self.bc_left, self.bc_right, self.bc_down, self.bc_up)).reshape((2, -1)).T
        
        eq = self.net(self.bc)
        return self.net.loss_criterion(eq, torch.ones_like(eq))
        