from Heat.Heat import Heat
from Net import *
from Heat import *
from PDE_Square import *
from torch.types import Number
import torch as tc

class NeuralGalerkin:
    net: Net
    pde: PDE_Square
    theta_step = 10000
    
    def __init__(self, pde) -> None:
        self.pde = pde
    
    def trainInit(self, eps: Number):
        self.net.train(10000, loss_fn=self.loss_theta)
            
    def loss_theta(self):
        t0 = self.pde.t[0]
        x_lhs = self.pde.x[0]
        x_rhs = self.pde.x[1]
        x_line = tc.arange(x_lhs, x_rhs, (x_rhs - x_lhs)/self.theta_step).reshape((-1, 1))
        ic = tc.hstack((t0 * tc.ones_like(x_line), x_line)).reshape((-1, 2))
        ic.requires_grad_()
        
        loss = tc.trapezoid(tc.square(heat.u0(ic) - self.net(ic)))
        loss.backward()
    
        self.net.loss_current = loss.clone()
        self.net.cnt_Epoch = self.net.cnt_Epoch + 1 
        self.net.loss_history.append(self.net.loss_current.item())
        
        return loss
    
        
        
heat = Heat((0, 1), (0, 1), 10000)
NG = NeuralGalerkin(heat)
NG.loss_theta()