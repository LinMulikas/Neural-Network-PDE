from Net import *
from Heat import Heat
from PDE_Square import *
from torch.types import Number
import torch as tc

class NeuralGalerkin:
    pde: PDE_Square
    theta_step = 10000
    
    def __init__(self, pde) -> None:
        self.pde = pde
    
    def trainInit(self, eps: Number):
        self.pde.net.train(10000, loss_fn=self.loss_theta)
            
    def loss_theta(self):
        t0 = self.pde.t[0]
        x_lhs = self.pde.x[0]
        x_rhs = self.pde.x[1]
        x_line = tc.arange(x_lhs, x_rhs, (x_rhs - x_lhs)/self.theta_step).reshape((-1, 1))
        ic = tc.hstack((t0 * tc.ones_like(x_line), x_line)).reshape((-1, 2))
        ic.requires_grad_()
        
        loss = tc.trapezoid(tc.square(heat.u0(ic) - self.pde.net(ic)).reshape((-1,)))
        loss.backward()
    
        self.pde.net.loss_current = loss.clone()
        self.pde.net.cnt_Epoch = self.pde.net.cnt_Epoch + 1 
        self.pde.net.loss_history.append(self.pde.net.loss_current.item())
        return loss
    
    
    
heat = Heat((0, 1), (0, 1), 10000)
best = 'Heat/models/best.pt'
nil = ''
net = Net(pde_size = (2, 1), 
          shape = (16, 32), 
          data = heat.data_generator(),
          loadFile = best, lr = 1e-3)

heat.setNet(net)
NG = NeuralGalerkin(heat)
NG.trainInit(1e-6)