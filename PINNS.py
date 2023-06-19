from PDE_Square import *
from ANN import ANN
from torch.autograd import grad

class PINNS:
    X: Tuple[Tensor, Tensor, Tensor]
    net: ANN 
    pde: PDE_Square
    
    re_sampling: bool = False
    
    def __init__(self, net: ANN, 
                 pde: PDE_Square,
                 re_sampling: bool = False) -> None:
        self.re_sampling = re_sampling
        self.net = net
        self.pde = pde
        self.X = self.pde.sampling_data()
    
    
    def train(self, epoch):
        self.net.train(epoch, self.loss)   
        
        
    def loss(self):
        X = self.X 
        net = self.net
        
        X_region = X[0]
        X_ic = X[1]
        X_bc = X[2]
        
        X_region.requires_grad_()
        X_ic.requires_grad_()
        X_bc.requires_grad_()
        
        u_pred: Tensor = net(X_region)
        u_ic: Tensor = net(X_ic)
        u_bc: Tensor = net(X_bc)
        
        u_pred.requires_grad_()
        
        dX1 = grad(u_pred, X_region, th.ones_like(u_pred), True, True)[0]
        dX2 = grad(dX1, X_region, th.ones_like(dX1), True, True)[0]

        pt = dX1[0].reshape((-1, 1))
        px2 = dX2[1].reshape((-1, 1))
        
        mse = MSELoss()
        
        loss_pde = mse(pt, px2)
        loss_ic = mse(u_ic, self.pde.u0(X_ic))
        loss_bc = mse(u_bc, th.zeros_like(u_bc))
        
        loss = loss_pde + loss_ic + loss_bc
        
        loss.backward()
        
        net.loss_current = loss.clone()
        net.cnt_Epoch = net.cnt_Epoch + 1 
        net.loss_history.append(net.loss_current.item())
        
        return loss_pde + loss_ic + loss_bc