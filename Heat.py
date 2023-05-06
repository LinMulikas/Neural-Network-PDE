from PDE_Square import *

class Heat(PDE_Square):
    net: Net
    
    def __init__(self, 
                 t: Tuple[int, int], 
                 x: Tuple[int, int], 
                 N: int) -> None:
        super().__init__(t, x, N)
       
    
    def u0(self, X: tc.Tensor):
        return tc.sin(tc.pi * X[:, 1]).reshape((-1, 1))
       
       
    def loss(self):
        
        #? Use the fixed X.
        X = self.net.X
        X.requires_grad_()
        
        U = self.net(X)
        dU = grad(U, X, tc.ones_like(U), True, True)[0]
        pt = dU[:, 0].reshape((-1, 1))
        
        dU2 = grad(dU[:, 1], X, tc.ones_like(dU[:, 1]), True, True)[0]
        pxx = dU2[:, 1].reshape((-1, 1))
        
        
        #? Loss_PDE
        
        loss_pde = self.net.loss_criterion(pt, pxx)
        
        #? Loss_IC
        eq_ic = self.net(self.net.IC)
        y_ic = self.u0(X)
        loss_ic = self.net.loss_criterion(eq_ic, y_ic)
        
        #? Loss_BC
        eq_bc = self.net(self.net.BC)
        y_bc = tc.zeros_like(eq_bc)
        loss_bc = self.net.loss_criterion(eq_bc, y_bc)
        
        #? Calculate the Loss
        loss = loss_pde + loss_ic + loss_bc
        loss.backward()
        
        self.net.loss_current = loss.clone()
        self.net.cnt_Epoch = self.net.cnt_Epoch + 1 
        self.net.loss_history.append(self.net.loss_current.item())
        
        return loss