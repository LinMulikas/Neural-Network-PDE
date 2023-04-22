from PDE_Square import *

class Heat(PDE_Square):
    net: Net
    
    def __init__(self, 
                 t: Tuple[int, int], 
                 x: Tuple[int, int], 
                 N: int) -> None:
        super().__init__(t, x, N)
            
            
    def loss(self):
        self.net.optim.zero_grad()
        
        U = self.net(self.net.X_region)
        dU = grad(U, self.net.X_region, tc.ones_like(U),
                  True, True)[0]
        dU2 = grad(dU, self.net.X_region, tc.ones_like(dU),
                   True, True)[0]
        
        pt = dU[:, 0]
        px = dU[:, 1]
        pt2 = dU[:, 0]
        px2 = dU[:, 1]
        
        #? Loss_PDE
        
        loss_pde = self.net.loss_criterion(pt,px2)
        
        #? Loss_BC
        y_pred = self.net(self.net.X_data)
        loss_data = self.net.loss_criterion(y_pred, self.net.y_data)
    
        #? Calculate the Loss
        loss = loss_pde + loss_data
        loss.backward()
        
        self.net.cnt_Epoch = self.net.cnt_Epoch + 1 
        self.net.loss_current = loss.clone()
        self.net.loss_history.append(self.net.loss_current.item())
        
        return loss