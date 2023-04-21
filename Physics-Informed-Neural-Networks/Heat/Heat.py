from PDE_Square import *

class Heat(PDE_Square):
            
    def loss(self):
        self.net.optim.zero_grad()
        self.U = self.net(self.X)
        self.dU = grad(self.U, self.X, tc.ones_like(self.U), True, True)[0]
        
        self.pt = self.dU[:, 0]
        self.px = self.dU[:, 1]
        
        self.dU2 = grad(self.dU, self.X, tc.ones_like(self.dU), True, True)[0]
        self.pt2 = self.dU2[:, 0]
        self.px2 = self.dU2[:, 1]
        
        criterion = MSELoss()
        eq_pde = (self.pt - self.px2).reshape((-1, 1))
        loss_pde = criterion(eq_pde, tc.zeros_like(eq_pde))
        
        eq_bc = self.net(self.BC)
        
        loss_bc = criterion(eq_bc, tc.zeros_like(eq_bc))
        
        eq_ic = self.net(self.IC).reshape((-1, 1)) - tc.sin(tc.pi * self.IC[:, 1]).reshape((-1, 1))
        loss_ic = criterion(eq_ic, tc.zeros_like(eq_ic))
        
        loss = loss_pde + loss_ic + loss_bc
        self.net.loss_current = loss
        loss.backward()
        return loss