from PDE2D import *

class HeatEq(PDE2D):
   
    def __init__(self, t: Tuple[int, int], x: Tuple[int, int], N: int, load_dict: bool) -> None:
        super().__init__()
        ##TODO: ensure that the loaded net has the same epoch.
        self.net.PDENAME = "HeatEq"
        
        if(load_dict):
            self.loadBestDict()
        
        self.t = t
        self.x = x
        self.N = N
    
   
    def calculateLoss(self) -> torch.Tensor:
        return 0.1 * self.loss_PDE() + 0.3 * self.loss_BC() + 0.6 * self.loss_IC()
    
    def loss_PDE(self) -> torch.Tensor:
        eq = self.pt - self.px2
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    
    def loss_BC(self) -> torch.Tensor:
        lhs = torch.stack(torch.meshgrid(self.t_line, self.x_line[0])).reshape((2, -1)).T
        rhs = torch.stack(torch.meshgrid(self.t_line, self.x_line[-1])).reshape((2, -1)).T
        bc = torch.vstack([lhs ,rhs])
        eq = self.net(bc)
        return self.net.loss_criterion(eq, torch.zeros_like(eq))

    
    def loss_IC(self) -> torch.Tensor:
        ic = torch.stack(torch.meshgrid(self.t_line[0], self.x_line)).reshape((2, -1)).T
        eq = self.net(ic).reshape((-1, 1)) - torch.sin(torch.pi * self.x_line.reshape((-1, 1)))
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    def realSolution(self, X: torch.Tensor):
        t = X[:, 0]
        x = X[:, 1]
        return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)