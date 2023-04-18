from PDE2D import *

class HeatEq(PDE2D):
    
    def __init__(self,
                 t: Tuple[int, int], x: Tuple[int, int], 
                 N: int,
                 load_dict = False,
                 auto_lr = False,
                 net = PDENN(
                     input_size=2, output_size=1, hidden_depth=10, hidden_size=6, lr = 1e-2),
                 ) -> None:
        
        self.net = net
        super().__init__()
        self.net.PDENAME = "HeatEq"
        
        if(load_dict):
            self.loadBestDict()
        
        self.t = t
        self.x = x
        self.N = N
        self.auto_lr = auto_lr
    
   
    def calculateLoss(self) -> torch.Tensor:
        return 0.1 * self.loss_PDE() + 0.3 * self.loss_BC() + 0.6 * self.loss_IC()
    
    def loss_PDE(self) -> torch.Tensor:
        eq = self.pt - self.px2
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    
    def loss_BC(self) -> torch.Tensor:
        t_line = torch.rand((int(self.N/2), ))
        x_line = torch.rand((int(self.N/2), ))
        
        lhs = torch.stack(torch.meshgrid(t_line, 
                                         torch.asarray(self.x[0], dtype=torch.float32))).reshape((2, -1)).T
        rhs = torch.stack(torch.meshgrid(t_line,
                                         torch.asarray(self.x[1], dtype=torch.float32))).reshape((2, -1)).T
        bc = torch.vstack([lhs ,rhs])
        eq = self.net(bc)
        return self.net.loss_criterion(eq, torch.zeros_like(eq))

    
    def loss_IC(self) -> torch.Tensor:
        x_line = torch.rand((int(self.N/2), ))
        ic = torch.stack(torch.meshgrid(
            torch.asarray(self.t[0], dtype=torch.float32), 
            x_line)).reshape((2, -1)).T
        eq = self.net(ic).reshape((-1, 1)) - torch.sin(torch.pi * x_line.reshape((-1, 1)))
        return self.net.loss_criterion(eq, torch.zeros_like(eq))
    
    
    def realSolution(self, X: torch.Tensor):
        t = X[:, 0]
        x = X[:, 1]
        return torch.sin(torch.pi * x) * torch.exp(-torch.pi**2 * t)