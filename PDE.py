from torch import Tensor
from Shape import *

class PDE:
    name: str
    domain: Shape
    
    def __init__(self, ) -> None:
        pass 
    
    
    def real(self, X: Tensor) -> Tensor:
        raise(RuntimeError("No implement of method."))
    
    

class HeatSin(PDE):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Heat"
        
    def u0(self, X: Tensor) -> Tensor:
        return th.sin(th.pi * X[:, 1]).reshape((-1, 1))
        
        
    def real(self, X: Tensor) -> Tensor:
        t = X[0, :].reshape((-1, 1))
        x = X[1, :].reshape((-1, 1))
        return th.