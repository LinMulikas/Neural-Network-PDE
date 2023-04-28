from PDE_Square import *

class Heat(PDE_Square):
    net: Net
    
    def __init__(self, 
                 t: Tuple[int, int], 
                 x: Tuple[int, int], 
                 N: int) -> None:
        super().__init__(t, x, N)
       
  