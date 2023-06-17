import torch as th
import numpy as np

from torch import Tensor
from typing import Tuple, List
from pyDOE import lhs as LHS

class Shape:
    def boundary(self, N: int) -> Tensor:
        raise(RuntimeError("No implement of method."))
    
    def region(self, N: int) -> Tensor:
        raise(RuntimeError("No implement of method."))


class Rectangle(Shape):
    dim: int
    
    @staticmethod
    def LHSSampling(left: float, right: float, N: int) -> Tensor:
        return Tensor(left + (right - left) * np.array(LHS(1, N)))
    
    
    
class Interval(Rectangle):
    LEFT: float
    RIGHT: float
    
    
    def __init__(self) -> None:
        super().__init__()
        self.dim = 1

    
class Square(Rectangle):
    LEFT: float
    RIGHT: float
    UP: float
    DOWN: float
    
    def __init__(self, L: float, R: float, U: float, D: float) -> None:
        super().__init__()
        self.dim = 2
    
    @staticmethod
    def Square(x_axis: Tuple[float, float], y_axis: Tuple[float, float]):
        return Square(x_axis[0], x_axis[1], y_axis[0], y_axis[1])
    
        
           
        
class Circle(Shape):
    def __init__(self, x_center: float, y_center: float, raius: float) -> None:
        super().__init__()
        