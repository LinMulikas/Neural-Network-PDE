import torch as tc
from torch import Tensor
from typing import Tuple

a = tc.Tensor([1, 1, 2])
b = tc.Tensor([1, 2, 2])

def fn(x: Tensor) -> Tensor:
    return tc.ones((1))


print(type(fn))