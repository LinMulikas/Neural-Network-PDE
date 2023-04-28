import torch as tc
from torch.autograd import grad

def fn(t: tc.Tensor, x: tc.Tensor):
    X = tc.hstack([t, x])
    return f(X)


def f(X: tc.Tensor):
    t = X[:, 0].reshape((-1, 1))
    x = X[:, 1].reshape((-1, 1))
    return tc.pow(t, 2) + tc.pow(x, 3)

t = tc.Tensor([1, 2, 3]).reshape((-1, 1))
t.requires_grad_()
x = tc.Tensor([1, 2, 3]).reshape((-1, 1))
x.requires_grad_()

X = tc.hstack([t, x])

U = f(X)
dU = grad(U, X, tc.ones_like(U), True, True)[0]
dU2 = grad(dU, X, tc.ones_like(dU), True, True)[0]

print(U)