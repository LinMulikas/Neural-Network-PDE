from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple
from torch import Tensor
from PDE_Square import *

import numpy as np
import torch as tc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Drawer:
    N: int
    PDE: PDE_Square
    
    def __init__(self, N, pde: PDE_Square) -> None:
        self.N = N
        self.PDE = pde

    
    def pred(self):
        pde = self.PDE
        self.draw(self.N, pde.t, pde.x, 
                  pde.net, pde.net.loss_current.item())
        
    def real(self):
        pde = self.PDE
        self.draw(self.N, pde.t, pde.x, 
                  pde.realSolution, pde.net.loss_current.item())
    
    def lossHistory(self):
        x_line = np.arange(len(self.PDE.net.loss_history))
        plt.plot(x_line, self.PDE.net.loss_history) 
    
    
    
    def drawLast(self, x: int):
        history = self.PDE.net.loss_history[-x:]
        x_line = np.arange(len(history))
        plt.plot(x_line, history)
    
    
    def draw(self, N: int,  t: Tuple[int, int], x: Tuple[int, int], fn, loss: float):
        t_line = tc.arange(t[0], t[1], 1/self.N)
        x_line = tc.arange(x[0], t[1], 1/self.N)
        T, X = tc.meshgrid(t_line, x_line)
        
        data_input = tc.vstack([T.flatten(), X.flatten()]).T
        u_pred = fn(data_input.clone().detach()).reshape((-1, 1))
        
        U = np.reshape(u_pred.detach().numpy(), (N + 1, N + 1))
        T = T.detach().numpy()
        X = X.detach().numpy()
        
        ax = plt.axes(projection = '3d')
        ax.plot_surface(T, X, U, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Loss={}'.format(round(loss, ndigits=10)))