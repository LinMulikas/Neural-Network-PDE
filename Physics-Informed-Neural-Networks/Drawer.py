from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Tuple
from torch import Tensor
from PDE_Square import *

import numpy as np
import torch as tc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

tc.set_default_device('cuda')

class Drawer:
    N: int
    PDE: PDE_Square
    
    def __init__(self, N, pde: PDE_Square) -> None:
        self.N = N
        self.PDE = pde

    def region(self):
        pde = self.PDE
        
        plt.scatter(Tensor.cpu(pde.net.X[:, 0]).detach().numpy(), 
                    Tensor.cpu(pde.net.X[:, 1]).detach().numpy(), 
                    linewidths=.1, marker='.')
        plt.scatter(Tensor.cpu(pde.net.IC[:, 0]).detach().numpy(), 
                    Tensor.cpu(pde.net.IC[:, 1]).detach().numpy(), 
                    linewidths=.1, marker='.')
        plt.scatter(Tensor.cpu(pde.net.BC[:, 0]).detach().numpy(), 
                    Tensor.cpu(pde.net.BC[:, 1]).detach().numpy(), 
                    linewidths=.1, marker='.')
        
    
    def plot(self, fn):
        pde = self.PDE
        self.draw3D(self.N, pde.t, pde.x, 
                  fn, pde.net.loss_current.item())
        
   
    def plotCounter(self):
        pde = self.PDE
        self.drawCounter(self.N, pde.t, pde.x, 
                  pde.net, pde.net.loss_current.item())
        
    
    
    def lossHistory(self):
        history = self.PDE.net.loss_history
        x_line = np.arange(len(history))
        plt.plot(x_line, history)
    
    
    
    def drawLast(self, x: int):
        history = self.PDE.net.loss_history[-x:]
        x_line = np.arange(len(history))
        plt.plot(x_line, history)
    
    
    def draw3D(self, N: int,  t: Tuple[int, int], x: Tuple[int, int], fn, loss: float):
        t_line = tc.arange(t[0], t[1], 1/self.N)
        x_line = tc.arange(x[0], t[1], 1/self.N)
        T, X = tc.meshgrid(t_line, x_line)
        
        data_input = tc.vstack([T.flatten(), X.flatten()]).T
        u_pred = fn(data_input.clone().detach()).reshape((-1, 1))
        
        U = np.reshape(Tensor.cpu(u_pred).detach().numpy(), (N, N))
        T = Tensor.cpu(T).detach().numpy()
        X = Tensor.cpu(X).detach().numpy()
        
        
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection ='3d')
        ax.plot_surface(T, X, U, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Loss={}'.format(round(loss, ndigits=10)))
        
        
    def drawCounter(self, N: int, t: Tuple[int, int], x: Tuple[int, int], fn, loss: float):
        t_line = tc.arange(t[0], t[1], 1/self.N)
        x_line = tc.arange(x[0], t[1], 1/self.N)
        T, X = tc.meshgrid(t_line, x_line)
        
        data_input = tc.vstack([T.flatten(), X.flatten()]).T
        u_pred = fn(data_input.clone().detach()).reshape((-1, 1))
        
        U = np.reshape(Tensor.cpu(u_pred).detach().numpy(), (N, N))
        T = Tensor.cpu(T).detach().numpy()
        X = Tensor.cpu(X).detach().numpy()
        
        
        plt.figure(figsize=(8, 8))
        ax = plt.contour(T, X, U, cmap='cool')
        artists, labels = ax.legend_elements()
        plt.legend(artists, labels, title= 'Value', fontsize= 8)