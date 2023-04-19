from PDENN import *
from typing import *
from pyDOE import lhs
import numpy as np
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
    The class of order 2 PDE with 2-dimension. 
"""
class PDE2D():
    N: int
    net: PDENN
    
    #TODO: Abstract Method.
    #TODO: Abstract Method.
    def __init__(self, net: PDENN, load_best = False, auto_lr = True) -> None:
        self.net = net
        self.net.PDENAME = self.__class__.__name__

        super().__init__()
        
        if(load_best):
            self.loadBestDict()
        
        self.auto_lr = auto_lr
        
        
    def realSolution(self):
        raise KeyError("No instance of this method.")
    
    def loss(self):
        raise KeyError("No instance of Method.")
       
       
    def loss_PDE(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
       
    def loss_BC(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            

    def loss_IC(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            


    #TODO: Methods. 
    def drawError(self, N: int, cmap: str):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        t, x, u_pred = self.getPredPts(int(N))
        t, x, u_true = self.getRealPts(int(N))
        
        ax = plt.axes(projection = '3d')
        ax.plot_surface(t, x, u_pred - u_true, cmap=cmap, edgecolor='none')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Error surface with Loss=' + str(round(self.net.best_loss, ndigits=12))) 
    
    def drawPred(self, N: int, cmap: str):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))

        t, x, u = self.getPredPts(int(N))
        ax = plt.axes(projection = '3d')
        ax.plot_surface(t, x, u, cmap=cmap, edgecolor='none')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Predict Solution with Loss=' + str(round(self.net.best_loss, ndigits=12)))
        
        
    def drawReal(self, N: int, cmap: str):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        t, x, u = self.getRealPts(int(N))
        ax = plt.axes(projection = '3d')
        ax.plot_surface(t, x, u, cmap=cmap, edgecolor='none')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Real Solution') 
    

    def drawLossLast(self, x: int):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        history = self.net.loss_history[-x:]
        x = np.arange(len(history))
        plt.plot(x, history)


    def drawLossLast100(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        history = self.net.loss_history[-100:]
        x = np.arange(len(history))
        plt.plot(x, history)
    

    
    def drawLossHistory(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        history = self.net.loss_history
        x = np.arange(len(history))
        plt.plot(x, history)
    
    

    def getRealPts(self, N: int):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        torch.no_grad()
        
        t_line = torch.arange(self.t[0], self.t[1] + 1/N, 1/N)
        x_line = torch.arange(self.x[0], self.x[1] + 1/N, 1/N)
        
        t, x = torch.meshgrid(t_line, x_line)
        
        data_input = torch.vstack([t.flatten(), x.flatten()]).T
        
        u_real: torch.Tensor = self.realSolution(
                data_input.clone().detach()
            ).reshape((-1, 1))
        
        u = np.reshape(u_real.detach().numpy(), (N + 1, N + 1))
        t = t.detach().numpy()
        x = x.detach().numpy()
        return (t, x, u)
         

    def getPredPts(self, N: int):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        torch.no_grad()
        
        t_line = torch.arange(self.t[0], self.t[1] + 1/N, 1/N)
        x_line = torch.arange(self.x[0], self.x[1] + 1/N, 1/N)

        t, x = torch.meshgrid(t_line, x_line)
        
        data_input = torch.vstack([t.flatten(), x.flatten()]).T
        
        u_pred: torch.Tensor = self.net(
                data_input.clone().detach()
            ).reshape((-1, 1))
        
        u = np.reshape(u_pred.detach().numpy(), (N + 1, N + 1))
        t = t.detach().numpy()
        x = x.detach().numpy()
        return (t, x, u)
        
    
    
    def loss_Data(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        
    
    def train(self, epoch):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        for self.net.cnt_Epoch in range(epoch):   
            self.net.optim.step(self.loss)
            if(self.auto_lr):
                self.net.sched.step(self.net.loss_tensor)
            self.trainInfo()

        self.net.lbfgs.step(self.loss)
        torch.save(self, "final_model.pth")
    
    
    def loadDict(self, fileName):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        rootPath = os.getcwd()
        savePath = "/models/" + "/checkpoints_" + self.net.optim.__class__.__name__ + "/"
        
        filePath = rootPath + savePath + fileName
        
        data = torch.load(filePath)
        
        self.net.load_state_dict(data['dict'])
        self.net.best_Epoch = data['best_Epoch']
        self.net.best_loss = data['best_loss']
        self.net.cnt_Epoch = data['cnt_Epoch']
        self.net.loss_history = data['loss_history']
        
    def loadBestDict(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        rootPath = os.getcwd()
        savePath = "/models/" + "checkpoints_" + self.net.optim.__class__.__name__ + "/"
        fileName = 'best_dict.pt'
            
        filePath = rootPath + savePath + fileName
        
        data = torch.load(filePath)
        
        self.net.load_state_dict(data['dict'], True)
        self.net.best_Epoch = data['best_Epoch']
        self.net.best_loss = data['best_loss']
        self.net.cnt_Epoch = data['cnt_Epoch']
        self.net.loss_history = data['loss_history']
  
  
    def autoSave(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        rootPath = os.getcwd()
        savePath = "/models/" + "/checkpoints_" + self.net.optim.__class__.__name__ + "/"
        fileName = "Gen_{}_Loss_{}.pt".format(self.net.cnt_Epoch // self.net.save_gap, str(round(self.net.loss_value, 10))) 
        
        filepath = rootPath + savePath + fileName
        
        data = {'dict': self.net.state_dict(), 
                'best_loss': self.net.best_loss,
                'best_Epoch': self.net.best_Epoch,
                'cnt_Epoch': self.net.cnt_Epoch,
                'loss_history': self.net.loss_history}
        
        torch.save(data, filepath)
   
  
    def saveBest(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
        
        rootPath = os.getcwd()
        save_path = "/models/" + "/checkpoints_" + self.net.optim.__class__.__name__ + "/" + 'best_dict.pt'
            
        filepath = rootPath + save_path
        
        data = {'dict': self.net.state_dict(), 
                'best_loss': self.net.best_loss,
                'best_Epoch': self.net.best_Epoch,
                'cnt_Epoch': self.net.cnt_Epoch,
                'loss_history': self.net.loss_history}
        
        torch.save(data, filepath)
        
  
    def trainInfo(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        print(
            "Epoch {}, loss = {}, lr = {}.".format(
                self.net.cnt_Epoch, round(self.net.loss_value, 10), self.net.optim.state_dict()['param_groups'][0]['lr']))

        if(self.net.loss_value < self.net.best_loss):
            self.net.best_loss = self.net.loss_value
            self.net.best_Epoch = self.net.cnt_Epoch
            
            self.saveBest()
                          
            print("- Best dict saved at epoch {}.".format(
                self.net.cnt_Epoch))
            print("  With best loss: {:.10f}.".format(
                self.net.loss_value))
            print()
            
            
        else:
            print("  Best loss: {:.10f} at epoch {}.".format(
                self.net.best_loss, self.net.best_Epoch))
                

        if(self.net.cnt_Epoch % self.net.save_gap == 0):
            self.autoSave()
            
            print("  - Auto saved finished at epoch {}.".format(
                self.net.cnt_Epoch))
            print("    With loss: {:.10f}.".format(
                self.net.loss_value))
            print()
    
    
    
    def clearDif(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        self.dX = torch.zeros((1, 1))
        self.pt = torch.zeros((1, 1))
        self.px = torch.zeros((1, 1))
        self.dX2 = torch.zeros((1, 1))
        self.pt2 = torch.zeros((1, 1))
        self.px2 = torch.zeros((1, 1))
    