from PDENN import *
from typing import *
import numpy as np

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""
    The class of order 2 PDE with 2-dimension. 
"""
class PDE2D():
    N: int
    
    def __init__(self) -> None:
        self.net = PDENN(input_size=2, output_size=1, hidden_depth=10, hidden_size=6, lr = 1e-2)
        self.net.PDENAME = None
        
        
    def setOptim(self, optim):
        self.net.optim = optim
        
        
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
        
        t_line = torch.arange(self.t_line[0], self.t_line[-1] + 1/N, 1/N)
        x_line = torch.arange(self.x_line[0], self.x_line[-1] + 1/N, 1/N)

        t, x = torch.meshgrid(t_line, x_line)
        
        data_input = torch.vstack([t.flatten(), x.flatten()]).T
        
        u_real: torch.Tensor = self.realSolution(
                data_input.clone().detach()
            ).reshape((-1, 1))
        
        u = np.reshape(u_real.detach().numpy(), (N + 1, N + 1))
        t = t.detach().numpy()
        x = x.detach().numpy()
        return (t, x, u)
         
    
    def realSolution(self):
        raise KeyError("No instance of this method.")
        
    def getPredPts(self, N: int):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        torch.no_grad()
        
        t_line = torch.arange(self.t_line[0], self.t_line[-1] + 1/N, 1/N)
        x_line = torch.arange(self.x_line[0], self.x_line[-1] + 1/N, 1/N)

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
            self.net.sched.step(self.net.loss_tensor)
            self.trainInfo()

        self.net.lbfgs.step(self.loss)
    
        torch.save(self, "final_model.pth")
    
    def loadDict(self, fileName):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        data = torch.load('./models/' + self.net.PDENAME + '/' + 'checkpoints_' + self.net.optim.__class__.__name__ + '/' + fileName + '.pt')
        self.net.load_state_dict(data['dict'])
        self.net.best_Epoch = data['best_Epoch']
        self.net.best_loss = data['best_loss']
        self.net.cnt_Epoch = data['cnt_Epoch']
        self.net.loss_history = data['loss_history']
        
    def loadBestDict(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        data = torch.load('./models/' + self.net.PDENAME + '/training/' + '/best_dict.pt')
        self.net.load_state_dict(data['dict'], True)
        self.net.best_Epoch = data['best_Epoch']
        self.net.best_loss = data['best_loss']
        self.net.cnt_Epoch = data['cnt_Epoch']
        self.net.loss_history = data['loss_history']
  
  
    def autoSave(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        save_path = "..PINNS/models/" + self.__class__.__name__ + "/checkpoints_" + self.net.optim.__class__.__name__ + "/"
            
        filepath = os.path.join(
            save_path, 'auto_save_Gen_{}_Loss_.'.format(self.net.cnt_Epoch // self.net.save_gap) + str(round(self.net.loss_value, 10)) + '.pt'.format(self.net.cnt_Epoch//self.net.save_gap))
        
        data = {'dict': self.net.state_dict(), 
                'best_loss': self.net.best_loss,
                'best_Epoch': self.net.best_Epoch,
                'cnt_Epoch': self.net.cnt_Epoch,
                'loss_history': self.net.loss_history}
        
        torch.save(data, filepath)
   
  
    def saveBest(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        save_path = '/home/wangdl/Public_Project/PDE_NN_Solver/model_dict/' + self.net.PDENAME + '/'  'checkpoints_' + self.net.optim.__class__.__name__
            
        filepath = os.path.join(save_path, 'best_dict.pt')
        
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
            self.drawPred(500, 'coolwarm')
                          
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
    
    
    def calculateLoss(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
    
    """
        The version of concacted input, with X = (t, x).
        
        X = [   (t_0, x_0), (t_0, x_1), ..., 
                (t_1, x_0), (t_1, x_1), ...,
                ...
                (t_N_t, x_0), ...]
        ##TODO: The input with high dimension x.
    """
    def loss(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        #? Calculate the Differential
        self.net.optim.zero_grad()
        
        t_line = torch.rand((self.N, )) * (self.t[1] - self.t[0]) + self.t[0]
        x_line = torch.rand((self.N, )) * (self.x[1] - self.x[0]) + self.x[0]
        X = torch.meshgrid(torch.vstack(t_line, x_line)).reshape((2, -1)).T
        
        self.U = self.net(self.X)
        self.dX = torch.autograd.grad(self.U, self.X, torch.ones_like(self.U), create_graph=True, retain_graph=True)[0]
        self.dX2 = torch.autograd.grad(self.dX, self.X, torch.ones_like(self.dX), create_graph=True, retain_graph=True)[0]
        
        self.pt = self.dX[:, 0]
        self.px = self.dX[:, 1]
        
        self.pt2 = self.dX2[:, 0]
        self.px2 = self.dX2[:, 1]
        #? Calculate the Loss
        loss = self.calculateLoss()
        loss.backward()
                
        self.net.loss_tensor = loss.clone()
        self.net.cnt_Epoch = self.net.cnt_Epoch + 1 
        self.net.loss_value = self.net.loss_tensor.item()
        self.net.loss_history.append(self.net.loss_value)
        
        #? Backward
        return self.net.loss_tensor
    

    
    
    def clearDif(self):
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
        self.dX = torch.zeros((1, 1))
        self.pt = torch.zeros((1, 1))
        self.px = torch.zeros((1, 1))
        self.dX2 = torch.zeros((1, 1))
        self.pt2 = torch.zeros((1, 1))
        self.px2 = torch.zeros((1, 1))
    
    
    
    
    
    def loss_PDE(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            
       
    def loss_BC(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            

    
    def loss_IC(self) -> torch.Tensor:
        if(self.net.PDENAME == None):
            raise(KeyError("No instance of method."))
            