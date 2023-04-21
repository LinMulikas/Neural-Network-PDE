import torch as tc
import os

from torch import nn

from typing import Tuple, List
from torch.types import Number

class Net(tc.nn.Module):
    #? Net parameters.
    depth: int
    width: int
    lr: float = 1e-2
    
    #? Training parameters.
    cnt_Epoch = 0
    save_Gap = 500
    
    PDENAME: str
    
    loss_best:tc.Tensor = tc.ones((1))
    loss_current: tc.Tensor
    loss_history = []
    
    def __init__(self, 
                 pde_size: Tuple[int, int], shape: Tuple[int, int], 
                 loadFile:str = '',
                 lr = 1e-3, act = nn.Tanh) -> None:
        super().__init__()
        
        self.lr = lr
        #? Build the net.
        self.depth = shape[0]
        self.width = shape[1]
        self.input = nn.Linear(pde_size[0], self.width)
        self.output = nn.Linear(self.width, pde_size[1])
        self.linears = list()
        self.activates = list()
        
        for i in range(self.depth):
            self.linears.append(nn.Linear(self.width, self.width))
            self.activates.append(act())
        
        #? Build the optimizer.
        self.adam = tc.optim.Adam(self.parameters(), self.lr)
        self.lbfgs = tc.optim.LBFGS(
            self.parameters(),
            lr = self.lr,
            max_iter=int(5e3),
            max_eval=int(5e3),
            tolerance_grad=1e-5,
            tolerance_change=1e-9
            )
        self.optim = self.adam
        
        #TODO: Load
        if(loadFile != ''):
            self.loadDict(loadFile)
            
    
    def forward(self, x):
        x = self.input(x)
        for i in range(self.depth):
            x = self.linears[i](x)
            x = self.activates[i](x)
        
        x = self.output(x)
        return x
    
    
    def train(self, epoch, loss):
        for i in range(epoch):
            self.cnt_Epoch += 1
            self.optim.step(loss)
            self.info()    
        
        
    def info(self):
        self.loss_history.append(self.loss_current.item())
        
        print("Epoch {}, lr = {}".format(
            self.cnt_Epoch,
            self.lr
        ))
        print("-- Current loss = {}".format(
            round(self.loss_current.item(), ndigits=10)
        ))
        print()
        
        #? Save Best
        if(tc.equal(self.loss_best, tc.ones((1)))):
            self.loss_best = self.loss_current
            print("Best save at Epoch {}, lr = {}".format(
                self.cnt_Epoch,
                self.lr))
            print("-- Best loss = {}".format(
                round(self.loss_best.item(), ndigits=10)
            ))
            print()
            self.saveBest()
            
        else:
            if(float(self.loss_current.item()) < float(self.loss_best.item())):
                self.loss_best = self.loss_current
                self.saveBest()
                
        #? Auto Save
        if(self.cnt_Epoch % self.save_Gap == 0):
            print("Auto Save at Epoch {}, lr = {}".format(
                self.cnt_Epoch,
                self.lr))
            print("-- Save loss = {}".format(
                round(self.loss_current.item(), ndigits=10)))
            print()
            
            #TODO: Save Best.
            rootPath = os.getcwd()
            self.saveDict(
                "Gen_{}_Loss_{}.pt".format(
                    self.cnt_Epoch / self.save_Gap + 1,
                    round(float(self.loss_current.item()), 8)))
       
    
    def loadDict(self, fileName):
        rootPath = os.getcwd()
        filePath = os.path.join(rootPath, fileName)
        
        data = tc.load(filePath)
        
        self.load_state_dict(data['dict'], True)
        self.best_Epoch = data['best_Epoch']
        self.best_loss = data['best_loss']
        self.cnt_Epoch = data['cnt_Epoch']
        self.loss_history = data['loss_history']
    
        
    def saveBest(self):
        self.saveDict("best.pt")
        
        
    def saveDict(self, fileName):
        rootPath = os.getcwd()
        filePath = 'models'
        
        data = {'dict': self.state_dict(), 
                'best_loss': self.best_loss,
                'best_Epoch': self.best_Epoch,
                'cnt_Epoch': self.cnt_Epoch,
                'loss_history': self.loss_history}
        
        
        tc.save(data, os.path.join(rootPath, filePath, fileName))