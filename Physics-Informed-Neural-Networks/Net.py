import torch as tc
import os

from torch import nn

from typing import Tuple, List

class Net(tc.nn.Module):
    #? Net parameters.
    depth: int
    width: int
    
    #? Training parameters.
    cnt_Epoch = 0
    save_Gap = 100
    
    PDENAME: str
    
    loss_best = -1
    loss_current: tc.Tensor
    loss_history = []
    
    def __init__(self, 
                 size: Tuple[int, int], shape: Tuple[int, int], 
                 lr = 1e-2, act = nn.Tanh) -> None:
        super().__init__()
        
        #? Build the net.
        self.depth = shape[0]
        self.width = shape[1]
        self.input = nn.Linear(size[0], self.width)
        self.output = nn.Linear(self.width, size[1])
        self.linears = list()
        self.activates = list()
        
        for i in range(self.depth):
            self.linears.append(nn.Linear(self.width, self.width))
            self.activates.append(act())
        
        #? Build the optimizer.
        self.adam = tc.optim.Adam(self.parameters(), lr)
        self.lbfgs = tc.optim.LBFGS(
            self.parameters(),
            lr = lr,
            max_iter=int(5e3),
            max_eval=int(5e3),
            tolerance_grad=1e-5,
            tolerance_change=1e-9
            )
        self.opt = self.adam
             
    
    def forward(self, x):
        x = self.input(x)
        for i in range(self.depth):
            x = self.linears[i](x)
            x = self.activates[i](x)
        
        x = self.output(x)
        
        
    def info(self):
        self.loss_history.append(self.loss_current.item())
        
        if(self.loss_best == -1):
            self.loss_best = self.loss_current
        else:
            if(self.loss_current.item() < self.loss_best.item()):
                self.loss_best = self.loss_current
                
                
                
        
        if(self.cnt_Epoch % self.save_Gap == 0):
            
            print("Auto Save at Epoch {}.")
            print("Current loss {}".format(
                round(self.loss_current.item(), ndigits=10)))
            
            #TODO: Save Best.
            self.saveDict()
            
            
        
        
    
    def train(self, epoch, loss):
        for i in range(epoch):
            self.cnt_Epoch += 1
            self.opt.step(loss)
            self.info()
        
        
    def saveDict(self, fileName):
        filePath = os.path.join("./", self.PDENAME, "/", fileName, "/")
        tc.save(self, )