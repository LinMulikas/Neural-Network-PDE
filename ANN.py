import torch as th
from torch import Tensor
import numpy as np
import os

from torch import nn
from torch.autograd import grad
from torch.autograd import grad
from typing import Tuple, List, Dict
from torch.types import Number
from torch.nn import Module


from pyDOE import lhs as LHS

class ANN(th.nn.Module):
    device = th.device('cuda') if(th.cuda.is_available()) else th.device('cpu')
        
    
    #? Net parameters.
    input_size: int
    output_size: int
    depth: int
    width: int
    lr: float = 1e-2
    
    #? Training parameters.
    cnt_Epoch = 0
    best_Epoch = -1
    save_Gap = 50
    
    loss_best: th.Tensor = th.ones((1)).to(device)
    loss_current: th.Tensor = th.ones((1)).to(device)
    loss_history = []
    loss_criterion = th.nn.MSELoss(reduce=True, reduction='mean')  
    
    def __init__(self, 
                 data_size: Tuple[int, int],
                 shape: Tuple[int, int], 
                 loadFile:str = '',
                 lr: float = 1e-2,
                 act = nn.Tanh,
                 auto_lr = True,) -> None:
        
        super().__init__()
        
        self.act = act
        self.lr = lr
        self.auto_lr = auto_lr
        #? Build the net.
        self.depth = shape[0]
        self.width = shape[1]
        
        self.input_size = data_size[0]
        self.output_size = data_size[1]
        input = nn.Linear(data_size[0], self.width)
        hidden = []
        output = nn.Linear(self.width, data_size[1])
        
        for i in range(self.depth):
            hidden.append(nn.Linear(self.width, self.width))
            hidden.append(act())
        
        self.model = nn.Sequential(
            input,
            self.act(),
            *hidden,
            output
        )
        
        #? Build the optimizer.
        self.adam = th.optim.Adam(self.parameters(), self.lr)
        self.sgd = th.optim.SGD(self.parameters(), self.lr)
        
        self.sched = th.optim.lr_scheduler.ReduceLROnPlateau(
            self.adam, mode = 'min', factor=0.1, patience=100)
        
        #TODO: Load best has no current loss
        if(loadFile != ''):
            self.loadDict(loadFile)        

        

   
    
    def forward(self, X):
        raise(RuntimeError("No implement of method."))
    
    
    #? Train, information visualization, save and load.
    
    def train(self, epoch, loss_fn):
        for self.cnt_Epoch in range(epoch):
            self.adam.zero_grad()
            self.adam.step(loss_fn)
            self.sched.step(self.loss_current)
            self.info()
        

    def info(self):
        #? Save Best
        
        if(th.equal(self.loss_best, th.ones((1)))):
            self.loss_best = self.loss_current.clone()
            self.best_Epoch = self.cnt_Epoch
            #self.optim.state_dict()['param_groups'][0]['lr']
            print("Best save at Epoch {}, lr = {}, Epoch = {}".format(
                self.cnt_Epoch,
                self.lr,
                self.cnt_Epoch))
            print("-- Best loss = {}. At Epoch {}".format(
                round(self.loss_best.item(), ndigits=10),
                self.cnt_Epoch
            ))
            print()
            self.saveBest()
            
        else:
            if(self.loss_current.item() < self.loss_best.item()):
                self.loss_best = self.loss_current.clone()
                self.best_Epoch = self.cnt_Epoch
                print("Best save at Epoch {}, lr = {}, Epoch = {}".format(
                    self.cnt_Epoch,
                    self.lr,
                    self.cnt_Epoch))
                print("-- Best loss = {}. At Epoch {}".format(
                    round(self.loss_best.item(), ndigits=10),
                    self.cnt_Epoch
                ))
                print()
                self.saveBest()
                    
        #? Auto Save
        if(self.cnt_Epoch % self.save_Gap == 0):
            print("Auto Save at Epoch {}, lr = {}, Epoch = {}".format(
                self.cnt_Epoch,
                self.lr,
                self.cnt_Epoch))
            print("-- Save loss = {}. At Epoch = {}".format(
                round(self.loss_current.item(), ndigits=10),
                self.cnt_Epoch))
            print()
            
            #TODO: Save Best.
            rootPath = os.gethwd()
            self.saveDict(
                "autosave/Gen_{}_Loss_{}.pt".format(
                    int(self.cnt_Epoch / self.save_Gap + 1),
                    round(float(self.loss_current.item()), 8)))
        
        
        
        self.loss_history.append(self.loss_current.item())
        
        print("Epoch {}, lr = {}".format(
            self.cnt_Epoch,
            self.lr
        ))
        print("-- Current loss = {}".format(
            round(self.loss_current.item(), ndigits=10)
        ))
        print("-- Best loss = {}. At Epoch {}".format(
                round(self.loss_best.item(), ndigits=10),
                self.best_Epoch
            ))
        print()
    
    
    def loadDict(self, fileName):
        rootPath = os.gethwd()
        filePath = os.path.join(rootPath, fileName)
        
        data = th.load(filePath, map_location=self.device)
        
        self.load_state_dict(data['dict'], True)
        self.loss_current = data['loss_current']
        self.best_Epoch = data['best_Epoch']
        self.loss_best = data['loss_best']
        self.cnt_Epoch = data['cnt_Epoch']
        self.loss_history = data['loss_history']
        
        
    def saveBest(self):
        self.saveDict("best.pt")
        
        
    def saveDict(self, fileName):
        rootPath = os.getcwd()
        filePath = 'models'
        
        data = {'dict': self.state_dict(), 
                'loss_current': self.loss_current,
                'loss_best': self.loss_best,
                'best_Epoch': self.best_Epoch,
                'cnt_Epoch': self.cnt_Epoch,
                'loss_history': self.loss_history
                }
        
        
        th.save(data, os.path.join(rootPath, filePath, fileName))
        