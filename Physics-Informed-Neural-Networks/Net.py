import torch as tc
import numpy as np
import os

from torch import nn
from torch.autograd import grad
from torch.autograd import grad
from typing import Tuple, List
from torch.types import Number

device = tc.device('cuda')

class Net(tc.nn.Module):
    #? Net parameters.
    depth: int
    width: int
    lr: float = 1e-2
    
    #? Data
    
    X: tc.Tensor
    BC: tc.Tensor 
    IC: tc.Tensor


    
    #? Data
    
    X: tc.Tensor
    BC: tc.Tensor 
    IC: tc.Tensor


    #? Training parameters.
    cnt_Epoch = 0
    best_Epoch = -1
    save_Gap = 5000
    
    PDENAME: str
    
    loss_best: tc.Tensor = tc.ones((1)).to(device)
    loss_current: tc.Tensor
    loss_history = []
    loss_criterion = tc.nn.MSELoss(reduce=True, reduction='mean')  
    
    def __init__(self, 
                 pde_size: Tuple[int, int], 
                 shape: Tuple[int, int], 
                 data: Tuple[tc.Tensor, tc.Tensor, tc.Tensor],
                 loadFile:str = '',
                 lr: float = 1e-2,
                 act = nn.Tanh,
                 auto_lr = True,) -> None:
        
        super().__init__()
        
        
        #? Data generator.
        
        self.X = data[0].to(device)
        self.IC = data[1].to(device)
        self.BC = data[2].to(device)
        
        self.X.requires_grad_()
        self.IC.requires_grad_()
        self.BC.requires_grad_()
        
        self.lr = lr
        self.auto_lr = auto_lr
        #? Build the net.
        self.depth = shape[0]
        self.width = shape[1]
        
        input = nn.Linear(pde_size[0], self.width)
        hiden = []
        output = nn.Linear(self.width, pde_size[1])
        
        for i in range(self.depth):
            hiden.append(nn.Linear(self.width, self.width))
            hiden.append(act())
        
        self.model = nn.Sequential(
            input,
            *hiden,
            output
        )
        
        #? Build the optimizer.
        self.adam = tc.optim.Adam(self.parameters(), self.lr)
        self.sgd = tc.optim.SGD(self.parameters(), self.lr)
        self.lbfgs = tc.optim.LBFGS(
            self.parameters(),
            lr = self.lr,
            max_iter=int(5e3),
            max_eval=int(5e3),
            tolerance_grad=1e-5,
            tolerance_change=1e-9
        )
        self.optim = self.adam
        self.sched = tc.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode = 'min', factor=0.1, patience=100)
        
        #TODO: Load best has no current loss
        if(loadFile != ''):
            self.loadDict(loadFile)
            
            
    
    def forward(self, x):
        return self.model(x)
    
    
    def train(self, epoch):
        for self.cnt_Epoch in range(epoch):
            self.optim.step(self.loss)
            # if(self.auto_lr):
            #     self.sched.step(self.loss_current)
                
            self.info()
    
    def loss(self):
        self.optim.zero_grad()
        
        X = tc.cat((
            self.X,
            self.BC,
            self.IC
        ))
        X.requires_grad_()
        
        U = self(X)
        dU = grad(U, X, tc.ones_like(U), True, True)[0]
        
        pt = dU[:, 0].reshape((-1, 1))
        px = dU[:, 1].reshape((-1, 1))
        
        
        #? Loss_PDE
        
        loss_pde = self.loss_criterion(pt, tc.square(px))
        
        #? Loss_IC
        eq_ic = self(self.IC)
        loss_ic = self.loss_criterion(
            eq_ic, 
            tc.sin(np.pi * self.IC[:, 1].reshape((-1, 1))))
        
        
        #? Loss_BC
        eq_bc = self(self.BC)
        loss_bc = self.loss_criterion(
            eq_bc, 
            tc.zeros_like(eq_bc))
        
    
        
        #? Calculate the Loss
        loss = loss_pde + 5 * loss_ic + 5 * loss_bc
        loss.backward()
        
        self.cnt_Epoch = self.cnt_Epoch + 1 
        self.loss_current = loss.clone()
        self.loss_history.append(self.loss_current.item())
        
        return loss
    
    
    
        
    def info(self):
        #? Save Best
        
        if(tc.equal(self.loss_best, tc.ones((1)))):
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
            rootPath = os.getcwd()
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
        rootPath = os.getcwd()
        filePath = os.path.join(rootPath, fileName)
        
        data = tc.load(filePath)
        
        self.load_state_dict(data['dict'], True)
        self.loss_current = data['loss_current']
        self.best_Epoch = data['best_Epoch']
        self.loss_best = data['loss_best']
        self.cnt_Epoch = data['cnt_Epoch']
        self.loss_history = data['loss_history']
        
        if(fileName == 'models/best.pt'):
            self.loss_best = self.loss_current
    
        
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
        
        
        tc.save(data, os.path.join(rootPath, filePath, fileName))