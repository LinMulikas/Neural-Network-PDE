import torch as tc
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

class Net(tc.nn.Module):
    device = tc.device('cuda') if(tc.cuda.is_available()) else tc.device('cpu')
        
    
    #? Net parameters.
    input_size: int
    output_size: int
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
        
        self.act = act
        #? Data generator.
        
        self.X = data[0].to(self.device)
        self.IC = data[1].to(self.device)
        self.BC = data[2].to(self.device)
        
        self.X.requires_grad_()
        self.IC.requires_grad_()
        self.BC.requires_grad_()
        
        self.lr = lr
        self.auto_lr = auto_lr
        #? Build the net.
        self.depth = shape[0]
        self.width = shape[1]
        
        self.input_size = pde_size[0]
        self.output_size = pde_size[1]
        input = nn.Linear(pde_size[0], self.width)
        hidden = []
        output = nn.Linear(self.width, pde_size[1])
        
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
        self.sched = tc.optim.lr_scheduler.ReduceLROnPlateau(
            self.adam, mode = 'min', factor=0.1, patience=100)
        
        #TODO: Load best has no current loss
        if(loadFile != ''):
            self.loadDict(loadFile)        
        

    def difTheta(self, X: tc.Tensor):
        X.requires_grad_()
        U = self(X)
        U_grad = grad(U, self.model.parameters(), tc.ones_like(U), True, True)
        vec = []
        
        for i in range(len(U_grad)):
            vec.append(tc.Tensor(U_grad[i]).reshape((-1, 1)))
            
        return tc.cat(vec)
        
        
    def paraList(self):

        biasList = []
        
        layersList = list(self.model._modules.items())
        layersModule: List[Module] = []
        
        for i in range(2 * (self.depth + 1) + 1):
            layersModule.append(layersList[i][1])
            
            
        biasList: List[Tensor] = []
        weightsList: List[Tensor] = []
        
        for i in range(len(layersModule)):
            if(i % 2 == 1):
                continue 
            else:
                layer = layersModule[i]
                biasList.append(tc.Tensor(layer.bias).reshape((-1, 1)))
                weightsList.append(tc.Tensor(layer.weight).reshape((-1, 1)))
                    
        return biasList, weightsList
        
        
    def paramVector(self) -> tc.Tensor:
        vec = []
        for param in self.model.parameters():
            vec.append(tc.Tensor(param).reshape((-1, 1)))
            
        return tc.cat(vec)
    
   
    def paramVec(self) -> tc.Tensor:
        biasList, weightsList = self.paraList()
        
        bias = tc.cat(biasList)
        weight = tc.cat(weightsList)
        
        return tc.cat((bias, weight)).reshape((-1, 1))
    
        
    def biasVec(self, biasList: List[Tensor]) -> tc.Tensor:
        bias = biasList[0].reshape((-1, ))
        
        for i in range(1, self.depth + 2):
            bias = tc.cat((bias, biasList[i].reshape((-1, ))))
        
        return bias

    def weightVec(self, weightList: List[Tensor]) -> tc.Tensor:
        weight = weightList[0].reshape((-1, 1))
        
        for i in range(1, self.depth + 2):
            weight = tc.cat((weight, weightList[i].reshape((-1, 1))))
        
        return weight
        
     
    def updateParameters(self, weights: tc.Tensor):
        bias_size = (1 + self.depth) * self.width + self.output_size
        bias = weights[:bias_size]
        weights = weights[bias_size:]
        
        self.updateBias(bias)
        self.updateWeight(weights)
    
    
    
    def updateBias(self, bias):
        inputBiasSize = self.width
        hiddenBiasSize = self.depth * self.width
        
        inputBias = bias[: inputBiasSize]
        bias = bias[inputBiasSize: ]
        hiddenBias = bias[: hiddenBiasSize]
        bias = bias[hiddenBiasSize: ]
        outputBias = bias
        
        with tc.no_grad():
            self.model._modules['0'].bias = tc.nn.Parameter(inputBias.reshape(
                (self.width,)))
        
            for i in range(1, self.depth + 1):
                self.model._modules[str(2*i)].bias = tc.nn.Parameter(hiddenBias[: self.width].reshape((self.width,)))
                hiddenBias = hiddenBias[self.width: ]
        
        
            self.model._modules[str(2*(self.depth + 1))].bias = tc.nn.Parameter(outputBias.reshape((self.output_size,)))
        
           
    
    def updateWeight(self, weights):
        inputLayerSize = self.input_size * self.width
        outputLayerSize = self.output_size * self.width
        
        inputWeights = weights[:inputLayerSize]
        weights = weights[inputLayerSize:]
        outputWeights = weights[-outputLayerSize:]
        weights = weights[:-outputLayerSize]

        hiddenWeights = weights
        with tc.no_grad():
            self.model._modules['0'].weight = tc.nn.Parameter(inputWeights.reshape(
                (self.width, self.input_size)))
        
            for i in range(1, self.depth + 1):
                layerWeight = hiddenWeights[:self.width * self.width]
                hiddenWeights = hiddenWeights[self.width * self.width: ]
                self.model._modules[str(2*i)].weight = tc.nn.Parameter(layerWeight.reshape((self.width, self.width)))
        
            self.model._modules[str(2*(self.depth + 1))].weight = tc.nn.Parameter(outputWeights.reshape((self.output_size, self.width)))
        
            
    
    def forward(self, X):
        x = X[:, 1].reshape((-1, 1))
        return (tc.square(x) - x) * self.model(X)
    
    
    def train(self, epoch, loss_fn):
        for self.cnt_Epoch in range(epoch):
            self.adam.zero_grad()
            self.adam.step(loss_fn)
            self.sched.step(self.loss_current)
            self.info()
    
    
        

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
        
        data = tc.load(filePath, map_location=self.device)
        
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
        
        
        tc.save(data, os.path.join(rootPath, filePath, fileName))
        
        
    def moveTo(self, device):
        self.to(device)
        self.X.to(device)
        self.IC.to(device)
        self.BC.to(device)