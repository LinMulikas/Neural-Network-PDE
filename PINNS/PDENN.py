import time, os
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict



"""
    Use NN to approximate a function with two variables t, x.
    
    The model is the approximate function u(t, x).
    
    ##TODO: Higher dimension X input.
"""
class PDENN(nn.Module):
    dtype_default = torch.float32
    save_gap = 100
    best_Epoch = -1
    cnt_Epoch = 0
     
    loss_tensor: torch.Tensor
    loss_value = .0
    loss_history = []
    best_loss = 100.0
    loss_criterion = torch.nn.MSELoss()        
    
    PDENAME: str

    def __init__(self, 
                 input_size: int, output_size: int, 
                 hidden_size: int, hidden_depth: int,
                 lr: float, 
                 act = nn.Tanh) -> None:
        
        super().__init__()
                
        torch.set_default_dtype(PDENN.dtype_default)
        

        layers = [('input', nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        for i in range(hidden_depth): 
            layers.append(
                ('hidden_%d' % i, torch.nn.Linear(hidden_size, hidden_size))
            )
            layers.append(('activation_%d' % i, act()))
        layers.append(('output', torch.nn.Linear(hidden_size, output_size)))

        layerDict = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layerDict)
        
        self.lr = lr
        self.adam = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.lbfgs = torch.optim.LBFGS(
            self.parameters(), 
            lr=1.0, 
            max_iter=50000, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change = 1.0 * np.finfo(float).eps
        )
        
        self.optim = self.adam
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode = 'min', factor=0.2, patience=50)
        

    def forward(self, input):
        out = self.layers(input)
        return out
