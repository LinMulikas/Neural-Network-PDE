from Net import *
from Heat import Heat
from PDE_Square import *
from torch.types import Number
import torch as tc
from pyDOE import lhs as LHS
from typing import Literal

import matplotlib.pyplot as plt

class NeuralGalerkin:
    pde: PDE_Square
    N = 1000
    
    def __init__(self, pde) -> None:
        self.pde = pde
    
    def trainInit(self, epoch: int):
        self.pde.net.train(epoch, self.loss_theta)
            
    def thetaStep(self, t: float, N_t: int, N_x: int, alpha):
        h = (t - self.pde.t[0])/N_t
        t_current = self.pde.t[0]

        cnt = 0
        while(t_current <= t):
            M, F, theta = self.calculateCoefs(t_current, N_x, 'lhs')
            I = tc.eye(M.shape[0])
            
            theta_dot = tc.linalg.solve(M + alpha * I, F)
            theta = theta + h * theta_dot

            self.pde.net.updateParams(theta)
            
            t_current += h
            
            if(t_current <= t):
                cnt += 1
                print("Step {}, t finished at {}".format(cnt, t_current))
               
               


    
    def loss_theta(self) -> float:
        x_sample = tc.Tensor(np.array(np.sort(np.array(LHS(1, self.N)), axis=0)))
        t_sample = self.pde.t[0] * tc.ones_like(x_sample).reshape((-1, 1))
        
        ic_sample = tc.hstack((t_sample, x_sample)).to(self.pde.net.device)
        loss = tc.mean(
            tc.square(
                self.pde.u0(ic_sample) - self.pde.net(ic_sample)))
        
        loss.backward()
        self.pde.net.loss_current = loss.clone()
        self.pde.net.cnt_Epoch = self.pde.net.cnt_Epoch + 1 
        self.pde.net.loss_history.append(self.pde.net.loss_current.item())
        
        return loss.item()
    
    
    
    def calculateCoefs(self, t, N_x, sampling: Literal['uni', 'lhs']) -> Tuple[Tensor, Tensor, Tensor]:
        x_lhs = self.pde.x[0]
        x_rhs = self.pde.x[1]
        
        if(sampling == 'uni'):
            x_sample = self.pde.x[0] + (self.pde.x[1] - self.pde.x[0]) * tc.arange(x_lhs, x_rhs, 1.0/N_x).reshape((-1, 1))
            x_step = 1.0 * (x_rhs - x_lhs)/N_x
            
            data_sample = tc.hstack((t * tc.ones_like(x_sample), x_sample)).to(device = self.pde.net.device).reshape((-1, 2))
            data_sample.requires_grad_()
            
            M_list = []
            F_list = []
            coef = []
            
            for i in range(1, N_x + 1):
                if(i == 1 or i == N_x):
                    coef.append(1)
                else:
                    if(i % 2 == 0):
                        coef.append(4)
                    else:
                        coef.append(2)
                
            theta = self.pde.net.paramVector()
            
            M = tc.zeros((theta.shape[0], theta.shape[0]))
            F = tc.zeros((theta.shape[0], 1))
            
            for i in range(N_x):
                data_input = data_sample[i, :].reshape((1, -1))
                U_theta = self.pde.net.difParams(data_input)
                U = self.pde.net(data_input)
                U.requires_grad_()
                U_x = grad(U, data_input, tc.ones_like(U), True, True)[0][:, 1]
                U_xx = grad(U_x, data_input, tc.ones_like(U_x), True, True)[0][:, 1]
                _M = U_theta @ U_theta.T
                _F = U_theta * U_xx
                
                M += x_step/3 * coef[i] * _M
                F += x_step/3 * coef[i] * _F
                
            return M, F, theta
        
        elif(sampling == 'lhs'):
            x_sample = self.pde.x[0] + (self.pde.x[1] - self.pde.x[0]) * Tensor(np.array(np.sort(np.array(LHS(1, N_x)), axis=0))).reshape((-1, 1))
            x_step = 1.0 * (x_rhs - x_lhs)/N_x
            
            data_sample = tc.hstack((t * tc.ones_like(x_sample), x_sample)).to(device = self.pde.net.device).reshape((-1, 2))
            data_sample.requires_grad_()
            
            M_list = []
            F_list = []
            coef = []
            
            for i in range(1, N_x + 1):
                coef.append(1)
                
            theta = self.pde.net.paramVector()
            
            M = tc.zeros((theta.shape[0], theta.shape[0]))
            F = tc.zeros((theta.shape[0], 1))
            
            for i in range(N_x):
                data_input = data_sample[i, :].reshape((1, -1))
                U_theta = self.pde.net.difParams(data_input)
                U = self.pde.net(data_input)
                U.requires_grad_()
                U_x = grad(U, data_input, tc.ones_like(U), True, True)[0][:, 1]
                U_xx = grad(U_x, data_input, tc.ones_like(U_x), True, True)[0][:, 1]
                _M = U_theta @ U_theta.T
                _F = U_theta * U_xx
                
                M += x_step * _M 
                F += x_step * _F
                
            return M, F, theta
            
            
        

    
    def drawPred(self, t):
        xlhs = self.pde.x[0]
        xrhs = self.pde.x[1]
        xline = tc.arange(xlhs, xrhs, 1/self.N).reshape((self.N, 1))
        tline = self.pde.t[0] * tc.ones_like(xline)
        
        X = tc.hstack((tline, xline))
        u_pred = tc.Tensor.cpu(self.pde.net(X)).detach().numpy()
        xline = tc.Tensor.cpu(xline)
        plt.plot(xline, u_pred)
        plt.title("Pred solution at t = {}".format(t))
        
    def draw(self, t):
        xlhs = self.pde.x[0]
        xrhs = self.pde.x[1]
        xline = tc.arange(xlhs, xrhs, 1/self.N).reshape((self.N, 1))
        tline = t * tc.ones_like(xline)
        
        X = tc.hstack((tline, xline))
        u_pred = tc.Tensor.cpu(self.pde.net(X)).detach().numpy()
        xline = tc.Tensor.cpu(xline)
        plt.plot(xline, u_pred, label = 'Predict')
        
        u_real = tc.Tensor.cpu(self.pde.real(X))
        plt.plot(xline, u_real, label = 'Real')
        plt.title("Comparison at t = {}".format(t))
        plt.legend()
        plt.show()
        
        
        
    def drawReal(self, t):
        xlhs = self.pde.x[0]
        xrhs = self.pde.x[1]
        xline = tc.Tensor.cpu(tc.arange(xlhs, xrhs, 1/self.N).reshape((self.N, 1)))
        tline = self.pde.t[0] * tc.ones_like(xline)
        
        X = tc.Tensor.cpu(tc.hstack((tline, xline)))
        u_real = tc.Tensor.cpu(self.pde.real(X))
        plt.plot(xline, u_real)
        plt.title("Real solution at t = {}".format(t))
        plt.show()
        