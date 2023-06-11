from Net import *
from Heat import Heat
from PDE_Square import *
from torch.types import Number
import torch as tc
from pyDOE import lhs as LHS

import matplotlib.pyplot as plt

class NeuralGalerkin:
    pde: PDE_Square
    N = 1000
    
    def __init__(self, pde) -> None:
        self.pde = pde
    
    def trainInit(self, epoch: int):
        self.pde.net.train(epoch, self.loss_theta)
            
    def thetaStep(self, t: float, N_t: int, N_x: int, alpha = 1e-4):
        h = (t - self.pde.t[0])/N_t
        theta = self.pde.net.paramVec()
        
        t_current = self.pde.t[0]
        while(t_current <= t):
            M, F = self.calMF(t_current, N_x)
            I = tc.eye(M.shape[0])
            
            theta_dot = tc.linalg.solve(M + alpha * I, F)
            theta = theta + h * theta_dot

            self.pde.net.updateParameters(theta)
            
            print("t step at {}".format(t_current + h))
            t_current += h


    
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
    
    
    def calMF(self, t, N_x) -> Tuple[Tensor, Tensor]:
        x_sample = self.pde.x[0] + (self.pde.x[1] - self.pde.x[0]) * tc.Tensor(np.array(np.sort(np.array(LHS(1, self.N)), axis=0)))
        
        theta = self.pde.net.paramVec()
        theta.requires_grad_()
        
        M = tc.zeros((theta.shape[0], theta.shape[0]))
        F = tc.zeros((theta.shape[0], 1))

        for xi in list(x_sample):
            data_sample = tc.Tensor((t, xi)).to(device=self.pde.net.device).reshape((1, 2))
            data_sample.requires_grad_()
            
            U_theta = self.pde.net.difTheta(data_sample)
            
            U = self.pde.net(data_sample)
            U.requires_grad_()
            U_x = grad(U, data_sample, tc.ones_like(U), True, True)[0][:, 1]
            U_xx = grad(U_x, data_sample, tc.ones_like(U_x), True, True)[0][:, 1]
            
            F += (U_xx * U_theta)/N_x
            M += tc.mul(U_theta, U_theta.T)/N_x


        return M, F
        

    
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
        