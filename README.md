# Neural-Network-PDE

# 0.Introduction

## Author
The repository Neural-Network-PDE belongs to LinMulikas, who is an UG student at SUSTech (Southern University of Science and Technology). Any question please contact wangdl2020@mail.sustech.edu.cn for academic purposes.

## Repository
The repository is built for self use corresponding to PDE solving in Neural Network with Pytorch. The python version is 3.9.16.

# PINNS

## 1. Framework introduction.
Implement of PINNS method in PDE, I've build a basic construciton 'PDENN' as the NN for PDE. User need to inherit the PDENN class to create a brand new PDE, which requires the definition of loss(), which can be represente as loss = loss_PDE + loss_BC + loss_IC.

And I've build PDE2D as a general framework for u(t, x), 2D PDE solving. Including a PDE-drawer, loss history recorder, some useful method to divides the region.

## 2. Build new PDEs.


## 3. Train and load.

