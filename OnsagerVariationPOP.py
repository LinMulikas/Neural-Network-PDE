import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers

from numpy import linalg as LA
# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)

def vecterize(weights):
    '''
    transfer the weight tensor to a vector
        '''

    v=tf.reshape(weights[0],shape=[-1,1])
    for weight in weights[1:]:
        v=tf.concat([v,tf.reshape(weight,shape=[-1,1])],axis=0)
    return v
def tranfer_weight(vector,nums=40):

    '''
    transfer the vector tensor to the weight tensor,default NN has 3 hiddlen layer with resnet structure.
    '''

    res=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    res[0]=tf.reshape(vector[0:nums*1],[1,nums])
    res[1]=tf.reshape(vector[nums*1:nums*2],[nums])

    res[2]=tf.reshape(vector[nums*2:nums*(2+nums)],[nums,nums])
    res[3]=tf.reshape(vector[nums*(2+nums):nums*(3+nums)],[nums])

    res[4]=tf.reshape(vector[nums*(3+nums):nums*(2*nums+3)],[nums,nums])
    res[5]=tf.reshape(vector[nums*(2*nums+3):nums*(2*nums+4)],[nums])

    res[6]=tf.reshape(vector[nums*(2*nums+4):nums*(3*nums+4)],[nums,nums])
    res[7]=tf.reshape(vector[nums*(3*nums+4):nums*(3*nums+5)],[nums])

    res[8]=tf.reshape(vector[nums*(3*nums+5):nums*(4*nums+5)],[nums,nums])
    res[9]=tf.reshape(vector[nums*(4*nums+5):nums*(4*nums+6)],[nums])

    res[10]=tf.reshape(vector[nums*(4*nums+6):nums*(5*nums+6)],[nums,nums])
    res[11]=tf.reshape(vector[nums*(5*nums+6):nums*(5*nums+7)],[nums])

    res[12]=tf.reshape(vector[nums*(5*nums+7):nums*(5*nums+8)],[nums,1])
    res[13]=tf.reshape(vector[nums*(5*nums+8):],[1])
    return res

def cus_model(num_neurons_per_layer=40,activation='tanh',Resnet=False):
    # define the layers
    x_in = Input(shape=(1,))
    
    x1=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                                         kernel_initializer='glorot_normal')(x_in)
    x1_=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                                         kernel_initializer='glorot_normal')(x1)
    x2=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                                         kernel_initializer='glorot_normal')(x1_)
    x2_=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                                         kernel_initializer='glorot_normal')(x2)

    x3_=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                                         kernel_initializer='glorot_normal')(x2_)
     
    x3=Dense(num_neurons_per_layer,
            activation=tf.keras.activations.get(activation),
                kernel_initializer='glorot_normal')(x3_)
    if Resnet:
    	x_add=layers.Add()([x3, x_in])
    	x_out_=Dense(1)(x_add)
    else:
    	x_out_=Dense(1)(x3)

   
    x_out=(x_in*x_in-1)*(x_in*x_in-1)*x_out_-1
    model=Model(inputs=x_in, outputs=x_out)
    return model

@tf.function
def mix_gradient(node,model):

    '''
        Bulid a computational graph to calculate the every order derivates for the input
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(node)
        tape.watch(model.trainable_variables)
        u=model(node)
        u_x=tape.gradient(u,node)
    u_theta=tape.gradient(u,model.trainable_variables)
    u_xtheta=tape.gradient(u_x,model.trainable_variables)

    del tape
    return u_x,vecterize(u_theta),vecterize(u_xtheta)

def gradient_node(node,model,dim=1):
	'''after set the weight of model is theta_k, calculate the laplacian u  with respect to x'''
	if dim==1:
		with tf.GradientTape(persistent=True) as tape:
			tape.watch(node)
			u=model(node)
			u_x=tape.gradient(u,node)
		u_xx=tape.gradient(u_x,node)
		del tape
		return u,u_xx
	if dim==2:
		pass
			

@tf.function
def gradient_theta(x,model):

	''' 
	u_theta
	Build the graph to calculate the derivatives respect to the weight of NN at different input points.
	'''
	  
	u = model(x)
	u_theta = tf.gradients(u, model.trainable_variables)
	return vecterize(u_theta)


def calculate_coefs_laplacian(sample_size,model,source_f):
    Coef_matrix=0
    Coef_b=0
    sample=tf.random.uniform([sample_size,1],-1,1)
    u, Laplace_u = gradient_node(sample, model)
    # u=model(sample)
    f=source_f(sample,u)#source function

    #calculate the u_theta at different data points and calculate the Monte Carlo integration
    #This step can be accelerated with parallel operations， By map in Python (但是会很吃内存)
    for i in range(sample_size):
        x=sample[i]
        u_theta=gradient_theta(x,model)

        Coef_matrix+=2*(tf.matmul(u_theta,tf.transpose(u_theta)))/sample_size

        Coef_b+=(Laplace_u[i]+f[i])*u_theta*2/sample_size
    return Coef_matrix,-tf.reshape(Coef_b,[-1,1])


def calculate_coefs(sample_size,model,source_f):
    Coef_matrix=0
    Coef_b=0
    sample=tf.random.uniform([sample_size,1],-1,1)
    # u,Laplace_u=self.gradient_node(sample)
    u=model(sample)
    f=source_f(sample,u)#source function

    #calculate the u_theta at different data points and calculate the Monte Carlo integration
    #This step can be accelerated with parallel operations， By map in Python (但是会很吃内存)
    for i in range(sample_size):
        x=sample[i]
        u_x,u_theta,u_xtheta=mix_gradient(x,model)

        Coef_matrix+=2*(tf.matmul(u_theta,tf.transpose(u_theta)))/sample_size

        Coef_b+=(u_x*u_xtheta+f[i]*u_theta)*2/sample_size
    return Coef_matrix,-tf.reshape(Coef_b,[-1,1])


def move_part_step(Coef_M,Coef_b,lam,para_nums=8321):
    A=Coef_M+lam*tf.linalg.diag(tf.ones(para_nums))
    return tf.matmul(tf.linalg.inv(A),Coef_b)

def Solve_regularization_system_tf(Coef_M,Coef_b,lam,para_nums=8321,dt=0.001):
    A=Coef_M+lam*tf.linalg.diag(tf.ones(para_nums))
    return dt*tf.matmul(tf.linalg.inv(A),Coef_b)

def Solve_regularization_system(Coef_M,Coef_b,lam,para_nums=8321,dt=0.001):
    A=Coef_M+lam*tf.linalg.diag(tf.ones(para_nums))
    return dt*LA.inv(A)@Coef_b


