#!env python
#coding=utf-8
# Author:  joshua_zero@outlook.com

import torch 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt  

def compare_numpy_torch():
    np_data = np.arange(6).reshape((2,3))
    torch_data = torch.from_numpy(np_data)
    torch2array = torch_data.numpy() 
    print(
       '\n numpy',np_data,
        '\n torhch', torch_data,
        '\n tensor2array',torch2array,
    )

def torch_math():
    #abs
    data = [-1,-2,1,2]
    tensor = torch.FloatTensor(data)

    print(
        '\n abs',
        '\n mumpy',np.abs(data),
        '\n torch ',torch.abs(tensor),
        '\n mumpy',np.sin(data),
        '\n torch ',torch.sin(tensor),
        '\n mumpy',np.mean(data),
        '\n torch ',torch.mean(tensor),
    )

    data_arr = [[1,2],[3,4]]
    tensor = torch.FloatTensor(data_arr)
    data_np = np.array(data_arr)

    print(
        '\n numpy matmul:', np.matmul(data_arr, data_arr),
        '\n numpy dot:', data_np.dot(data_np),
        '\n torch mm:', torch.mm(tensor, tensor),
    )

def variable_test():
    tensor = torch.FloatTensor([[1,2],[3,4]])
    variable = Variable(tensor, requires_grad=True)
    t_out = torch.mean(tensor*tensor)
    v_out = torch.mean(variable*variable)

    v_out.backward()
    #v_out = 1/4*sum(var*var)
    #d(v_out)/d(var) = 1/4*2*variable = variable/2
    print(
        '\n tensor:', tensor, 
        '\n variable:', variable, 
        '\n t_out:',t_out,
        '\n v_out:', v_out,
        '\n variable grad', variable.grad,
        '\n variable', variable,
        '\n variable data numpy', variable.data.numpy(),

    )

def activated_func():
  #fack data
    x = torch.linspace(-5,5,200)  #x data(tensor),shape=(100,1) 
    x = Variable(x)
    x_np = x.data.numpy()

    y_relu = F.relu(x).data.numpy()
    #y_sigmoid = F.sigmoid(x).data.numpy()
    y_sigmoid = torch.sigmoid(x).data.numpy()
    #y_tanh = F.tanh(x).data.numpy()
    y_tanh = torch.tanh(x).data.numpy()
    y_softplus= F.softplus(x).data.numpy()

    plt.figure(1, figsize=(8,6))
    plt.subplot(221)
    plt.plot(x_np,y_relu, c='red', label='relu')
    plt.ylim((-1,5))
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(x_np,y_sigmoid, c='red', label='sigmoid')
    plt.ylim((-0.2,1.2))
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(x_np,y_tanh, c='red', label='tanh')
    plt.ylim((-1.2,1.2))
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(x_np,y_softplus, c='red', label='softplus')
    plt.ylim((-0.2,6))
    plt.legend(loc='best')

    plt.show()

def regress_test():
    x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1) 
    y = x.pow(2) + 0,2*torch.rand(x.size())
    print(x,'\n',y)
    xv = Variable(x)
    yv = Variable(y)

    plt.scatter(xv.data.numpy(), yv.data.numpy())
    plt.show()



if __name__=="__main__":
    #compare_numpy_torch()
    #torch_math()
    #variable_test()
    #activated_func()
    regress_test()
