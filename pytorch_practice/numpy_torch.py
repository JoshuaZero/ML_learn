#!env python
#coding=utf-8
# Author:  joshua_zero@outlook.com

import torch 
import numpy as np

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


if __name__=="__main__":
    #compare_numpy_torch()
    torch_math()
