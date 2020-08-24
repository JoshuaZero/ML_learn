#!env python
#coding=utf-8
# Author:  joshua_zero@outlook.com


import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt 

 
class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)


    def forward(self,x):
        x = torch.relu(self.hidden(x))
        v_out = self.predict(x)
        return v_out

def net_test():

    
    x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1) 
    y = x.pow(2) + 0,2*torch.rand(x.size())
    

    net =Net(1,10,1)
    print(net)
    
    plt.ion()
    plt.show()

    optimizer = torch.optim.SGD(net.parameters(),lr=0.5)
    loss_func = torch.nn.MSELoss()
    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if t%5 == 0:
            #plot and show learning process
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(),'r-', lw=5)
            plt.text(0.5,0,'Loss=%.4f'% loss.data[0],fontdict={'size':20, 'color':'red'}) 
            plt.pause(1)
    plt.ioff()
    plt.show()



if __name__=="__main__":
    net_test()





