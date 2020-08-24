#!env python
#coding=utf-8
# Author:  joshua_zero@outlook.com

import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

#hyper parameters

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,   #图片周围填充
            ),               #->16*28*28
            nn.ReLU(),               #->16*28*28
            nn.MaxPool2d(kernel_size=2),  #16*14*14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),  #32*14*14
            nn.ReLU(),               #32*14*14 
            nn.MaxPool2d(2)          #32*7*7
        )
        self.out = nn.Linear(32*7*7, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)       #batch 32,7,7
        x = x.view(x.size(0),-1)  #batch, 32*7*7
        output = self.out(x)
        return output

def train_minist():
    
    EPOCH =1    #
    BATCH_SIZE=50
    LR = 0.001      #learning rate 
    DOWNLOAD_MNIST = True
    
    train_data = torchvision.datasets.MNIST(
        root = './minist',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download = DOWNLOAD_MNIST
    )
    
    """
    #plot one example 
    print(train_data.data.size())
    print(train_data.targets.size())
    plt.imshow(train_data.data[0].numpy(), cmap='gray')
    plt.title('{}'.format(train_data.targets[0]))
    plt.show()
    """
    train_loader = Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(root='./minist', train=False)
    test_x = Variable(torch.unsqueeze(test_data.data,dim=1),volatile=True).type(torch.FloatTensor)[:2000]/255.
    test_y = test_data.targets[:2000]
    
    net = CNN_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH):
        for step, (b_x,b_y) in enumerate(train_loader):
            b_x = Variable(b_x)
            b_y = Variable(b_y)
            output = net(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step%50 == 0:
                test_out = net(test_x)
                pred_y = torch.max(test_out,1)[1].data.squeeze()
                accuracy = sum(pred_y==test_y) // test_y.size(0)
                print('Epoch:{}'.format(epoch), '| train loss:{}'.format(loss.item()), '| test accuracy:{}'.format(accuracy))

    test_out = net(test_x[:10])
    pred_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
    print(pred_y, "prediction num:")
    print(test_y[:10].numpy(), "real num")



if __name__=="__main__":
    train_minist()


