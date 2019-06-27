# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:09:27 2019

@author: 吳嵩裕
"""

import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.utils import shuffle


# Define the basic conv unit
def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)

class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape = same_shape
        stride=1 if self.same_shape else 2
        
        self.conv1 = conv3x3(in_channel, out_channel, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        
        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, 1, stride=stride)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)
        
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(x+out, True)

# Define the ResNet
class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Conv2d(in_channel, 64, 3, 2)
        
        self.block2 = nn.Sequential(
            nn.MaxPool2d(3, 2),
            residual_block(64, 64),
            residual_block(64, 64)
        )
        
        self.block3 = nn.Sequential(
            residual_block(64, 128, False),
            residual_block(128, 128)
        )
        
        self.block4 = nn.Sequential(
            residual_block(128, 256, False),
            residual_block(256, 256)
        )
        
        self.block5 = nn.Sequential(
            residual_block(256, 512, False),
            residual_block(512, 512),
            nn.AvgPool2d(3)
        )
        
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        if self.verbose:
            print('block 1 output: {}'.format(x.shape))
        x = self.block2(x)
        if self.verbose:
            print('block 2 output: {}'.format(x.shape))
        x = self.block3(x)
        if self.verbose:
            print('block 3 output: {}'.format(x.shape))
        x = self.block4(x)
        if self.verbose:
            print('block 4 output: {}'.format(x.shape))
        x = self.block5(x)
        if self.verbose:
            print('block 5 output: {}'.format(x.shape))
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x




# Construct a VGG Stack of 5 VGG Blocks
net = resnet(1,25)

print(net)

use_gpu = torch.cuda.is_available()
if use_gpu:
    net = net.cuda()
    print ('USE GPU')
else:
    print ('USE CPU')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print ("1. Loading data")
train1 = pd.read_csv("train_data.csv").values
train = shuffle(train1)
test  = pd.read_csv("test_data.csv").values

print ("2. Converting data")
X_data  = train[:, 1:].reshape(train.shape[0], 1, 28, 28)
#把X_data從27455*1*28*28接成27648+1024*1*28*28，才比較好調batchsize
X_data = np.append(X_data,train[0:193+1024, 1:].reshape(193+1024, 1, 28, 28),axis=0)
X_data  = X_data.astype(float)
X_data /= 255.0
X_data  = torch.from_numpy(X_data);
X_label = train[:,0];
X_label = np.append(X_label,train[0:193+1024, 0],axis=0)
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0]+193+1024,-1);
print (X_data.size(), X_label.size())

print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 1000000
nb_index = 0
nb_batch = 1024

for epoch in range(nb_epoch):
    if(nb_index + nb_batch >= nb_train):
        nb_index = 0
    else:
        nb_index = nb_index + nb_batch

    mini_data  = Variable(X_data[nb_index:(nb_index+nb_batch)].clone())
    mini_label = Variable(X_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
    mini_data  = mini_data.type(torch.FloatTensor)
    mini_label = mini_label.type(torch.LongTensor)
    if use_gpu:
        mini_data  = mini_data.cuda()
        mini_label = mini_label.cuda()
    optimizer.zero_grad()
    mini_out   = net(mini_data)
    mini_out   = mini_out.view(mini_out.size(0),-1)
    mini_label = mini_label.view(nb_batch)
    mini_loss  = criterion(mini_out, mini_label)
    mini_loss.backward()
    optimizer.step()
    correct = int(torch.sum(torch.argmax(mini_out, dim=1) == mini_label))
    a=correct/nb_batch
    b=mini_loss.data
#    if(b<0.001 or a>0.999):
#        print("Epoch = %d, Loss = %10f Accuracy = %10f" %(epoch+1, mini_loss.data, correct/nb_batch))
#        break;
    if (epoch + 1) % 1 == 0:
        print("Epoch = %d, Loss = %10f Accuracy = %10f" %(epoch+1, mini_loss.data, correct/nb_batch))

print ("4. Testing phase")

Y_data  = test.reshape(test.shape[0], 1, 28, 28)
Y_data  = Y_data.astype(float)
Y_data /= 255.0
Y_data  = torch.from_numpy(Y_data);
print (Y_data.size())
nb_test = test.shape[0]

net.eval()

final_prediction = np.ndarray(shape = (nb_test, 2), dtype=int)
for each_sample in range(nb_test):
	sample_data = Variable(Y_data[each_sample:each_sample+1].clone())
	sample_data = sample_data.type(torch.FloatTensor)
	if use_gpu:
		sample_data = sample_data.cuda()
	sample_out = net(sample_data)
	_, pred = torch.max(sample_out, 1)
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred.data[0]
	if (each_sample + 1) % 2000 == 0: 
		print("Total tested = %d" %(each_sample + 1))

print ('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['samp_id', 'label'])
submission.to_csv('pytorch_LeNet.csv', index=False, header=True)

# 保存模型
torch.save(net.state_dict(), './cnn.pth')