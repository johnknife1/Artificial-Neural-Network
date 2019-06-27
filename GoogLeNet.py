# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:30:38 2019

@author: 吳嵩裕
"""

import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.utils import shuffle

# Define a simple block comprising a conv2d, batch normalization, and a relu
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    layer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True)
    )
    return layer

# Define the Inception Net
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5, out4_1):
        super(inception, self).__init__()
        # Route 1
        self.branch1x1 = conv_relu(in_channel, out1_1, 1)
        
        # Route 2
        self.branch3x3 = nn.Sequential( 
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1)
        )
        
        # Route 3
        self.branch5x5 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2)
        )
        
        # Route 4
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1)
        )
        
    def forward(self, x):
        f1 = self.branch1x1(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        f4 = self.branch_pool(x)
        output = torch.cat((f1, f2, f3, f4), dim=1)
        return output

class googlenet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(googlenet, self).__init__()
        self.verbose = verbose
        
        self.block1 = nn.Sequential(
            conv_relu(in_channel, out_channel=64, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            conv_relu(64, 64, kernel=1),
            conv_relu(64, 192, kernel=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(2, 2)
        )
        
        self.block4 = nn.Sequential(
            inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(2, 2)
        )
        
        self.block5 = nn.Sequential(
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128),
            nn.AvgPool2d(2)
        )
        
        self.classifier = nn.Linear(1024, num_classes)
        
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
net = googlenet(1,25)

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
nb_batch = 64

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