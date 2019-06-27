# -*- coding: utf-8 -*-
"""
Created on Fri May 10 13:56:21 2019

@author: 吳嵩裕
"""

import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.utils import shuffle


import sys
sys.path.append('..')


def vgg_block(num_convs, in_channels, out_channels):
    net = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.ReLU(True)] # Define the 1st Layer
    
    for i in range(num_convs-1): # Define the subsequent layers
        net.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        net.append(nn.ReLU(True))
        #net.append(nn.Dropout2d())
        
    net.append(nn.MaxPool2d(2, 2)) # Define a pooling layer
    return nn.Sequential(*net) # return the model

## Construct a VGG Block
#block_demo = vgg_block(1, 64, 128) # 3 layers, 64 in_channels, 128 out_channels
#
## Test with an initial input
#input_demo = Variable(torch.zeros(1, 64, 300, 300)) # input of all zeros
#output_demo = block_demo(input_demo)
#print(output_demo.shape)

# Define a stack of VGG blocks
def vgg_stack(num_convs, channels):
    net = []
    for n, c in zip(num_convs, channels):
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n, in_c, out_c))
    return nn.Sequential(*net)

# Construct a VGG Stack of 5 VGG Blocks
net = vgg_stack((1, 2, 3), ((1, 32), (32, 64), (64, 25)))

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
nb_epoch = 24000
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
#    if(b<0.01 or a>0.99):
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
	pred = int(torch.argmax(sample_out))
	final_prediction[each_sample][0] = 1 + each_sample
	final_prediction[each_sample][1] = pred
	if (each_sample + 1) % 2000 == 0: 
		print("Total tested = %d" %(each_sample + 1))

print ('5. Generating submission file')

submission = pd.DataFrame(final_prediction, dtype=int, columns=['samp_id', 'label'])
submission.to_csv('vgg6.csv', index=False, header=True)

# 保存模型
torch.save(net.state_dict(), './vgg6.pth')