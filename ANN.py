# -*- coding: utf-8 -*-
"""
Created on Fri May 10 02:18:03 2019

@author: 吳嵩裕
"""

import pandas as pd
import numpy as np
import torch

from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from sklearn.utils import shuffle


net = torch.nn.Sequential()
net.add_module("dropout1",torch.nn.Dropout(p=0.2))
net.add_module("dense1",torch.nn.Linear(784,500))
net.add_module("relu1",torch.nn.ReLU())
net.add_module("dropout2",torch.nn.Dropout(p=0.5))
net.add_module("dense2",torch.nn.Linear(500,300))
net.add_module("relu2",torch.nn.ReLU())
net.add_module("dropout3",torch.nn.Dropout(p=0.5))
net.add_module("dense3",torch.nn.Linear(300,25))


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
X_data  = X_data.astype(float)
X_data /= 255.0
X_data  = torch.from_numpy(X_data);
X_label = train[:,0];
X_label = X_label.astype(int);
X_label = torch.from_numpy(X_label);
X_label = X_label.view(train.shape[0],-1);
print (X_data.size(), X_label.size())

print ("3. Training phase")
nb_train = train.shape[0]
nb_epoch = 1000000
nb_index = 0
nb_batch = 5491

for epoch in range(nb_epoch):
    if nb_index + nb_batch >= nb_train:
        nb_index = 0
    else:
        nb_index = nb_index + nb_batch

    mini_data  = Variable(X_data[nb_index:(nb_index+nb_batch)].clone())
    mini_label = Variable(X_label[nb_index:(nb_index+nb_batch)].clone(), requires_grad = False)
    mini_data  = mini_data.type(torch.FloatTensor)
    mini_label = mini_label.type(torch.LongTensor)
    mini_data  = mini_data.view(mini_data.size(0),-1)
    if use_gpu:
        mini_data  = mini_data.cuda()
        mini_label = mini_label.cuda()
    optimizer.zero_grad()
    mini_out   = net(mini_data)
    mini_label = mini_label.view(nb_batch)
    mini_loss  = criterion(mini_out, mini_label)
    mini_loss.backward()
    optimizer.step()
    correct = int(torch.sum(torch.argmax(mini_out, dim=1) == mini_label))
    a=correct/nb_batch
    b=mini_loss.data
    if(b<0.1 or a>0.98):
        print("Epoch = %d, Loss = %10f Accuracy = %10f" %(epoch+1, mini_loss.data, correct/nb_batch))
        break;
    if (epoch + 1) % 100 == 0:
        print("Epoch = %d, Loss = %10f Accuracy = %10f" %(epoch+1, mini_loss.data, correct/nb_batch))

print ("4. Testing phase")

Y_data  = test.reshape(test.shape[0],-1)
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