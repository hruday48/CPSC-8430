
pip install tensorflow==2.4

# Commented out IPython magic to ensure Python compatibility.
# This file calculates gradient norm during the training of a DNN 

import tensorflow as tf
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


print(tf.__version__)




class M (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val


def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val
print(calcParams(M()))

simulatedInput = 20 * torch.rand((1000, 1)) - 10
groundTruth = np.sin(1.5*simulatedInput)

class GradientNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 15)
        self.fc3 = nn.Linear(15, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = self.fc4(val)
        return val


gradNet = GradientNN()
costFunc = nn.MSELoss()
opt = optim.Adam(gradNet.parameters(), lr=0.001)
EPOCHS = 2000

 
costList = []
gradNormList = []
counterList = []
counter = 1
for index in range(EPOCHS):
    counterList.append(counter)
    counter += 1
    gradNet.zero_grad()
    output = gradNet(simulatedInput)
    cost = costFunc(output, groundTruth)
    costList.append(cost)
    cost.backward()
    opt.step() 
    
   
    gradAll = 0.0
    for p in gradNet.parameters():
        grad = 0.0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()
        gradAll += grad
    gradNorm = gradAll ** 0.5
    gradNormList.append(gradNorm)


plt.plot(counterList, costList, 'b', label='Model')
plt.title("loss for y=sin(1.5x)")
plt.xlabel("epochs")
plt.ylabel("loss")

plt.show()


plt.plot(counterList, gradNormList, 'b', label='Model')
plt.title("Gradient Norm for y=sin(1.5x)")
plt.xlabel("epochs")
plt.ylabel("grad")

plt.show()





