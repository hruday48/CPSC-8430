# -*- coding: utf-8 -*-

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


# 3 models to implement various models
class ShallowSimNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 434)
        self.fc2 = nn.Linear(434, 1)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = self.fc2(val)
        return val


class MiddleSimNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 40)
        self.fc3 = nn.Linear(40,20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = self.fc4(val)
        return val    
  
class DeepSimNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, 18)
        self.fc4 = nn.Linear(18, 9)        
        self.fc5 = nn.Linear(9, 3)
        self.fc6 = nn.Linear(3, 1)        
        
    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = F.relu(self.fc4(val))
        val = F.relu(self.fc5(val))
        val = self.fc6(val)
        return val
        
np.random.seed(10)
simulatedInput = 10 * torch.rand((1000, 1)) - 5
x=simulatedInput
groundTruthSin = (np.sin(1.5*x))
groundTruthCos = np.cos(0.5*x)

def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val
model1 = ShallowSimNN()
print(calcParams(MiddleSimNN()))
print(calcParams(DeepSimNN()))
print(calcParams(ShallowSimNN()))


shallowCosNN = ShallowSimNN()
middleCosNN = MiddleSimNN()
deepCosNN = DeepSimNN()
shallowSinNN = ShallowSimNN()
middleSinNN = MiddleSimNN()
deepSinNN = DeepSimNN()
costFunc = nn.MSELoss()
shallowCosOpt = optim.Adam(shallowCosNN.parameters(), lr=0.001)
middleCosOpt = optim.Adam(middleCosNN.parameters(), lr=0.001)
deepCosOpt = optim.Adam(deepCosNN.parameters(), lr=0.001)
shallowSinOpt = optim.Adam(shallowSinNN.parameters(), lr=0.001)
middleSinOpt = optim.Adam(middleSinNN.parameters(), lr=0.001)
deepSinOpt = optim.Adam(deepSinNN.parameters(), lr=0.001)


EPOCHS = 5000
counter = 0
counterList = []
shallowCosCostList = []
for index in range(EPOCHS):
    counterList.append(counter)
    counter += 1
    shallowCosNN.zero_grad()
    output = shallowCosNN(simulatedInput)
    cost = costFunc(output, groundTruthCos)
    shallowCosCostList.append(cost)
    cost.backward()
    shallowCosOpt.step()

middleCosCostList = []
for index in range(EPOCHS):
    middleCosNN.zero_grad()
    output = middleCosNN(simulatedInput)
    cost = costFunc(output, groundTruthCos)
    middleCosCostList.append(cost)
    cost.backward()
    middleCosOpt.step()    
    
deepCosCostList = []
for index in range(EPOCHS):
    deepCosNN.zero_grad()
    output = deepCosNN(simulatedInput)
    cost = costFunc(output, groundTruthCos)
    deepCosCostList.append(cost)
    cost.backward()
    deepCosOpt.step()

shallowSinCostList = []
for index in range(EPOCHS):
    shallowSinNN.zero_grad()
    output = shallowSinNN(simulatedInput)
    cost = costFunc(output, groundTruthSin)
    shallowSinCostList.append(cost)
    cost.backward()
    shallowSinOpt.step()

middleSinCostList = []
for index in range(EPOCHS):
    middleSinNN.zero_grad()
    output = middleSinNN(simulatedInput)
    cost = costFunc(output, groundTruthSin)
    middleSinCostList.append(cost)
    cost.backward()
    middleSinOpt.step()    
    
deepSinCostList = []
for index in range(EPOCHS):
    deepSinNN.zero_grad()
    output = deepSinNN(simulatedInput)
    cost = costFunc(output, groundTruthSin)
    deepSinCostList.append(cost)
    cost.backward()
    deepSinOpt.step()

plt.plot(counterList, deepSinCostList, 'r', label='model1(deep)')
plt.plot(counterList, middleSinCostList, 'g', label='model2(normal)')
plt.plot(counterList, shallowSinCostList, 'b', label='model3(shallow)')
plt.title("sin(1.5*x)")
plt.xlabel("ephocs")
plt.ylabel("loss")
plt.legend(loc="center")
plt.show()

plt.plot(counterList, deepCosCostList, 'r', label='model1(deep)')
plt.plot(counterList, middleCosCostList, 'g', label='model2(normal)')
plt.plot(counterList, shallowCosCostList, 'b', label='model3(shallow)')
plt.title("cos(0.5*x)")
plt.xlabel("ephocs")
plt.ylabel("loss")
plt.legend(loc="center")
plt.show()


simulatedInput = 10 * torch.rand((1000, 1)) - 5
x=simulatedInput
groundTruthSin = (np.sin(1.5*x))
groundTruthCos = (np.cos(0.5*x))


shallowCosOutput = shallowCosNN(simulatedInput)
middleCosOutput = middleCosNN(simulatedInput)
deepCosOutput = deepCosNN(simulatedInput)
shallowSinOutput = shallowSinNN(simulatedInput)
middleSinOutput = middleSinNN(simulatedInput)
deepSinOutput = deepSinNN(simulatedInput)


plt.plot(simulatedInput, deepCosOutput.tolist(), 'go', label='model1(deep)')
plt.plot(simulatedInput, middleCosOutput.tolist(), 'ro', label='model2(normal)')
plt.plot(simulatedInput, shallowCosOutput.tolist(), 'co', label='model3(shalloow)')
plt.plot(simulatedInput, groundTruthCos.tolist(), 'bo', label='Ground-Truth')

plt.title("Predictions curve and Truth Value for cos(0.5x)")
plt.xlabel("input(x)")
plt.ylabel("output(y=cos(0.5x)")
plt.legend(loc="top right")
plt.show()



plt.plot(simulatedInput, deepSinOutput.tolist(), 'go', label='model1(deep)')
plt.plot(simulatedInput, middleSinOutput.tolist(), 'ro', label='model2(normal)')
plt.plot(simulatedInput, shallowSinOutput.tolist(), 'co', label='model3(shallow)')
plt.plot(simulatedInput, groundTruthSin.tolist(), 'bo', label='Ground-Truth')

plt.title("Predictions curve and Truth Value for sin(1.5x)")
plt.xlabel("input(x)")
plt.ylabel("output(y=sin(1.5x)")
plt.legend(loc="upper left")
plt.show()

print(tf.__version__)


class ShallowTrainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = self.fc2(val)
        return val


class MiddleTrainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 25)
        self.fc2 = nn.Linear(25, 60)
        self.fc3 = nn.Linear(60, 38)
        self.fc4 = nn.Linear(38, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = self.fc4(val)
        return val
    
class DeepTrainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 60)
        self.fc4 = nn.Linear(60, 50)        
        self.fc5 = nn.Linear(50, 30)
        self.fc6 = nn.Linear(30, 10)        
        
    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = F.relu(self.fc3(val))
        val = F.relu(self.fc4(val))
        val = F.relu(self.fc5(val))
        val = self.fc6(val)
        return val

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

def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val
print(calcParams(MiddleTrainNN()))
print(calcParams(DeepTrainNN()))
print(calcParams(ShallowTrainNN()))
print(calcParams(GradientNN()))
