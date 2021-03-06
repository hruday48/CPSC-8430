
pip install tensorflow==2.4

# Commented out IPython magic to ensure Python compatibility.
# A DNN is trained on the function y = cos(x) and the process of optimizing weights is shown. 

import tensorflow as tf
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt





def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val


class OptimizeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val


optimNet = OptimizeNN()
costFunc = nn.MSELoss()
opt = optim.Adam(optimNet.parameters(), lr=0.001)

calcParams(optimNet)

print(tf.__version__)

simulatedInput = 20 * torch.rand((1000, 1)) - 10
groundTruth = np.sin(1.5*simulatedInput)

trainingIter = 8
EPOCHS = 30
epochNum = round(EPOCHS / 3)
layer1Connections = 1 * 3
layer2Connections = 3 * 3
layer3Connections = 3 * 1
totModelConnections = layer1Connections + layer2Connections + layer3Connections


costList = []
counter = 0
layer2WeightsTensor = torch.zeros((trainingIter * epochNum, layer2Connections))
layer1WeightsTensor = torch.zeros((trainingIter * epochNum, layer1Connections))
wholeModelWeightsTensor = torch.zeros((trainingIter * epochNum, totModelConnections))
costTensor = torch.zeros((trainingIter * epochNum, 1))
for trainingIndex in range(trainingIter): 
    
    
    optimNet = OptimizeNN()
    costFunc = nn.MSELoss()
    opt = optim.Adam(optimNet.parameters(), lr=0.001)
    for epochIndex in range(EPOCHS):
        optimNet.zero_grad()
        output = optimNet(simulatedInput)
        cost = costFunc(output, groundTruth)
        cost.backward()
        opt.step()

        
        if epochIndex % 3 == 0:
            
            layer1Weights = torch.reshape(torch.flatten(optimNet.fc1.weight), (1, layer1Connections))
            layer2Weights = torch.reshape(torch.flatten(optimNet.fc2.weight), (1, layer2Connections))
            layer3Weights = torch.reshape(torch.flatten(optimNet.fc3.weight), (1, layer3Connections))
           
            temp = torch.cat((layer2Weights, layer1Weights), dim=1)
            wholeModelWeights = torch.cat((temp, layer3Weights), dim=1)
            
            wholeModelWeightsTensor[counter] = wholeModelWeights
            layer2WeightsTensor[counter] = layer2Weights
            layer1WeightsTensor[counter] = layer1Weights
            costTensor[counter] = cost
            counter += 1


def pcaImplementation(inputArray, dimToReturn):
    m, n = inputArray.shape
    

    meansOfInputs = np.array([np.mean(inputArray[:, index]) for index in range(n)])
    inputNormalized = inputArray - meansOfInputs
    

    mtr = np.dot(np.transpose(inputNormalized),inputNormalized)
    eigenvalues, eigenvectors = np.linalg.eig(mtr)
    pairs = [(np.abs(eigenvalues[index]), eigenvectors[:, index]) for index in range(n)]


    pairs.sort(key=lambda x: x[0], reverse=True)
    featFromData = np.array([value[1] for value in pairs[:dimToReturn]])
    reducedDimData = np.dot(inputNormalized, np.transpose(featFromData))
    return reducedDimData


layer2Reduced = layer2WeightsTensor.detach().cpu().numpy()
wholeModelReduced = wholeModelWeightsTensor.detach().cpu().numpy()
costVector = costTensor.detach().cpu().numpy()
layer1Reduced = layer1WeightsTensor.detach().cpu().numpy()
wholeMode1Reduced = wholeModelWeightsTensor.detach().cpu().numpy()
costVector = costTensor.detach().cpu().numpy()
layer2Reduced = pcaImplementation(layer2Reduced, 2)
wholeModelReduced = pcaImplementation(wholeModelReduced, 2)


colorList = ['r*', 'b*', 'g*', 'm*', 'c*', 'y*', 'k*']
colorNameList = ["red", "blue", "green", "purple", "cyan", "yellow", 'black']
counter = 0
colorCounter = 0
for pair in layer2Reduced:
    if (counter % 10) == 0 and counter != 0:
       
        if colorCounter >= len(colorList) - 1:
            colorCounter = 0
        else:
            colorCounter += 1
    plt.plot(pair[0], pair[1], colorList[colorCounter])
    plt.annotate(str(round(costVector[counter][0], 2)), (pair[0], pair[1]), color=colorNameList[colorCounter])
    counter += 1
plt.title("Optimization Process Layer 1") 
plt.show()


colorList = ['r*', 'b*', 'g*', 'm*', 'c*', 'y*', 'k*']
colorNameList = ["red", "blue", "green", "purple", "cyan", "yellow", 'black']
counter = 0
colorCounter = 0
for pair in layer1Reduced:
    if (counter % 10) == 0 and counter != 0:
       
        if colorCounter >= len(colorList) - 1:
            colorCounter = 0
        else:
            colorCounter += 1
    plt.plot(pair[0], pair[1], colorList[colorCounter])
    plt.annotate(str(round(costVector[counter][0], 2)), (pair[0], pair[1]), color=colorNameList[colorCounter])
    counter += 1
plt.title("Optimization Process Layer 2") 
plt.show()


counter = 0
colorCounter = 0
for pair in wholeModelReduced:
    if (counter % 10) == 0 and counter != 0:
        if colorCounter >= len(colorList) - 1:
            colorCounter = 0
        else:
            colorCounter += 1
    plt.plot(pair[0], pair[1], colorList[colorCounter])
    plt.annotate(str(round(costVector[counter][0], 2)), (pair[0], pair[1]), color=colorNameList[colorCounter])
    counter += 1
plt.title("Optimization Process Whole Model")
plt.show()

print(counter)
print(layer2Weights)
print(layer1Weights)
print(wholeModelWeights)
