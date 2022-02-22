import tensorflow as tf
import numpy as np
import torch
import torchvision as tv
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
print(tf.__version__)

#models designed
class ShallowTrainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 41)
        self.fc2 = nn.Linear(41, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = self.fc2(val)
        return val


class MiddleTrainNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 35)
        self.fc2 = nn.Linear(35, 60)
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
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 40)
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
        

trainingSet = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testingSet = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train = torch.utils.data.DataLoader(trainingSet, batch_size=50, shuffle=True)
test = torch.utils.data.DataLoader(testingSet, batch_size=50, shuffle=True)

def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val
print(calcParams(MiddleTrainNN()))
print(calcParams(DeepTrainNN()))
print(calcParams(ShallowTrainNN()))


shallownn = ShallowTrainNN()
middlenn = MiddleTrainNN()
deepnn = DeepTrainNN()
costFunc = nn.CrossEntropyLoss()
shallowOpt = optim.Adam(shallownn.parameters(), lr=0.001)
middleOpt = optim.Adam(middlenn.parameters(), lr=0.001)
deepOpt = optim.Adam(deepnn.parameters(), lr=0.001)


EPOCHS = 100
counter = 0
counterList = []
shallowCostList = []
shallowTestAccuracyList = []
shallowTrainAccuracyList = []
for index in range(EPOCHS):
    counterList.append(counter)
    counter += 1
    for batch in train:
        inputImages, groundTruth = batch
        shallownn.zero_grad()
        output = shallownn(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        shallowOpt.step()
    shallowCostList.append(cost)
   
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in train:
            inputImages, groundTruth = batch
            output = shallownn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    shallowTrainAccuracyList.append(round(correct/total, 3))


    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test:
            inputImages, groundTruth = batch
            output = shallownn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    shallowTestAccuracyList.append(round(correct/total, 3))


middleCostList = []
middleTrainAccuracyList = []
middleTestAccuracyList = []
for index in range(EPOCHS):
    
    for batch in train:
        inputImages, groundTruth = batch
        middlenn.zero_grad()
        output = middlenn(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        middleOpt.step()
    middleCostList.append(cost)
    
   
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in train:
            inputImages, groundTruth = batch
            output = middlenn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    middleTrainAccuracyList.append(round(correct/total, 3))

   
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test:
            inputImages, groundTruth = batch
            output = middlenn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    middleTestAccuracyList.append(round(correct/total, 3))


deepCostList = []
deepTrainAccuracyList = []
deepTestAccuracyList = []
for index in range(EPOCHS):
    # Train model and keep track of loss
    for batch in train:
        inputImages, groundTruth = batch
        deepnn.zero_grad()
        output = deepnn(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        deepOpt.step()
    deepCostList.append(cost)
    
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in train:
            inputImages, groundTruth = batch
            output = deepnn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    deepTrainAccuracyList.append(round(correct/total, 3))

    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test:
            inputImages, groundTruth = batch
            output = deepnn(inputImages.view(-1,784))
            for i, outputTensor in enumerate(output):
                if torch.argmax(outputTensor) == groundTruth[i]:
                    correct += 1
                total += 1
    deepTestAccuracyList.append(round(correct/total, 3))

plt.plot(counterList, deepCostList, 'r', label='model1(deep)')
plt.plot(counterList, middleCostList, 'g', label='model2(normal)')
plt.plot(counterList, shallowCostList, 'b', label='model3(shallow)')
plt.title("Trainig Process with Loss on MNIST data")
plt.xlabel("ephocs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()


plt.plot(counterList, deepTrainAccuracyList, 'b--', label='Train-model1(deep)')
plt.plot(counterList, deepTestAccuracyList, 'b', label='Test-model1(deep)')
plt.plot(counterList, middleTrainAccuracyList, 'g--', label='Train-model2(normal)')
plt.plot(counterList, middleTestAccuracyList, 'g', label='Test-model2(normal)')
plt.plot(counterList, shallowTrainAccuracyList, 'r--', label='Train-model3(shallow)')
plt.plot(counterList, shallowTestAccuracyList, 'r', label='Test-model3(shallow)')


plt.title("Training process with Accuracy on MNIST data")
plt.xlabel("ephocs")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()
