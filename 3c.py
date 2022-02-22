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

class Model1 (nn.Module):
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


class Model2 (nn.Module):
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


class Model3 (nn.Module):
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
    

class Model4 (nn.Module):
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
    

class Model5 (nn.Module):
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

model1 = Model1()
model2 = Model2()
model3 = Model3()
model4 = Model4()
model5 = Model5()
costFunc = nn.CrossEntropyLoss()
model1Opt = optim.Adam(model1.parameters(), lr=0.001)
model2Opt = optim.Adam(model2.parameters(), lr=0.001)
model3Opt = optim.Adam(model3.parameters(), lr=0.001)
model4Opt = optim.Adam(model4.parameters(), lr=0.001)
model5Opt = optim.Adam(model5.parameters(), lr=0.001)


trainingSet = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testingSet = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

batchSizes = [5, 50, 500, 1000, 5000]
train1 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[0], shuffle=True)
test1 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[0], shuffle=True)
train2 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[1], shuffle=True)
test2 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[1], shuffle=True)
train3 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[2], shuffle=True)
test3 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[2], shuffle=True)
train4 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[3], shuffle=True)
test4 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[3], shuffle=True)
train5 = torch.utils.data.DataLoader(trainingSet, batch_size=batchSizes[4], shuffle=True)
test5 = torch.utils.data.DataLoader(testingSet, batch_size=batchSizes[4], shuffle=True)




EPOCHS = 50
for index in range(EPOCHS):
    
    print(index)
    
    
    for batch in train1:
        inputImages, groundTruth = batch
        model1.zero_grad()
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model1Opt.step()

    
    for batch in train2:
        inputImages, groundTruth = batch
        model2.zero_grad()
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model2Opt.step()
        
   
    for batch in train3:
        inputImages, groundTruth = batch
        model3.zero_grad()
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model3Opt.step()
        
   
    for batch in train4:
        inputImages, groundTruth = batch
        model4.zero_grad()
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model4Opt.step()
        
   
    for batch in train5:
        inputImages, groundTruth = batch
        model5.zero_grad()
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model5Opt.step()

trainCostList = []
trainAccList = []


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train1:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train2:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train3:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train4:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3)) 


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train5:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
trainCostList.append(costTotal / costCounter)
trainAccList.append(round(correct/total, 3))


testCostList = []
testAccList = []


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test1:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test2:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test3:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test4:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test5:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
testCostList.append(costTotal / costCounter)
testAccList.append(round(correct/total, 3))


sensitivityList = []


gradAll = 0.0
fNormAll = 0
counter = 0
for p in model1.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)



gradAll = 0.0
fNormAll = 0
counter = 0
for p in model2.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)


gradAll = 0.0
fNormAll = 0
counter = 0
for p in model3.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)


gradAll = 0.0
fNormAll = 0
counter = 0
for p in model4.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)


gradAll = 0.0
fNormAll = 0
counter = 0
for p in model5.parameters():
    grad = 0.0
    if p.grad is not None:
        grad = p.grad
        fNorm = torch.linalg.norm(grad).numpy()
        fNormAll += fNorm
        counter += 1
sensitivityList.append(fNormAll / counter)


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batchSizes, trainAccList, 'b', label='train_loss')
ax1.plot(batchSizes, testAccList, 'b--', label='test_loss')
ax2.plot(batchSizes, sensitivityList, 'r', label='Sensitivity')
ax1.set_title('Batch Size vs Accuracy and Sensitivity')
ax1.set_xlabel('batch size (log scale)')
ax1.set_xscale('log')
ax1.set_ylabel('Accuracy', color='b')
ax2.set_ylabel('Sensitivity', color='r')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(batchSizes, trainCostList, 'b', label='train_acc')
ax1.plot(batchSizes, testCostList, 'b--', label='test_acc')
ax2.plot(batchSizes, sensitivityList, 'r', label='Sensitivity')
ax1.set_title('Batch Size vs Loss and Sensitivity')
ax1.set_xlabel('batch size (log scale)')
ax1.set_xscale('log')
ax1.set_ylabel('Loss', color='b')
ax2.set_ylabel('Sensitivity', color='r')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

