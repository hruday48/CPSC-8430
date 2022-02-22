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





def calcParams(inputModel):
    val = sum(params.numel() for params in inputModel.parameters() if params.requires_grad)
    return val


class Model1 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 2)
        self.fc2 = nn.Linear(2, 4)
        self.fc3 = nn.Linear(4, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    

class Model2 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    

class Model3 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, 10)

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
        self.fc1 = nn.Linear(784, 32)
        self.fc2 = nn.Linear(32, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    

class Model6 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val

class Model7 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    

class Model8 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val


class Model9 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
    
class Model10 (nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, val):
        val = F.relu(self.fc1(val))
        val = F.relu(self.fc2(val))
        val = self.fc3(val)
        return val
trainingSet = datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
testingSet = datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train = torch.utils.data.DataLoader(trainingSet, batch_size=50, shuffle=True)
test = torch.utils.data.DataLoader(testingSet, batch_size=50, shuffle=True)

model1 = Model1()
model2 = Model2()
model3 = Model3()
model4 = Model4()
model5 = Model5()
model6 = Model6()
model7 = Model7()
model8 = Model8()
model9 = Model9()
model10 = Model10()
costFunc = nn.CrossEntropyLoss()
model1Opt = optim.Adam(model1.parameters(), lr=0.001)
model2Opt = optim.Adam(model2.parameters(), lr=0.001)
model3Opt = optim.Adam(model3.parameters(), lr=0.001)
model4Opt = optim.Adam(model4.parameters(), lr=0.001)
model5Opt = optim.Adam(model5.parameters(), lr=0.001)
model6Opt = optim.Adam(model6.parameters(), lr=0.001)
model7Opt = optim.Adam(model7.parameters(), lr=0.001)
model8Opt = optim.Adam(model8.parameters(), lr=0.001)
model9Opt = optim.Adam(model9.parameters(), lr=0.001)
model10Opt = optim.Adam(model10.parameters(), lr=0.001)
model1Params = calcParams(model1)
model2Params = calcParams(model2)
model3Params = calcParams(model3)
model4Params = calcParams(model4)
model5Params = calcParams(model5)
model6Params = calcParams(model6)
model7Params = calcParams(model7)
model8Params = calcParams(model8)
model9Params = calcParams(model9)
model10Params = calcParams(model10)
print(model1Params)
print(model10Params)


EPOCHS = 50
counter = 0
counterList = []
for index in range(EPOCHS):
    counterList.append(counter)
    counter += 1
   
    print(counter)
    
   
    for batch in train:
        inputImages, groundTruth = batch
        model1.zero_grad()
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model1Opt.step()

   
    for batch in train:
        inputImages, groundTruth = batch
        model2.zero_grad()
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model2Opt.step()
        
   
    for batch in train:
        inputImages, groundTruth = batch
        model3.zero_grad()
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model3Opt.step()
        
    
    for batch in train:
        inputImages, groundTruth = batch
        model4.zero_grad()
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model4Opt.step()
        
   
    for batch in train:
        inputImages, groundTruth = batch
        model5.zero_grad()
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model5Opt.step()
        
   
    for batch in train:
        inputImages, groundTruth = batch
        model6.zero_grad()
        output = model6(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model6Opt.step()
        
    
    for batch in train:
        inputImages, groundTruth = batch
        model7.zero_grad()
        output = model7(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model7Opt.step()
        
    
    for batch in train:
        inputImages, groundTruth = batch
        model8.zero_grad()
        output = model8(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model8Opt.step()
        
   
    for batch in train:
        inputImages, groundTruth = batch
        model9.zero_grad()
        output = model9(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model9Opt.step()
        
    
    for batch in train:
        inputImages, groundTruth = batch
        model10.zero_grad()
        output = model10(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        cost.backward()
        model10Opt.step()


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model1TrainCost = costTotal / costCounter
model1TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model2TrainCost = costTotal / costCounter
model2TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model3TrainCost = costTotal / costCounter
model3TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model4TrainCost = costTotal / costCounter
model4TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model5TrainCost = costTotal / costCounter
model5TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model6(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model6TrainCost = costTotal / costCounter
model6TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model7(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model7TrainCost = costTotal / costCounter
model7TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model8(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model8TrainCost = costTotal / costCounter
model8TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model9(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model9TrainCost = costTotal / costCounter
model9TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in train:
        inputImages, groundTruth = batch
        output = model10(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model10TrainCost = costTotal / costCounter
model10TrainAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model1(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model1TestCost = costTotal / costCounter
model1TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model2(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model2TestCost = costTotal / costCounter
model2TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model3(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model3TestCost = costTotal / costCounter
model3TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model4(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model4TestCost = costTotal / costCounter
model4TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model5(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model5TestCost = costTotal / costCounter
model5TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model6(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model6TestCost = costTotal / costCounter
model6TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model7(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model7TestCost = costTotal / costCounter
model7TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model8(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model8TestCost = costTotal / costCounter
model8TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model9(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model9TestCost = costTotal / costCounter
model9TestAcc = round(correct/total, 3)


correct = 0
total = 0
costTotal = 0
costCounter = 0
with torch.no_grad():
    for batch in test:
        inputImages, groundTruth = batch
        output = model10(inputImages.view(-1,784))
        cost = costFunc(output, groundTruth)
        costTotal += cost
        costCounter += 1
        for i, outputTensor in enumerate(output):
            if torch.argmax(outputTensor) == groundTruth[i]:
                correct += 1
            total += 1
model10TestCost = costTotal / costCounter
model10TestAcc = round(correct/total, 3)


plt.plot(model1Params, model1TrainCost, 'go',label='train_loss')
plt.plot(model2Params, model2TrainCost, 'go')
plt.plot(model3Params, model3TrainCost, 'go')
plt.plot(model4Params, model4TrainCost, 'go')
plt.plot(model5Params, model5TrainCost, 'go')
plt.plot(model6Params, model6TrainCost, 'go')
plt.plot(model7Params, model7TrainCost, 'go')
plt.plot(model8Params, model8TrainCost, 'go')
plt.plot(model9Params, model9TrainCost, 'go')
plt.plot(model10Params, model10TrainCost, 'go')

plt.plot(model1Params, model1TestCost, 'ro',label='test_loss')
plt.plot(model2Params, model2TestCost, 'ro')
plt.plot(model3Params, model3TestCost, 'ro')
plt.plot(model4Params, model4TestCost, 'ro')
plt.plot(model5Params, model5TestCost, 'ro')
plt.plot(model6Params, model6TestCost, 'ro')
plt.plot(model7Params, model7TestCost, 'ro')
plt.plot(model8Params, model8TestCost, 'ro')
plt.plot(model9Params, model9TestCost, 'ro')
plt.plot(model10Params, model10TestCost, 'ro')
plt.title("model loss")
plt.xlabel("number of parameters")
plt.ylabel("loss")
plt.legend(loc="upper right")

plt.show()


plt.plot(model1Params, model1TrainAcc, 'go', label='train_acc')
plt.plot(model2Params, model2TrainAcc, 'go')
plt.plot(model3Params, model3TrainAcc, 'go')
plt.plot(model4Params, model4TrainAcc, 'go')
plt.plot(model5Params, model5TrainAcc, 'go')
plt.plot(model6Params, model6TrainAcc, 'go')
plt.plot(model7Params, model7TrainAcc, 'go')
plt.plot(model8Params, model8TrainAcc, 'go')
plt.plot(model9Params, model9TrainAcc, 'go')
plt.plot(model10Params, model10TrainAcc, 'go')

plt.plot(model1Params, model1TestAcc, 'ro', label='test_acc')
plt.plot(model2Params, model2TestAcc, 'ro')
plt.plot(model3Params, model3TestAcc, 'ro')
plt.plot(model4Params, model4TestAcc, 'ro')
plt.plot(model5Params, model5TestAcc, 'ro')
plt.plot(model6Params, model6TestAcc, 'ro')
plt.plot(model7Params, model7TestAcc, 'ro')
plt.plot(model8Params, model8TestAcc, 'ro')
plt.plot(model9Params, model9TestAcc, 'ro')
plt.plot(model10Params, model10TestAcc, 'ro')
plt.title("model accuracy")
plt.xlabel("number of parameters")
plt.ylabel("loss")
plt.legend(loc="lower right")

plt.show()



