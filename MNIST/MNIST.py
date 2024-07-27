import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt


# Definition of  the LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forwardTester(self, x):
        x = nn.functional.relu(self.conv1(x))
        self.visualize(x,'conV1 + ReLU')
        x = nn.functional.max_pool2d(x, 2)
        self.visualize(x,'MAXPOOL1 + ReLU')
        
        x = nn.functional.relu(self.conv2(x))
        self.visualize(x,'conV2 + ReLU')
        
        x = nn.functional.max_pool2d(x, 2)
        self.visualize(x,'MAXPOOL2 + ReLU')
        
        x = x.view(-1, 16*4*4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def visualize(self, x, layer_name):
        print(f'{layer_name} output shape: {x.shape}')
        fig, axes = plt.subplots(1, min(x.shape[1], 10), figsize=(15, 15))
        for i, ax in enumerate(axes):
            if i >= x.shape[1]:
                break
            ax.imshow(x[0, i].detach().cpu().numpy(), cmap='gray')
            ax.axis('off')
        plt.show()


transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=64, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)



num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")



model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), f"LeNet_MNISTweights.pth")



### Visualizer

iterator = iter(testloader)  # Create an iterator
first_batch = next(iterator) # Retrieve the first batch

# Extract the first image and its label from the batch
first_image, first_label = first_batch[0][0], first_batch[1][0]
print(type(first_image))
first_image = first_image.unsqueeze(0).to(device)
model.forwardTester(first_image)

