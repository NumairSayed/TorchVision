import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

print(datasets)

# Define the transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with a probability of 0.5
    transforms.RandomRotation(10),           # Randomly rotate the image by up to 10 degrees
    transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),  # Randomly crop and resize the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
    transforms.ToTensor(),                   # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))     # Normalize the image
])


# Load the dataset
dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

batch_size = 32
k_folds = 3

# Define the k-fold cross-validator
kfold = KFold(n_splits=k_folds, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(type(model))
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 300 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# K-Fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = Subset(dataset, train_ids)
    test_subsampler = Subset(dataset, test_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subsampler, batch_size=batch_size, shuffle=False)

    # Initialize the neural network
    model = NeuralNetwork().to(device)

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-3)

    # Run the training loop for defined number of epochs
    epochs = 10
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}\n-------------------------------')
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)

    # Save the model state for this fold
    torch.save(model.state_dict(), f"fashionMNISTweights_fold{fold+1}.pth")
    print(f"Saved PyTorch Model State for fold {fold+1} as 'fashionMNISTweights_fold{fold+1}.pth'")
    print('--------------------------------')

print('Training complete.')

"""
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("fashionMNISTweights.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
image = Image.open("/home/numair/Downloads/image17210800560.jpg")

transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to match model input size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=mean,  # Normalize 
                         std=std)
])

image = transform(image)

transformed_image = image.permute(1, 2, 0).numpy()  # Change the order of dimensions for visualization

# Display the transformed image
plt.figure()
plt.title("Transformed Image")
plt.imshow(transformed_image, cmap='gray')
plt.axis('off')
plt.show()

label = 'ankle_boot'

with torch.no_grad():
    image = image.to(device)
    pred = model(image)
    print(pred)
    predicted, actual = classes[pred.argmax()], label
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
"""