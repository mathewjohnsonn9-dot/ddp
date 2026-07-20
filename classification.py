import torch
import torchvision.models as models
import torch.nn as nn
import tqdm
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.datasets
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import  tqdm
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        # Remove the final classifier (fully connected) layer
        self.resnet = torch.nn.Sequential(*list(model.children())[:-1])
        # Add a custom classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(2048, 512),  # Adjust input and output dimensions as needed
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Output layer with the number of classes
        )
        # self.model = nn.Sequential(model, classifier)

    def forward(self, x):
        with torch.no_grad():
           x = self.resnet(x)
        x = self.classifier(x)
        return x



def train(model, optimizer, train_data, criterion, device, epochs):
    for epoch in range(epochs):
        total_loss = 0
        if (epoch+1) % 10 == 0:
            accuracy(model, train_data, device)
            name = 'model_ck/model_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), name)
        for index_, (data, target) in enumerate(tqdm(train_data)):
            data = data.to(device=device)
            target = target.to(device=device)
            prediction = model(data)
            optimizer.zero_grad()
            loss = criterion(prediction, target)
            total_loss += loss
            loss.backward()
            optimizer.step()

        print(total_loss)

    return

def train_fn():
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 64
    learning_rate = 0.0001
    epochs = 1000
    root = 'classifier'
    train_dataset = ImageFolder(root, transform=transform)
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    # for name, param in model.named_parameters():
    #     print(name,  param.requires_grad)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    train(model, optimizer, train_data, criterion, device, epochs)


def test (path):
    # Load and preprocess your image
    image = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((720, 1280)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    checkpoint = torch.load('model_ck/model_50.pth')


    # Load the checkpoint's state_dict into your model

    model = CNN()
    model.load_state_dict(checkpoint)

    model = model.to(device)
    # Put the model in evaluation mode
    model.eval()

    # Make a prediction
    with torch.no_grad():
        output = model(image)

    # Get the predicted class
    _, predicted = output.max(1)
    print(predicted)

    # You may need to map the index back to a class label using your mapping.


def accuracy(model, data, device):
    correct_sample = 0
    sample_number = 0
    model.eval()
    for _, (data_, target) in enumerate(data):
        data_ = data_.to(device=device)
        target = target.to(device=device)
        prediction_ = model(data_)
        _, m = prediction_.max(1)
        correct_sample += (m == target).sum()
        sample_number += prediction_.shape[0]
    print(f'got {correct_sample}/{sample_number} with accuracy {float(correct_sample/sample_number) *100:.2f}')

if __name__=='__main__':
    print('train started !')
    train_fn()
    # path = '/home/fariborz_taherkhani/Downloads/place_side_1010/left_confirm_images/left_152857_20230719T131141.531481.jpg'
    # test(path)
