import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from EmotionNet import EmotionNet
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import platform
import time
from torchvision.transforms import functional as TF
import random


def get_device():
    os_name = platform.system()
    if os_name == "Windows" or os_name == "Linux": #for Win and Liunx
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    elif os_name == "Darwin": # for Macs
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    else:
        return torch.device("cpu")

class RandomSpecificRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)  
        return TF.rotate(img, angle)

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),
     transforms.RandomVerticalFlip(), 
    #transforms.RandomRotation(7),
    RandomSpecificRotation([90, 180, 270]) 
])
  
  #change the path if required for training with new data 
  # workers are calluated after cheking system
train_dataset = datasets.ImageFolder('/Users/xavier/Desktop/CS712/finalproject/fer2013/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=6)

    #change the path if required for testing with new data
test_dataset = datasets.ImageFolder('/Users/xavier/Desktop/CS712/finalproject/fer2013/validation', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=6)

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, num_epochs,device):
    model.train()

    # Lists to track performance metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_correct / train_total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_losses.append(val_loss / len(test_loader))
        val_accs.append(val_correct / val_total)

        # Print accuracy on the test set
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}')
        print(f'Accuracy of the network on the test images: {100 * val_correct / val_total:.2f}%')
    
    # Plotting training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return train_losses, train_accs, val_losses, val_accs
def main(train):
    device = get_device()
    model = EmotionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    if train:
        print(f"Using device {device}")
        start = time.time()
        train_and_validate(model, train_loader, test_loader, criterion, optimizer, 100,device)
        end = time.time()
        total = end-start
        print(f"Total time taken = {total : .2f} seconds")
        torch.save(model.state_dict(), '/Users/xavier/Desktop/CS712/finalproject/new_weights_emotions.pth')
    else:
        model.load_state_dict(torch.load('/Users/xavier/Desktop/CS712/finalproject/weights_emotions.pth'))
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        print(f"Device set to {device} for evaluation")
        print("Model loaded and set to evaluation mode.")

if __name__ == "__main__":    
    response = input("Do you want to train the model? (yes/no): ").strip().lower()
    train = response == 'yes'

    main(train)