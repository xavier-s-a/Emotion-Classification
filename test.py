from time import time
import multiprocessing as mp
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
])

import torch
import platform

# Detect the operating system
os_name = platform.system()

if os_name == "Windows" or os_name == "Linux":
    # For Windows and Linux
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
elif os_name == "Darwin":  # macOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    # Default to CPU for other operating systems
    device = torch.device("cpu")

print(f"Using device: {device}")
train_dataset = datasets.ImageFolder('/Users/xavier/Desktop/CS712/project/fer2013/train', transform=transform)
if __name__ == '__main__':
    for num_workers in range(2, mp.cpu_count(), 2):  
        train_loader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finished in: {} seconds, num_workers={}".format(end - start, num_workers))