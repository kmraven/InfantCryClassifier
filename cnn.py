import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from consts import *

seed = 123
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

CLASS_NUM = 5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 50
NUM_WORKERS = 2

train_dir = os.path.join(DATASETS_PATH, 'train')
val_dir = os.path.join(DATASETS_PATH, 'val')


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 13, 512)
        self.fc2 = nn.Linear(512, CLASS_NUM)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 13)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((128, 111)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 必要？適切？dBの正規化は特殊かも？
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    gpu_id = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda")

    model = SimpleCNN()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loss_value=[]
    train_acc_value=[]
    test_loss_value=[]
    test_acc_value=[]

    for epoch in range(EPOCH):
        print('epoch', epoch + 1)
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()

        print("train mean loss={}, accuracy={}".format(
            sum_loss * BATCH_SIZE / len(train_loader.dataset),
            float(sum_correct / sum_total)
        ))
        train_loss_value.append(sum_loss * BATCH_SIZE / len(train_loader.dataset))
        train_acc_value.append(float(sum_correct / sum_total))

        sum_loss = 0.0
        sum_correct = 0
        sum_total = 0

        for (inputs, labels) in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()

        print("test  mean loss={}, accuracy={}".format(
            sum_loss * BATCH_SIZE / len(val_loader.dataset),
            float(sum_correct / sum_total)
        ))
        test_loss_value.append(sum_loss*BATCH_SIZE/len(val_loader.dataset))
        test_acc_value.append(float(sum_correct/sum_total))

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))


    # plot result
    plt.figure(figsize=(6,6))

    plt.plot(range(EPOCH), train_loss_value)
    plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.ylim(0, 2.5)
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.legend(['train loss', 'test loss'])
    plt.title('loss')
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_image.png"))
    plt.clf()

    plt.plot(range(EPOCH), train_acc_value)
    plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
    plt.xlim(0, EPOCH)
    plt.ylim(0, 1)
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.legend(['train acc', 'test acc'])
    plt.title('accuracy')
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_image.png"))


if __name__ == "__main__":
    main()
