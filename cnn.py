import os
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from consts import *

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

CLASS_NUM = 5
BATCH_SIZE = 256
EPOCH = 20
NUM_WORKERS = 2
in_height = 111
in_width = 128
kernel = 3


class Net(nn.Module):
    def __init__(self, num_layer, mid_units, num_filters):
        super(Net, self).__init__()
        self.activation = nn.ReLU()
        #第1層
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters[0], kernel_size=3)])
        self.out_height = in_height - kernel + 1
        self.out_width = in_width - kernel + 1
        #第2層以降
        for i in range(1, num_layer):
            self.convs.append(nn.Conv2d(in_channels=num_filters[i-1], out_channels=num_filters[i], kernel_size=3))
            self.out_height = self.out_height - kernel + 1
            self.out_width = self.out_width - kernel + 1
        #pooling層
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_height = int(self.out_height / 2)
        self.out_width = int(self.out_width / 2)
        #線形層
        self.out_feature = self.out_height * self.out_width * num_filters[num_layer - 1]
        self.fc1 = nn.Linear(in_features=self.out_feature, out_features=mid_units) 
        self.fc2 = nn.Linear(in_features=mid_units, out_features=CLASS_NUM)

    def forward(self, x):
        for l in self.convs:
            x = l(x)
            x = self.activation(x)
        x = self.pool(x)
        x = x.view(-1, self.out_feature)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    shutil.rmtree(OUTPUT_DIR)
    trials_dir = 'trials'
    os.makedirs(os.path.join(OUTPUT_DIR, trials_dir), exist_ok=True)

    transform = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]
    val_transform = transforms.Compose(transform)
    transform.extend([
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    train_transform = transforms.Compose(transform)

    train_dir = os.path.join(DATASETS_PATH, 'train')
    val_dir = os.path.join(DATASETS_PATH, 'val')
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    gpu_id = sys.argv[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda")

    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        num_layer = trial.suggest_int('num_layer', 3, 7)
        mid_units = int(trial.suggest_discrete_uniform("mid_units", 100, 500, 100))
        num_filters = [int(trial.suggest_discrete_uniform("num_filter_"+str(i), 16, 128, 16)) for i in range(num_layer)]
        
        model = Net(num_layer, mid_units, num_filters).to(device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        def train():
            model.train()
            running_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss = running_loss / len(train_loader)
            return train_loss

        def valid():
            model.eval()
            running_loss = 0
            correct = 0
            total = 0
            val_labels = []
            val_probs = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    predicted = outputs.max(1, keepdim=True)[1]
                    labels = labels.view_as(predicted)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
                    val_labels.extend(labels.cpu().numpy())
                    val_probs.extend(outputs.cpu().numpy())
            val_loss = running_loss / len(val_loader)
            val_acc = correct / total
            val_probs = np.array(val_probs)
            val_auc = roc_auc_score(val_labels, val_probs, multi_class='ovr')
            return val_loss, val_acc, val_auc

        loss_list = []
        val_loss_list = []
        val_acc_list = []
        val_auc_list = []

        for _ in range(EPOCH):
            loss = train()
            val_loss, val_acc, val_auc = valid()
            loss_list.append(loss)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            val_auc_list.append(val_auc)
        
        pd.DataFrame({
            'epoch': list(range(1, EPOCH + 1)),
            'loss': loss_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
            'val_auc': val_auc_list,
        }).to_csv(os.path.join(OUTPUT_DIR, trials_dir, f"{trial.number}.csv"), index=False)

        return val_auc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(OUTPUT_DIR, "trials_dataframe.csv"), index=False)


if __name__ == "__main__":
    main()
