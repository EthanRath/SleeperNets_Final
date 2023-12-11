import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T

def load_fmnist_torch(root="./data", transform=None, download=True):
    
    if transform == None:
        transform = T.ToTensor()
    
    train_set = FashionMNIST(root=root,  transform=transform, download=download, train=True)
    test_set = FashionMNIST(root=root,  transform=transform, download=download, train=False)
    
    # Each item in this dictionary is a torch Dataset object
    # To feed the data into a model, you may have to use a DataLoader 
    return {"train": train_set, "test": test_set}

class DataSetCustom:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def DataloaderToTensor(loader):
    size = [len(loader.dataset)] + list(loader.dataset[0][0].size())
    x = torch.zeros(size = size)
    y = torch.zeros(len(loader.dataset))

    indexer = 0
    for dx, dy in loader:
        bs = len(dx)
        x[indexer: indexer + bs] = dx
        y[indexer: indexer + bs] = dy
        indexer += bs

    return x, y

def TensorToDataloader(x, y, batchsize = 64, shuffle = False):
    return DataLoader(DataSetCustom(x, y), batch_size = batchsize, shuffle=shuffle, )

def plot(x, y, labels, size, name = "plot.png"):
    total = size[0]*size[1]
    plt.figure(dpi = 150)
    for i in range(total):
        plt.subplot(size[0], size[1], i+1)
        plt.imshow(x[i][0], cmap = "gray")
        plt.title(labels[int(y[i])])
    plt.savefig("images/" + name)