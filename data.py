import torch
import torchvision
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np

def fill(img):
    _, W, H = img.shape
    assert W == H, 'ffs'
    new_img = img.copy().reshape(W, H)

    while np.sum(img - new_img) / np.sum(img) <= 0.5: # fraction of image cut off
        pt1 = torch.randint(0, W, (2,))
        pt2 = torch.randint(0, W, (2,))
        pt1 = (pt1[0].item(), pt1[1].item())
        pt2 = (pt2[0].item(), pt2[1].item())
        cv2.rectangle(new_img, pt1, pt2, color=0, thickness=-1)


    return torch.Tensor(new_img).view(1, W, W)


class MNIST_traindata(Dataset):

    def __init__(self, root, transform=transforms.ToTensor(), train=True):
        self.root = root
        self.MNIST = torchvision.datasets.MNIST(root, train=train, transform=transform, download=True)
        # self.all_labels = torch.Tensor([self.MNIST[i][1] for i in range(len(self.MNIST))])
        # self.data = torch.cat([self.MNIST[i][0] for i in range(len(self.MNIST))])
        # ones = torch.ones_like(self.all_labels)
        # self.class_indices = [(self.all_labels == (ones * c)) for c in range(10)]
        # self.class_data = [self.data[c] for c in self.class_indices]
        # self.class_lengths = [len(d) for d in self.class_data]

    def __len__(self):
        return len(self.MNIST)

    def __getitem__(self, idx):
        base, label = self.MNIST.__getitem__(idx)
        cutoff = fill(base)
        # same_class = self.class_data[label][torch.randint(0, self.class_lengths[label], (1,))]
        # return [base, cutoff, same_class, label]
        return [base, cutoff, label]

def draw_from_strokes(strokes):
    base_img = np.zeros((256, 256))
    for stroke in strokes:
        for i in range(len(stroke[0]) - 1):
            pt1 = (stroke[0][i], stroke[1][i])
            pt2 = (stroke[0][i + 1], stroke[1][i + 1])
            cv2.line(base_img, pt1, pt2, color=1, thickness=4, lineType=cv2.LINE_AA)
    return base_img.reshape(1, 256, 256)

def quickdraw_loader(filename):
    strokes = torch.load(filename)
    drawing = draw_from_strokes(strokes)
    return drawing

class Quickdraw_traindata(Dataset):

    def __init__(self, root, transform=None, loader=quickdraw_loader):
        self.root = root
        self.quickdraw_data = torchvision.datasets.DatasetFolder(self.root, loader, transform=transform, extensions='.pt')


    def __len__(self):
        return len(self.quickdraw_data)


    def __getitem__(self, idx):
        base, label = self.quickdraw_data[idx]
        cutoff = fill(base)

        classes_onehot = torch.zeros(20)
        classes_onehot[label] = 1

        return [base, cutoff, classes_onehot]

# class Quickdraw_traindata(Dataset):
#
#     def __init__(self, root, transform=transforms.ToTensor(), loader=quickdraw_loader):
#         self.root = root
#         self.quickdraw_data = torchvision.datasets.DatasetFolder(self.root, loader, transform=transform, extensions='.pt')
#
#
#     def __len__(self):
#         return len(self.quickdraw_data)
#
#
#     def __getitem__(self, idx):
#         base, label = self.quickdraw_data[idx]
#         cutoff = fill(base)
#         return [base, cutoff, label]
