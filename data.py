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
    new_img = np.array(img.clone().detach().view(28, 28))


    while torch.sum(img - new_img) / torch.sum(img) <= 0.3: # fraction of image cut off
        pt1 = torch.randint(0, 28, (2,))
        pt2 = torch.randint(0, 28, (2,))
        pt1 = (pt1[0].item(), pt1[1].item())
        pt2 = (pt2[0].item(), pt2[1].item())
        cv2.rectangle(new_img, pt1, pt2, color=0, thickness=-1)


    return torch.Tensor(new_img).view(1, 28, 28)


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
