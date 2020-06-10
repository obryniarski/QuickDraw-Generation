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
    # val = torch.randint(1, 5, (1,))
    # binary_randoms = torch.randint(0, 2, (4,))

    # min_x = W + 1
    # max_x = 0
    # min_y = H + 1
    # max_y = 0
    # for x in range(W):
    #     for y in range(H):
    #         if img[0,x,y] > 0.5:
    #             min_x = min(min_x, x)
    #             max_x = max(max_x, x)
    #             min_y = min(min_y, y)
    #             max_y = max(max_y, y)

    # avg_x = (min_x + max_x) // 2
    # avg_y = (min_y + max_y) // 2
    #
    # x_split = torch.randint(avg_x - 1, avg_x + 1, (1,))
    # y_split = torch.randint(avg_y - 1, avg_y + 1, (1,))
    # if val == 1:
    #     for x in range(x_split, W):
    #         for y in range(H):
    #             new_img[0,x,y] = 0.
    # elif val == 2:
    #     for x in range(torch.randint(10, 18, (1,))):
    #         for y in range(H):
    #             new_img[0,x,y] = 0.
    # elif val == 3:
    #     for x in range(W):
    #         for y in range(torch.randint(10, 18, (1,)), H):
    #             new_img[0,x,y] = 0.
    # elif val == 4:
    #     for x in range(W):
    #         for y in range(torch.randint(10, 18, (1,))):
    #             new_img[0,x,y] = 0.

    # first, does x split
    # then does y split
    # then if split
        # split left or right on x

    # if binary_randoms[0].item(): # x split ?
    #     if binary_randoms[2].item(): # direction of x split
    #         for x in range(0, x_split):
    #             if binary_randoms[1].item(): # split on y ?
    #                 if binary_randoms[3].item():
    #                     for y in range(0, y_split):
    #                         new_img[0,x,y] = 0.
    #                 else:
    #                     for y in range(y_split, H):
    #                         new_img[0,x,y] = 0.
    #             else:
    #                 for y in range(H):
    #                     new_img[0,x,y] = 0.
    #
    #     else:
    #         for x in range(x_split, W):
    #             if binary_randoms[1].item(): # split on y ?
    #                 if binary_randoms[3].item():
    #                     for y in range(0, y_split):
    #                         new_img[0,x,y] = 0.
    #                 else:
    #                     for y in range(y_split, H):
    #                         new_img[0,x,y] = 0.
    #             else:
    #                 for y in range(H):
    #                     new_img[0,x,y] = 0.
    #
    # else:
    #     for x in range(W):
    #         if binary_randoms[1].item(): # split on y ?
    #             if binary_randoms[3].item():
    #                 for y in range(0, y_split):
    #                     new_img[0,x,y] = 0.
    #             else:
    #                 for y in range(y_split, H):
    #                     new_img[0,x,y] = 0.
    #         else:
    #             for y in range(H):
    #                 return fill(img)

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

# data = MNIST_traindata('data/')
# dataloader = DataLoader(data, batch_size=16, shuffle=True)
# test = next(iter(dataloader))
# print(test[0].shape, test[1].shape, test[2].shape)
# # print(data[0][0].shape)
# img = 1
# full_img, cut_img = test[0][img], test[2][img]
# fig, axes = plt.subplots(1, 2)
# axes[0].imshow(full_img.view(28,28), cmap='Greys')
# axes[1].imshow(cut_img.view(28,28), cmap='Greys')
# plt.show()
# print(generate_MNIST_class(5))
