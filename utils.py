import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt


def base_loss(predicted, sketch):
    return torch.sum(nn.functional.relu(sketch - 100 * predicted)) / len(predicted)

def real_accuracy(predicted):
    return torch.sum(torch.round(predicted), dtype=torch.double) / len(predicted)

def fake_accuracy(predicted):
    return 1 - real_accuracy(predicted)

# https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/3
def test_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()
