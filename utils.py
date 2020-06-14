import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

device = (0 if torch.cuda.is_available() else 'cpu')

def base_loss(predicted, sketch):
    return torch.sum(nn.functional.relu(sketch - 2 * predicted)) / len(predicted)

def real_accuracy(predicted):
    return torch.sum(torch.round(predicted), dtype=torch.double) / len(predicted)

def fake_accuracy(predicted):
    return 1 - real_accuracy(predicted)

# https://discuss.pytorch.org/t/dropout-at-test-time-in-densenet/6738/3
def test_dropout(m):
    if type(m) == nn.Dropout2d:
        m.train()

def plot_fake_images(gen, full_imgs, cut_imgs, labels, num_images=5):
    assert num_images <= len(cut_imgs), 'not enough images to show all {}'.format(num_images)
    gen.eval()
    z = torch.randn(size=(num_images, 100)).to(device)
    generated = gen(cut_imgs[:num_images], labels[:num_images], z)
    fig, axes = plt.subplots(3, num_images, figsize=(12,6))
    for i in range(num_images):
        axes[0, i].imshow(full_imgs[i].view(256,256).cpu().detach(), cmap="Greys")
        axes[0, i].axis('off')
        axes[1, i].imshow(cut_imgs[i].view(256,256).cpu().detach(), cmap="Greys")
        axes[1, i].axis('off')
        axes[2, i].imshow(generated[i].detach().view(256,256).cpu().detach(), cmap="Greys")
        axes[2, i].axis('off')
        # axes[2, i].set_title(labels[i].item())

    gen.train()
    return fig
