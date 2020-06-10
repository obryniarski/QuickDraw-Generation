import numpy
import torch
import torchvision
import torch.nn as nn
from torchsummary import summary


class ResBlock(nn.Module):
    def __init__(self, in_filters, num_filters, slope=0.02, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn

        self.scale = nn.utils.spectral_norm(nn.Conv2d(in_filters, num_filters, kernel_size=1, stride=1, padding=0))

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1))
        self.relu1 = nn.LeakyReLU(slope)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
        if self.bn:
            self.bn1 = nn.BatchNorm2d(num_filters)
            self.bn2 = nn.BatchNorm2d(num_filters)

        self.relu2 = nn.LeakyReLU(slope)


    def forward(self, input):
        if self.bn:
            features = self.relu1(self.bn1(self.conv1(input)))
            features = self.bn2(self.conv2(features))
        else:
            features = self.relu1(self.conv1(input))
            features = self.conv2(features)

        out = self.relu2(features + self.scale(input))
        return out



class Generator(nn.Module):
    def __init__(self, imsize=28):
        super(Generator, self).__init__()
        F = 64
        slope = 0.02

        self.imsize = imsize

        self.linear1 = nn.Linear(1, self.imsize**2)
        self.l_relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, self.imsize**2)
        self.l_relu2 = nn.ReLU()

        self.e_conv1 = nn.utils.spectral_norm(nn.Conv2d(3, F, kernel_size=4, stride=2, padding=1))
        self.e_relu1 = nn.LeakyReLU(slope)
        self.e_conv2 = nn.utils.spectral_norm(nn.Conv2d(F, F*2, kernel_size=4, stride=2, padding=1))
        self.e_bn2 = nn.BatchNorm2d(F*2)
        self.e_relu2 = nn.LeakyReLU(slope)
        self.e_conv3 = nn.utils.spectral_norm(nn.Conv2d(F*2, F*4, kernel_size=4, stride=2, padding=1))
        self.e_bn3 = nn.BatchNorm2d(F*4)
        self.e_relu3 = nn.LeakyReLU(slope)

        self.d_conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*4, F*2, kernel_size=3, stride=2, padding=0))
        self.d_bn1 = nn.BatchNorm2d(F*2)
        self.d_relu1 = nn.ReLU(0.2)
        self.d_conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*2, F, kernel_size=4, stride=2, padding=1))
        self.d_bn2 = nn.BatchNorm2d(F)
        self.d_relu2 = nn.ReLU(0.2)
        # self.d_conv3 = nn.ConvTranspose2d(F, F, kernel_size=4, stride=2, padding=1)
        # self.d_bn3 = nn.BatchNorm2d(F)
        # self.d_relu3 = nn.ReLU(0.2)

        self.final_conv = nn.utils.spectral_norm(nn.ConvTranspose2d(F, 1, kernel_size=4, stride=2, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, input, label, noise):
        noise = self.l_relu1(self.linear2(noise).view(-1, 1, self.imsize, self.imsize))
        up_class = self.l_relu2(self.linear1(label).view(-1, 1, self.imsize, self.imsize))

        input = torch.cat([input, up_class, noise], dim=1)

        # encoder
        input = self.e_relu1(self.e_conv1(input))
        input = self.e_relu2(self.e_bn2(self.e_conv2(input)))
        input = self.e_relu3(self.e_bn3(self.e_conv3(input)))

        # decoder
        input = self.d_relu1(self.d_bn1(self.d_conv1(input)))
        input = self.d_relu2(self.d_bn2(self.d_conv2(input)))
        # input = self.d_relu3(self.d_bn3(self.d_conv3(input)))

        return self.sig(self.final_conv(input))




class Skip_Generator(nn.Module):
    def __init__(self, imsize=28):
        super(Skip_Generator, self).__init__()
        F = 64
        slope = 0.02

        self.bottleneck_size = (((imsize // 2) // 2) // 2)
        self.linear1 = nn.Linear(1, self.bottleneck_size ** 2)
        self.l_relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, self.bottleneck_size ** 2)
        self.l_relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout2d(p=0.5)
        self.dropout2 = nn.Dropout2d(p=0.5)

        self.e_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, F, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_relu1 = nn.LeakyReLU(slope)
        self.resblock1 = ResBlock(F, F)

        self.e_conv2 = nn.utils.spectral_norm(nn.Conv2d(F, F*2, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_bn2 = nn.BatchNorm2d(F*2)
        self.e_relu2 = nn.LeakyReLU(slope)
        self.resblock2 = ResBlock(F*2, F*2)

        self.e_conv3 = nn.utils.spectral_norm(nn.Conv2d(F*2, F*4, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_bn3 = nn.BatchNorm2d(F*4)
        self.e_relu3 = nn.LeakyReLU(slope)


        self.resblock3 = ResBlock(2 + F*4, F*4)

        self.d_conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*4, F*2, kernel_size=3, stride=2, padding=0))
        self.d_bn1 = nn.BatchNorm2d(F*2)
        self.d_relu1 = nn.LeakyReLU(slope)

        self.resblock4 = ResBlock(2 * F*2, 2 * F*2)
        self.d_conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(2 * F*2, F, kernel_size=4, stride=2, padding=1))
        self.d_bn2 = nn.BatchNorm2d(F)
        self.d_relu2 = nn.LeakyReLU(slope)
        # self.d_conv3 = nn.ConvTranspose2d(F, 1, kernel_size=4, stride=2, padding=1)
        # self.d_bn3 = nn.BatchNorm2d(F)
        # self.d_relu3 = nn.ReLU(0.2)

        self.resblock5 = ResBlock(2 * F, 2 * F)
        self.final_conv = nn.utils.spectral_norm(nn.ConvTranspose2d(2 * F, 1, kernel_size=4, stride=2, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, input, label, noise):
        noise = self.l_relu2(self.linear2(noise).view(-1, 1, self.bottleneck_size, self.bottleneck_size))
        up_class = self.l_relu1(self.linear1(label).view(-1, 1, self.bottleneck_size, self.bottleneck_size))

        # input = torch.cat([input, up_class, noise], dim=1)

        # encoder
        first_skip = self.resblock1(self.e_relu1(self.e_conv1(input)))
        second_skip = self.resblock2(self.e_relu2(self.e_bn2(self.e_conv2(first_skip))))
        bottleneck = self.e_relu3(self.e_bn3(self.e_conv3(second_skip)))


        bottleneck = self.resblock3(torch.cat([bottleneck, up_class, noise], dim=1))
        bottleneck = self.dropout1(bottleneck)
        # decoder
        upsampled = self.d_relu1(self.d_bn1(self.d_conv1(bottleneck)))
        upsampled = self.resblock4(torch.cat([upsampled, second_skip], dim=1))
        upsample = self.dropout2(upsampled)
        upsampled = self.d_relu2(self.d_bn2(self.d_conv2(upsampled)))
        upsampled = self.resblock5(torch.cat([upsampled, first_skip], dim=1))
        return self.sig(self.final_conv(upsampled))




class Add_Skip_Generator(nn.Module):
    def __init__(self, imsize=28):
        super(Add_Skip_Generator, self).__init__()
        F = 64
        slope = 0.02

        self.bottleneck_size = (((imsize // 2) // 2) // 2)
        self.linear1 = nn.Linear(1, self.bottleneck_size ** 2)
        self.l_relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, self.bottleneck_size ** 2)
        self.l_relu2 = nn.ReLU()

        self.dropout1 = nn.Dropout2d(p=0.5)
        self.dropout2 = nn.Dropout2d(p=0.5)

        self.e_conv1 = nn.utils.spectral_norm(nn.Conv2d(1, F, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_relu1 = nn.LeakyReLU(slope)
        self.resblock1 = ResBlock(F, F)

        self.e_conv2 = nn.utils.spectral_norm(nn.Conv2d(F, F*2, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_bn2 = nn.BatchNorm2d(F*2)
        self.e_relu2 = nn.LeakyReLU(slope)
        self.resblock2 = ResBlock(F*2, F*2)

        self.e_conv3 = nn.utils.spectral_norm(nn.Conv2d(F*2, F*4, kernel_size=4, stride=2, padding=1)) # floor(/2)
        self.e_bn3 = nn.BatchNorm2d(F*4)
        self.e_relu3 = nn.LeakyReLU(slope)


        self.resblock3 = ResBlock(2 + F*4, F*4)

        self.d_conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*4, F*2, kernel_size=3, stride=2, padding=0))
        self.d_bn1 = nn.BatchNorm2d(F*2)
        self.d_relu1 = nn.LeakyReLU(slope)

        self.resblock4 = ResBlock(2 * F*2, 2 * F*2)
        self.d_conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(2 * F*2, F, kernel_size=4, stride=2, padding=1))
        self.d_bn2 = nn.BatchNorm2d(F)
        self.d_relu2 = nn.LeakyReLU(slope)
        # self.d_conv3 = nn.ConvTranspose2d(F, 1, kernel_size=4, stride=2, padding=1)
        # self.d_bn3 = nn.BatchNorm2d(F)
        # self.d_relu3 = nn.ReLU(0.2)

        self.resblock5 = ResBlock(2 * F, 2 * F)
        self.final_conv = nn.utils.spectral_norm(nn.ConvTranspose2d(2 * F, 1, kernel_size=4, stride=2, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, input, label, noise):
        noise = self.l_relu2(self.linear2(noise).view(-1, 1, self.bottleneck_size, self.bottleneck_size))
        up_class = self.l_relu1(self.linear1(label).view(-1, 1, self.bottleneck_size, self.bottleneck_size))

        # input = torch.cat([input, up_class, noise], dim=1)

        # encoder
        first_skip = self.resblock1(self.e_relu1(self.e_conv1(input)))
        second_skip = self.resblock2(self.e_relu2(self.e_bn2(self.e_conv2(first_skip))))
        bottleneck = self.e_relu3(self.e_bn3(self.e_conv3(second_skip)))


        bottleneck = self.resblock3(torch.cat([bottleneck, up_class, noise], dim=1))
        bottleneck = self.dropout1(bottleneck)
        # decoder
        upsampled = self.d_relu1(self.d_bn1(self.d_conv1(bottleneck)))
        upsampled = self.resblock4(torch.cat([upsampled, second_skip], dim=1))
        upsample = self.dropout2(upsampled)
        upsampled = self.d_relu2(self.d_bn2(self.d_conv2(upsampled)))
        upsampled = self.resblock5(torch.cat([upsampled, first_skip], dim=1))

        return torch.clamp(self.sig(self.final_conv(upsampled)) + input, min=0, max=1)


class Discriminator(nn.Module):
    def __init__(self, imsize=28):
        super(Discriminator, self).__init__()
        F = 64
        slope = 0.02

        self.imsize = imsize
        self.linear = nn.Linear(1, self.imsize**2)

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(2, F, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(F)
        self.relu1 = nn.LeakyReLU(slope)
        self.resblock1 = ResBlock(F, F)

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(F, F*2, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(F*2)
        self.relu2 = nn.LeakyReLU(slope)
        self.resblock2 = ResBlock(F*2, F*2)

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(F*2, F*4, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(F*4)
        self.relu3 = nn.LeakyReLU(slope)
        self.resblock3 = ResBlock(F*4, F*4)

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(F*4, 1, kernel_size=4, stride=2, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, completed, label):

        up_label = self.linear(label).view(-1, 1, self.imsize, self.imsize)
        input = torch.cat([completed, up_label], dim=1)

        input = self.resblock1(self.relu1(self.bn1(self.conv1(input))))
        input = self.resblock2(self.relu2(self.bn2(self.conv2(input))))
        input = self.resblock3(self.relu3(self.bn3(self.conv3(input))))

        return self.sig(self.final_conv(input)).view(-1, 1)
        # return self.final_conv(input).view(-1)

class DiscriminatorWithSketch(nn.Module):
    def __init__(self, imsize=28):
        super(DiscriminatorWithSketch, self).__init__()
        F = 64
        slope = 0.02

        self.imsize = imsize
        self.linear = nn.Linear(1, self.imsize**2)

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(3, F, kernel_size=4, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(F)
        self.relu1 = nn.LeakyReLU(slope)
        self.resblock1 = ResBlock(F, F)

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(F, F*2, kernel_size=4, stride=2, padding=1))
        self.bn2 = nn.BatchNorm2d(F*2)
        self.relu2 = nn.LeakyReLU(slope)
        self.resblock2 = ResBlock(F*2, F*2)

        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(F*2, F*4, kernel_size=4, stride=2, padding=1))
        self.bn3 = nn.BatchNorm2d(F*4)
        self.relu3 = nn.LeakyReLU(slope)
        self.resblock3 = ResBlock(F*4, F*4)

        self.final_conv = nn.utils.spectral_norm(nn.Conv2d(F*4, 1, kernel_size=4, stride=2, padding=1))
        self.sig = nn.Sigmoid()

    def forward(self, completed, sketch, label):

        up_label = self.linear(label).view(-1, 1, self.imsize, self.imsize)
        input = torch.cat([completed, sketch, up_label], dim=1)

        input = self.resblock1(self.relu1(self.bn1(self.conv1(input))))
        input = self.resblock2(self.relu2(self.bn2(self.conv2(input))))
        input = self.resblock3(self.relu3(self.bn3(self.conv3(input))))

        return self.sig(self.final_conv(input)).view(-1, 1)
        # return self.final_conv(input).view(-1)

import numpy as np
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

# gen = Generator().to(0)
# summary(gen, [(2, 28, 28), (1, 28, 28)])

# skip_gen = Skip_Generator().to(0)
# print(num_params(skip_gen))

# print([param[0] for param in skip_gen.named_parameters()])
# summary(skip_gen, [(1, 28, 28), (1, 1), (1, 100)])
#
# disc = Discriminator().to(0)
# print(num_params(disc))
# print([param[0] for param in disc.named_parameters()])

# summary(disc, [(1, 28, 28), (1, 1)])
