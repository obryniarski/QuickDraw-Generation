import numpy
import torch
import torchvision
import torch.nn as nn
from torchsummary import summary

# residual block as in ResNet
class ResBlock(nn.Module):
    def __init__(self, in_filters, num_filters, size=None, slope=0.02, bn=True):
        super(ResBlock, self).__init__()
        self.bn = bn

        self.scale = nn.utils.spectral_norm(nn.Conv2d(in_filters, num_filters, kernel_size=1, stride=1, padding=0))

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_filters, num_filters, kernel_size=3, stride=1, padding=1))
        self.relu1 = nn.LeakyReLU(slope)
        self.bn1 = nn.BatchNorm2d(num_filters) if self.bn else nn.LayerNorm([num_filters, size, size])

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
        self.relu2 = nn.LeakyReLU(slope)
        self.bn2 = nn.BatchNorm2d(num_filters) if self.bn else nn.LayerNorm([num_filters, size, size])

        # self.conv3 = nn.utils.spectral_norm(nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1))
        # self.relu3 = nn.LeakyReLU(slope)
        # self.bn3 = nn.BatchNorm2d(num_filters) if self.bn else nn.LayerNorm([num_filters, size, size])


    def forward(self, input):
        features = self.relu1(self.bn1(self.conv1(input)))
        features = self.bn2(self.conv2(features))
        # features = self.bn3(self.conv3(features))

        out = self.relu2(features + self.scale(input))
        return out


# as described in https://arxiv.org/pdf/1805.08318.pdf
# implementation by https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class SelfAttention(nn.Module):
    def __init__(self, in_filters, k=8):
        super(SelfAttention, self).__init__()
        self.f = nn.utils.spectral_norm(nn.Conv2d(in_filters, in_filters // k, kernel_size=1, stride=1, padding=0))
        self.g = nn.utils.spectral_norm(nn.Conv2d(in_filters, in_filters // k, kernel_size=1, stride=1, padding=0))
        # self.h = nn.utils.spectral_norm(nn.Conv2d(in_filters, in_filters // k, kernel_size=1, stride=1, padding=0))

        self.v = nn.utils.spectral_norm(nn.Conv2d(in_filters, in_filters, kernel_size=1, stride=1, padding=0))

        self.softmax = nn.Softmax(dim=-1) # softmax over each row: (b, c, -> h, w)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):

        # f_x = self.f(input)
        # g_x = self.g(input)
        # attention_map = self.softmax(torch.matmul(torch.transpose(f_x, 2, 3), g_x))
        # h_x = self.h(input)
        # o = self.v(torch.matmul(attention_map, h_x))
        #
        # return self.gamma * o + input

        m_batchsize,C,width ,height = input.size()
        proj_query  = self.f(input).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.g(input).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.v(input).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + input
        return out



# passes input all the way to end, so we are only adding to it (no subtraction because of sigmoid)
class Add_Skip_Generator(nn.Module):
    def __init__(self, imsize=28):
        super(Add_Skip_Generator, self).__init__()
        F = 100
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

        # self.d_conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*4, F*2, kernel_size=3, stride=2, padding=0))
        self.d_conv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(F*4, F*2, kernel_size=4, stride=2, padding=1))

        self.d_bn1 = nn.BatchNorm2d(F*2)
        self.d_relu1 = nn.LeakyReLU(slope)

        self.resblock4 = ResBlock(2 * F*2, 2 * F*2)
        self.d_conv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(2 * F*2, F, kernel_size=4, stride=2, padding=1))
        self.d_bn2 = nn.BatchNorm2d(F)
        self.d_relu2 = nn.LeakyReLU(slope)


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

        # in bottleneck
        bottleneck = self.resblock3(torch.cat([bottleneck, up_class, noise], dim=1))
        # print(bottleneck.shape)
        bottleneck = self.dropout1(bottleneck)
        # print(bottleneck.shape)
        # decoder
        upsampled = self.d_relu1(self.d_bn1(self.d_conv1(bottleneck)))
        # print(upsampled.shape, second_skip.shape)
        upsampled = self.resblock4(torch.cat([upsampled, second_skip], dim=1))
        upsample = self.dropout2(upsampled)
        upsampled = self.d_relu2(self.d_bn2(self.d_conv2(upsampled)))
        upsampled = self.resblock5(torch.cat([upsampled, first_skip], dim=1))

        return torch.clamp(self.sig(self.final_conv(upsampled)) + input, min=0, max=1)

def bottle_size(imsize, n):
    for i in range(n):
        imsize //= 2
    return imsize

def receptive_field(n):
    f = lambda out_size, ksize, stride: (out_size - 1) * stride + ksize
    if n == 0:
        return 1
    return f(receptive_field(n-1), 4, 2)

# passes input all the way to end, so we are only adding to it (no subtraction because of sigmoid)
class Deep_Generator(nn.Module):
    def __init__(self, imsize=28, F=16):
        super(Deep_Generator, self).__init__()
        # F = 16
        slope = 0.02

        self.imsize = imsize
        self.bottleneck_size = bottle_size(imsize, 4)
        self.linear1 = nn.Linear(20, self.bottleneck_size ** 2)
        self.linear2 = nn.Linear(100, self.bottleneck_size ** 2)


        # self.dropout1 = nn.Dropout2d(p=0.5)
        # self.dropout2 = nn.Dropout2d(p=0.5)

        self.downscale1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F),
                                        nn.LeakyReLU(slope)) # 64

        self.downscale2 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*2),
                                        nn.LeakyReLU(slope)) # 64

        self.downscale3 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*4),
                                        nn.LeakyReLU(slope)) # 128

        self.downscale4 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*4),
                                        nn.LeakyReLU(slope)) # 128

        # bottleneck

        self.e_resblock1 = ResBlock(1, F)
        self.e_resblock2 = ResBlock(F, F*2)
        self.e_resblock3 = ResBlock(F*2, F*4)
        self.e_resblock4 = ResBlock(F*4, F*4)

        self.d_resblock1 = ResBlock(2 + F*4, F*4)
        self.d_resblock2 = ResBlock(2 * F*4, F*4)
        self.d_resblock3 = ResBlock(4 * F + 2*F, F*2)
        self.d_resblock4 = ResBlock(2 * F + F, F)


        # decoder
        self.upscale1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*4),
                                      nn.LeakyReLU(slope)) # 512

        self.upscale2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*4),
                                      nn.LeakyReLU(slope)) # 512

        self.upscale3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*2),
                                      nn.LeakyReLU(slope)) # 512

        self.upscale4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F, 1, kernel_size=3, stride=1, padding=1)),
                                      nn.Sigmoid())


    def forward(self, input, label, noise):
        noise = self.linear2(noise).view(-1, 1, self.bottleneck_size, self.bottleneck_size)
        up_class = self.linear1(label).view(-1, 1, self.bottleneck_size, self.bottleneck_size)

        # input = torch.cat([input, up_class, noise], dim=1)

        # encoder
        # class_info = torch.cat([input, up_class], dim=1)
        first_skip = self.downscale1(self.e_resblock1(input))
        second_skip = self.downscale2(self.e_resblock2(first_skip))
        third_skip = self.downscale3(self.e_resblock3(second_skip))
        bottleneck = self.downscale4(self.e_resblock4(third_skip))


        # bottleneck
        bottleneck = torch.cat([bottleneck, up_class, noise], dim=1)

        # decoder
        upsampled = torch.cat([self.upscale1(self.d_resblock1(bottleneck)), third_skip], dim=1)
        upsampled = torch.cat([self.upscale2(self.d_resblock2(upsampled)), second_skip], dim=1)
        upsampled = torch.cat([self.upscale3(self.d_resblock3(upsampled)), first_skip], dim=1)

        return torch.clamp(self.upscale4(self.d_resblock4(upsampled)) + input, min=0, max=1)

class SA_Generator(nn.Module):
    def __init__(self, imsize=28, F=16):
        super(SA_Generator, self).__init__()
        # F = 16
        slope = 0.02

        self.imsize = imsize
        self.bottleneck_size = bottle_size(imsize, 4)
        self.linear1 = nn.utils.spectral_norm(nn.Linear(20, self.imsize ** 2))
        self.linear2 = nn.utils.spectral_norm(nn.Linear(100, self.bottleneck_size ** 2))


        # self.dropout1 = nn.Dropout2d(p=0.5)
        # self.dropout2 = nn.Dropout2d(p=0.5)

        self.downscale1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F),
                                        nn.LeakyReLU(slope)) # 64

        self.downscale2 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*2),
                                        nn.LeakyReLU(slope)) # 64

        self.downscale3 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*4),
                                        nn.LeakyReLU(slope)) # 128

        self.downscale4 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=2, padding=1)),
                                        nn.BatchNorm2d(F*4),
                                        nn.LeakyReLU(slope)) # 128

        self.attention1 = SelfAttention(F*2)
        self.attention2 = SelfAttention(F*4)


        # bottleneck

        self.e_resblock1 = ResBlock(2, F)
        self.e_resblock2 = ResBlock(F, F*2)
        self.e_resblock3 = ResBlock(F*2, F*4)
        self.e_resblock4 = ResBlock(F*4, F*4)

        self.d_resblock1 = ResBlock(1 + F*4, F*4)
        self.d_resblock2 = ResBlock(2 * F*4, F*4)
        self.d_resblock3 = ResBlock(4 * F + 2*F, F*2)
        self.d_resblock4 = ResBlock(2 * F + F, F)


        # decoder
        self.upscale1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*4),
                                      nn.LeakyReLU(slope),
                                      nn.Dropout2d(0.5)) # 512

        self.upscale2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*4, F*4, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*4),
                                      nn.LeakyReLU(slope)) # 512

        self.upscale3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=1, padding=1)),
                                      nn.BatchNorm2d(F*2),
                                      nn.LeakyReLU(slope)) # 512

        self.upscale4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                      nn.utils.spectral_norm(nn.Conv2d(F, 1, kernel_size=3, stride=1, padding=1)),
                                      nn.Sigmoid())


    def forward(self, input, label, noise):
        noise = self.linear2(noise).view(-1, 1, self.bottleneck_size, self.bottleneck_size)
        up_class = self.linear1(label).view(-1, 1, self.imsize, self.imsize)

        # input = torch.cat([input, up_class, noise], dim=1)

        # encoder
        class_info = torch.cat([input, up_class], dim=1)
        first_skip = self.downscale1(self.e_resblock1(class_info))
        second_skip = self.attention1(self.downscale2(self.e_resblock2(first_skip)))
        third_skip = self.downscale3(self.e_resblock3(second_skip))
        bottleneck = self.downscale4(self.e_resblock4(third_skip))


        # bottleneck
        bottleneck = torch.cat([bottleneck, noise], dim=1)

        # decoder
        upsampled = torch.cat([self.upscale1(self.d_resblock1(bottleneck)), third_skip], dim=1)
        upsampled = torch.cat([self.attention2(self.upscale2(self.d_resblock2(upsampled))), second_skip], dim=1)
        upsampled = torch.cat([self.upscale3(self.d_resblock3(upsampled)), first_skip], dim=1)

        return torch.clamp(self.upscale4(self.d_resblock4(upsampled)) + input, min=0, max=1)


class Discriminator(nn.Module):
    def __init__(self, imsize=28):
        super(Discriminator, self).__init__()
        F = 32
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
        # return self.final_conv(input).view(-1) # for wgan

class Deep_Discriminator(nn.Module):
    def __init__(self, imsize=28, F=32):
        super(Deep_Discriminator, self).__init__()
        # F = 32
        slope = 0.02

        self.imsize = imsize
        self.linear = nn.Linear(20, self.imsize**2)

        self.block1 = nn.Sequential(ResBlock(2, F, bn=False, size=self.imsize),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 2, self.imsize // 2]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.5)) # 64
        self.block2 = nn.Sequential(ResBlock(F, F, bn=False, size=self.imsize // 2),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 4, self.imsize // 4]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.5)) # 128
        self.block3 = nn.Sequential(ResBlock(F, F, bn=False, size=self.imsize // 4),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 8, self.imsize // 8]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.5)) # 256
        self.block4 = nn.Sequential(ResBlock(F, F*2, bn=False, size=self.imsize // 8),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 16, self.imsize // 16]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.5)) # 256
        self.block5 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 16),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 32, self.imsize // 32]),
                                    nn.LeakyReLU(slope)) # 512
        self.block6 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 32),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 64, self.imsize // 64]),
                                    nn.LeakyReLU(slope)) # 512
        self.block7 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 64),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 128, self.imsize // 128]),
                                    nn.LeakyReLU(slope)) # 512

        # self.resblock1 = ResBlock(F*4, F*4)
        # self.resblock2 = ResBlock(F*4, F*4)
        # self.resblock3 = ResBlock(F*4, F*4)
        # self.resblock4 = ResBlock(F*4, F*4)

        self.block8 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*2, 1, kernel_size=3, stride=2, padding=1)))

    def forward(self, completed, label):

        up_label = self.linear(label).view(-1, 1, self.imsize, self.imsize)
        input = torch.cat([completed, up_label], dim=1)

        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.block4(input)
        input = self.block5(input)
        input = self.block6(input)
        input = self.block7(input)

        return self.block8(input).view(-1, 1)

class SA_Discriminator(nn.Module):
    def __init__(self, imsize=28, F=32):
        super(SA_Discriminator, self).__init__()
        # F = 32
        slope = 0.02

        self.imsize = imsize
        self.linear = nn.utils.spectral_norm(nn.Linear(20, self.imsize**2))

        self.block1 = nn.Sequential(ResBlock(2, F, bn=False, size=self.imsize),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 2, self.imsize // 2]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.3)) # 64
        self.block2 = nn.Sequential(ResBlock(F, F, bn=False, size=self.imsize // 2),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 4, self.imsize // 4]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.3)) # 128
        self.block3 = nn.Sequential(ResBlock(F, F, bn=False, size=self.imsize // 4),
                                    nn.utils.spectral_norm(nn.Conv2d(F, F, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F, self.imsize // 8, self.imsize // 8]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.3)) # 256
        self.block4 = nn.Sequential(ResBlock(F, F*2, bn=False, size=self.imsize // 8),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 16, self.imsize // 16]),
                                    nn.LeakyReLU(slope),
                                    nn.Dropout2d(0.3)) # 256
        self.block5 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 16),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 32, self.imsize // 32]),
                                    nn.LeakyReLU(slope)) # 512
        self.block6 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 32),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 64, self.imsize // 64]),
                                    nn.LeakyReLU(slope)) # 512
        self.block7 = nn.Sequential(ResBlock(F*2, F*2, bn=False, size=self.imsize // 64),
                                    nn.utils.spectral_norm(nn.Conv2d(F*2, F*2, kernel_size=3, stride=2, padding=1)),
                                    nn.LayerNorm([F*2, self.imsize // 128, self.imsize // 128]),
                                    nn.LeakyReLU(slope)) # 512

        # self.resblock1 = ResBlock(F*4, F*4)
        # self.resblock2 = ResBlock(F*4, F*4)
        # self.resblock3 = ResBlock(F*4, F*4)
        # self.resblock4 = ResBlock(F*4, F*4)

        self.block8 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(F*2, 1, kernel_size=3, stride=2, padding=1)))


        self.attention1 = SelfAttention(F*2)
        self.attention2 = SelfAttention(F*2)


    def forward(self, completed, label):

        up_label = self.linear(label).view(-1, 1, self.imsize, self.imsize)
        input = torch.cat([completed, up_label], dim=1)

        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.attention1(self.block4(input))
        input = self.block5(input)
        input = self.attention2(self.block6(input))
        input = self.block7(input)

        return self.block8(input).view(-1, 1)

import numpy as np
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

# gen = Generator().to(0)
# summary(gen, [(2, 28, 28), (1, 28, 28)])
#
# skip_gen = SA_Generator(imsize=256, F=64).to(0)
# print(num_params(skip_gen))

# self_attention = SelfAttention(32).to(0)
# summary(self_attention, [(32, 256, 256)])
# print([param[0] for param in self_attention.named_parameters()])


# print([param[0] for param in skip_gen.named_parameters()])
# summary(skip_gen, [(1, 256, 256), (1, 20), (1, 100)])


# disc = SA_Discriminator(imsize=256, F=64).to(0)
# print(num_params(disc))
# print([param[0] for param in disc.named_parameters()])

# summary(disc, [(1, 256, 256), (1, 20)])
