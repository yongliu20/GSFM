import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import mod_resnet
from model import cbam


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
 
        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)
 
    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = cbam.CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)


    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x

class ValueEncoderSO(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x


# Multiple objects version, used in other times
class ValueEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = mod_resnet.resnet18(pretrained=True, extra_chan=2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1 # 1/4, 64
        self.layer2 = resnet.layer2 # 1/8, 128
        self.layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

    def forward(self, image, key_f16, mask, other_masks):
        # key_f16 is the feature from the key encoder

        f = torch.cat([image, mask, other_masks], 1)

        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x) # 1/8, 128
        x = self.layer3(x) # 1/16, 256

        x = self.fuser(x, key_f16)

        return x
 

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = mod_resnet.resnet50(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.layer2 = resnet.layer2 # 1/8, 512
        self.layer3 = resnet.layer3 # 1/16, 1024

    def forward(self, f):
        # print(f.dtype)
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x) # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024

        return f16, f8, f4



class UpsampleBlock(nn.Module):
    def __init__(self, skip_c, up_c, out_c, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_c, up_c, kernel_size=3, padding=1)
        self.out_conv = ResBlock(up_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, skip_f, up_f):
        x = self.skip_conv(skip_f)
        x = x + F.interpolate(up_f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x):
        return self.key_proj(x)

class LFM(nn.Module):
    def __init__(self, num_channels):
        super(LFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()


        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)       #1, 1, H, W


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)


        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        y = y * (1 - high_filter)

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y

class HFM(nn.Module):
    def __init__(self, num_channels):
        super(HFM, self).__init__()
        self.conv1 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2 * num_channels, 2 * num_channels, kernel_size=1, stride=1, padding=0)

    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()


        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))

        return g.unsqueeze(0)       #1, 1, H, W


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x)


        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w)
        y = y * high_filter

        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=1)
        y = F.relu(self.conv1(y_f))

        y = self.conv2(y).float()
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)

        y = torch.fft.ifft2(y, s=(h, w)).float()
        return x + y




