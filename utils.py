#!/usr/bin/env python
# coding: utf-8

# In[1]:

import math
from torch import nn
import torch.nn.functional as F
import torch


# In[4]:

def make_coordinate_grid(size, type):
    x = torch.arange(size[1]).type(type)
    y = torch.arange(size[0]).type(type)

    x = ((x*2) / (size[1] - 1)) - 1
    y = ((y*2) / (size[0] - 1)) - 1

    x_repeated = x.view(1, -1).repeat(size[0], 1)
    y_repeated = y.view(-1, 1).repeat(1, size[1])
    
    return torch.cat([x_repeated.unsqueeze_(2), y_repeated.unsqueeze_(2)], 2)


class AntiAliasInterpolation2d(nn.Module):
    def __init__(self, channels, scale):
        # super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = math.floor(kernel_size / 2)
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        
        kernel = 1
        meshgrids = torch.meshgrid([
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out




class DownBlock2d(nn.Module):
   
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


# In[5]:


class UpBlock2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


# In[6]:


class Hourglass(nn.Module):
  
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.DB1 = DownBlock2d(in_features ,64, kernel_size=3, padding=1)
        self.DB2 = DownBlock2d(64 ,128, kernel_size=3, padding=1)
        self.DB3 = DownBlock2d(128 ,265, kernel_size=3, padding=1)
        self.DB4 = DownBlock2d(265 ,512, kernel_size=3, padding=1)
        self.DB5 = DownBlock2d(512 ,1024, kernel_size=3, padding=1)
        
        self.UB1 = UpBlock2d(1024 ,512, kernel_size=3, padding=1)
        self.UB2 = UpBlock2d(1024 ,265, kernel_size=3, padding=1)
        self.UB3 = UpBlock2d(512 ,128, kernel_size=3, padding=1)
        self.UB4 = UpBlock2d(265 ,64, kernel_size=3, padding=1)
        self.UB5 = UpBlock2d(128 ,32, kernel_size=3, padding=1)
        self.out_filters = block_expansion + in_features
     

    def forward(self, x):
 
        out1 =  self.DB1 (x)
        out2 =  self.DB2 (out1)
        out3 =  self.DB3 (out2)
        out4 =  self.DB4 (out3)
        out5 =   self.DB5 (out4)
        
        out =  self.UB1 (out5) 
        out =  self.DB2 (torch.cat([out, out4], dim=1))
        out =  self.DB3 (torch.cat([out, out3], dim=1))
        out =  self.DB4 (torch.cat([out, out2], dim=1))
        out =  self.DB5 (torch.cat([out, out1], dim=1))
        return out


# In[7]:


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm(x)
        out = F.relu(out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        out = self.conv(out)
        out += x
        return out


# In[8]:


class SameBlock2d(nn.Module):
    
    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


# In[ ]:




