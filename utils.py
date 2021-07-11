#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import math
# from torch import nn
# import torch.nn.functional as F
# import torch


# In[4]:

# def make_coordinate_grid(size, type):
#     x = torch.arange(size[1]).type(type)
#     y = torch.arange(size[0]).type(type)

#     x = ((x*2) / (size[1] - 1)) - 1
#     y = ((y*2) / (size[0] - 1)) - 1

#     x_repeated = x.view(1, -1).repeat(size[0], 1)
#     y_repeated = y.view(-1, 1).repeat(1, size[1])
    
#     return torch.cat([x_repeated.unsqueeze_(2), y_repeated.unsqueeze_(2)], 2)


# class AntiAliasInterpolation2d(nn.Module):
#     def __init__(self, channels, scale):
#         # super(AntiAliasInterpolation2d, self).__init__()
#         sigma = (1 / scale - 1) / 2
#         kernel_size = 2 * round(sigma * 4) + 1
#         self.ka = math.floor(kernel_size / 2)
#         self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

#         kernel_size = [kernel_size, kernel_size]
#         sigma = [sigma, sigma]
        
#         kernel = 1
#         meshgrids = torch.meshgrid([
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#                 ])
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
#         self.groups = channels
#         self.scale = scale
#         inv_scale = 1 / scale
#         self.int_inv_scale = int(inv_scale)

#     def forward(self, input):
#         if self.scale == 1.0:
#             return input

#         out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
#         out = F.conv2d(out, weight=self.weight, groups=self.groups)
#         out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

#         return out




# class DownBlock2d(nn.Module):
   
#     def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
#         super(DownBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, groups=groups)
#         self.norm = BatchNorm2d(out_features, affine=True)
#         self.pool = nn.AvgPool2d(kernel_size=(2, 2))

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = F.relu(out)
#         out = self.pool(out)
#         return out


# # In[5]:


# class UpBlock2d(nn.Module):

#     def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
#         super(UpBlock2d, self).__init__()

#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
#                               padding=padding, groups=groups)
#         self.norm = BatchNorm2d(out_features, affine=True)

#     def forward(self, x):
#         out = F.interpolate(x, scale_factor=2)
#         out = self.conv(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out


# # In[6]:


# class Hourglass(nn.Module):
  
#     def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
#         super(Hourglass, self).__init__()
#         self.DB1 = DownBlock2d(in_features ,64, kernel_size=3, padding=1)
#         self.DB2 = DownBlock2d(64 ,128, kernel_size=3, padding=1)
#         self.DB3 = DownBlock2d(128 ,265, kernel_size=3, padding=1)
#         self.DB4 = DownBlock2d(265 ,512, kernel_size=3, padding=1)
#         self.DB5 = DownBlock2d(512 ,1024, kernel_size=3, padding=1)
        
#         self.UB1 = UpBlock2d(1024 ,512, kernel_size=3, padding=1)
#         self.UB2 = UpBlock2d(1024 ,265, kernel_size=3, padding=1)
#         self.UB3 = UpBlock2d(512 ,128, kernel_size=3, padding=1)
#         self.UB4 = UpBlock2d(265 ,64, kernel_size=3, padding=1)
#         self.UB5 = UpBlock2d(128 ,32, kernel_size=3, padding=1)
#         self.out_filters = block_expansion + in_features
     

#     def forward(self, x):
 
#         out1 =  self.DB1 (x)
#         out2 =  self.DB2 (out1)
#         out3 =  self.DB3 (out2)
#         out4 =  self.DB4 (out3)
#         out5 =   self.DB5 (out4)
        
#         out =  self.UB1 (out5) 
#         out =  self.DB2 (torch.cat([out, out4], dim=1))
#         out =  self.DB3 (torch.cat([out, out3], dim=1))
#         out =  self.DB4 (torch.cat([out, out2], dim=1))
#         out =  self.DB5 (torch.cat([out, out1], dim=1))
#         return out


# # In[7]:


# class ResBlock2d(nn.Module):
#     """
#     Res block, preserve spatial resolution.
#     """

#     def __init__(self, in_features, kernel_size, padding):
#         super(ResBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
#                                padding=padding)
#         self.norm = BatchNorm2d(in_features, affine=True)

#     def forward(self, x):
#         out = self.norm(x)
#         out = F.relu(out)
#         out = self.conv(out)
#         out = self.norm(out)
#         out = F.relu(out)
#         out = self.conv(out)
#         out += x
#         return out


# # In[8]:


# class SameBlock2d(nn.Module):
    
#     def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
#         super(SameBlock2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
#                               kernel_size=kernel_size, padding=padding, groups=groups)
#         self.norm = BatchNorm2d(out_features, affine=True)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.norm(out)
#         out = F.relu(out)
#         return out


# # In[ ]:




#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn
import torch.nn.functional as F
import torch


# In[4]:
class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
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

def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    
    h, w = spatial_size 
    x = torch.arange(w).type(type) #torch.Size([58])
    y = torch.arange(h).type(type) #torch.Size([58])

    x = (2 * (x / (w - 1)) - 1) #map the tensor of numbers with range 0 -> 57 to numbers the range -1 -> 1
    y = (2 * (y / (h - 1)) - 1) #map the tensor of numbers with range 0 -> 57 to numbers the range -1 -> 1

    yy = y.view(-1, 1).repeat(1, w) # torch.Size([58, 1]).repeat(1, w) = torch.Size([58, 58])
    """
    tensor([[-1.0000, -1.0000, -1.0000,  ..., -1.0000, -1.0000, -1.0000],
        [-0.9649, -0.9649, -0.9649,  ..., -0.9649, -0.9649, -0.9649],
        [-0.9298, -0.9298, -0.9298,  ..., -0.9298, -0.9298, -0.9298],
        ...,
        [ 0.9298,  0.9298,  0.9298,  ...,  0.9298,  0.9298,  0.9298],
        [ 0.9649,  0.9649,  0.9649,  ...,  0.9649,  0.9649,  0.9649],
        [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000]])
    """
    xx = x.view(1, -1).repeat(h, 1) # torch.Size([1, 58]).repeat(h, 1) = torch.Size([58, 58])
    """
    tensor([[-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000],
        [-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000],
        [-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000],
        ...,
        [-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000],
        [-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000],
        [-1.0000, -0.9649, -0.9298,  ...,  0.9298,  0.9649,  1.0000]])
    """

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2) #torch.Size([58, 58, 2])
    """
    
    [xx.unsqueeze_(2), yy.unsqueeze_(2)] = # [torch.Size([58, 58, 1]), torch.Size([58, 58, 1]) ]
    [tensor([[[-1.0000],
         [-0.9649],
         [-0.9298],
         ...,
         [ 0.9298],
         [ 0.9649],
         [ 1.0000]],

        [[-1.0000],
         [-0.9649],
         [-0.9298],
         ...,
         [ 0.9298],
         [ 0.9649],
         [ 1.0000]],

        ...,

        [[-1.0000],
         [-0.9649],
         [-0.9298],
         ...,
         [ 0.9298],
         [ 0.9649],
         [ 1.0000]],

        [[-1.0000],
         [-0.9649],
         [-0.9298],
         ...,
         [ 0.9298],
         [ 0.9649],
         [ 1.0000]]]), 
         tensor([[[-1.0000],
         [-1.0000],
         [-1.0000],
         ...,
         [-1.0000],
         [-1.0000],
         [-1.0000]],

        [[-0.9649],
         [-0.9649],
         [-0.9649],
         ...,
         [-0.9649],
         [-0.9649],
         [-0.9649]],

        ...,


        [[ 0.9649],
         [ 0.9649],
         [ 0.9649],
         ...,
         [ 0.9649],
         [ 0.9649],
         [ 0.9649]],

        [[ 1.0000],
         [ 1.0000],
         [ 1.0000],
         ...,
         [ 1.0000],
         [ 1.0000],
         [ 1.0000]]])]

    torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2) = 
    
    tensor([[[-1.0000, -1.0000],
         [-0.9649, -1.0000],
         [-0.9298, -1.0000],
         ...,
         [ 0.9298, -1.0000],
         [ 0.9649, -1.0000],
         [ 1.0000, -1.0000]],

        ...,
  
        [[-1.0000,  1.0000],
         [-0.9649,  1.0000],
         [-0.9298,  1.0000],
         ...,
         [ 0.9298,  1.0000],
         [ 0.9649,  1.0000],
         [ 1.0000,  1.0000]]])
    """

    return meshed
    
    
    
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    print("kp2gaussian")
    mean = kp['value']
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) #torch.Size([64, 64, 2])
    coordinate_grid = coordinate_grid.view((1,1)+coordinate_grid.shape) 
    #print("4",coordinate_grid.shape) #torch.Size([1, 1, 64, 64, 2])
    Trd = coordinate_grid.repeat((1,10,1,1,1))
    #print("6",Trd.shape) #torch.Size([1, 10, 64, 64, 2])
    z = mean.unsqueeze(2).unsqueeze(2)
    #print("8",mean.shape) #torch.Size([1, 10, 1, 1, 2])
    out = torch.exp(-0.5 * ((Trd - z) ** 2).sum(-1) / kp_variance)
    #print("10",out.shape) #torch.Size([1, 10, 64, 64])
    

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


class Hourglassnew(nn.Module):
  
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

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


