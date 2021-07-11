from torch import nn
import torch
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d



class encoderDecoder(nn.Module):
  
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(encoderDecoder, self).__init__()
        self.DB1 = DownBlock2d(in_features ,64, kernel_size=3, padding=1)
        self.DB2 = DownBlock2d(64 ,128, kernel_size=3, padding=1)
        self.DB3 = DownBlock2d(128 ,256, kernel_size=3, padding=1)
        self.DB4 = DownBlock2d(256 ,512, kernel_size=3, padding=1)
        self.DB5 = DownBlock2d(512 ,1024, kernel_size=3, padding=1)
        
        self.UB1 = UpBlock2d(1024 ,512, kernel_size=3, padding=1)
        self.UB2 = UpBlock2d(1024 ,256, kernel_size=3, padding=1)
        self.UB3 = UpBlock2d(512 ,128, kernel_size=3, padding=1)
        self.UB4 = UpBlock2d(256 ,64, kernel_size=3, padding=1)
        self.UB5 = UpBlock2d(128 ,32, kernel_size=3, padding=1)
        
        self.out_filters = block_expansion + in_features
     

    def forward(self, x):
 
        out1 =  self.DB1 (x)
        out2 =  self.DB2 (out1)
        out3 =  self.DB3 (out2)
        out4 =  self.DB4 (out3)
        out5 =  self.DB5 (out4)
        print("new", out5.shape)
        out =  self.UB1 (out5)    
        print("new", out.shape, out4.shape)
        out =  self.UB2 (torch.cat([out, out4], dim=1))
        print("new", out.shape, out3.shape)
        out =  self.UB3 (torch.cat([out, out3], dim=1))
        print("new", out.shape, out2.shape)
        out =  self.UB4 (torch.cat([out, out2], dim=1))
        print("new", out.shape, out1.shape)
        out =  self.UB5 (torch.cat([out, out1], dim=1))
        return  torch.cat([out, x], dim=1)
    
class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.featureMapExtractor = encoderDecoder(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.heatMapsExtractor = nn.Conv2d(in_channels=self.featureMapExtractor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.featureMapExtractor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None
            
        self.temperature = temperature
        self.scale_factor = scale_factor
        
        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    

    def forward(self, x): #torch.Size([1, 3, 256, 256]
        
        x = self.down(x)  #torch.Size([1, 3, 64, 64])
        feature_map = self.featureMapExtractor(x) # torch.Size([1, 35, 64, 64])
        heatmaps = self.heatMapsExtractor(feature_map) # torch.Size([1, 10, 58, 58])

        final_shape = heatmaps.shape # torch.Size([1, 10, 58, 58])
        print("final_shape", final_shape)
        heatmap = heatmaps.view(final_shape[0], final_shape[1], -1)  #torch.Size([1, 10, 3364])

        print("heatmap", heatmap.shape)
        heatmap = F.softmax(heatmap / self.temperature, dim=2) #Softmax(xi​)=exp(xi​)​ / ∑j​ exp(xj​)
        #It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
        print("heatmap", heatmap.shape)
        heatmap0 = heatmap.view(*final_shape) #torch.Size([1, 10, 58, 58])
        h, w, = final_shape[2],final_shape[3] #torch.Size([1, 10, 58, 58])
        heatmap = heatmap0.unsqueeze(-1) # torch.Size([1, 10, 58, 58, 1])
        print("heatmap", heatmap.shape)
        
        x = torch.arange(w).type(heatmap.type()) #torch.Size([58])
        y = torch.arange(h).type(heatmap.type()) #torch.Size([58])
        print("X", x.shape)
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
        print("meshed", meshed.shape)
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
        
        
        unsqueezed_meshed = meshed.unsqueeze_(0).unsqueeze_(0)# torch.Size([58, 58, 2]).unsqueeze_(0).unsqueeze_(0)  = torch.Size([1, 1, 58, 58, 2]) 
        print("LOOK")
        print(heatmap.shape, unsqueezed_meshed.shape)
        value = (heatmap * unsqueezed_meshed).sum(dim=(2, 3)) # torch.Size([1, 10, 58, 58, 1]) * torch.Size([1, 1, 58, 58, 2]) = torch.Size([1, 10, 58, 58, 2])..sum(dim=(2, 3)) = torch.Size([1, 10, 2])
        out = {'value': value} 
        
        print(feature_map.shape) #torch.Size([1, 35, 64, 64])
        jac_feature_map = self.jacobian(feature_map) #torch.Size([1, 40, 58, 58])
        print("1",jac_feature_map.shape)
        jac_feature_map = jac_feature_map.view(final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3])# torch.Size([1, 10, 4, 58, 58])
        print("2",jac_feature_map.shape)
        heatmap = heatmap0.unsqueeze(2) #torch.Size([1, 10, 1, 58, 58])
        print("3",heatmap.shape)

        jacobian = heatmap * jac_feature_map #torch.Size([1, 10, 4, 58, 58])
        print("4",jacobian.shape)
        jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)#torch.Size([1, 10, 4, 3364])
        print("5",jacobian.shape)
        jacobian = jacobian.sum(dim=-1) #torch.Size([1, 10, 4])
        print("6",jacobian.shape)
        jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)#torch.Size([1, 10, 2, 2])
        print("7",jacobian.shape)
        out['jacobian'] = jacobian
        return out

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

    
class AntiAliasInterpolation2d(nn.Module):
    
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