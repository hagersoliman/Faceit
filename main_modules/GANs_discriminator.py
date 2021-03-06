from torch import nn
import torch.nn.functional as F

import torch

def gaussian_heatmaps_from_keypoints(kp, size, kp_variance):
    mean = kp['value']
    shape_of_keypoints=len(mean.shape)
    leading_dim_num =  shape_of_keypoints - 1

    grid = generate_grid(size, mean.type())
    
    grid = grid.view((1,) * leading_dim_num + grid.shape)#1 1 256 256 2
    
    #5 10 1 1 1
    grid = grid.repeat(mean.shape[:leading_dim_num] + (1, 1, 1))#5 10 256 256 2
      
    mean = mean.view( mean.shape[:leading_dim_num] + (1, 1, 2))#5 10 1 1 2

    output = torch.exp(-0.5 * ( (grid - mean) ** 2).sum(-1) / kp_variance)
    #5 10 256 256 
    return output

#grid from -1:1 in x and y 
def generate_grid(size, type):
   
    h = size[0]
    w = size[1]
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    
    x = x / (w - 1)
    x = x*2 -1
    y = y / (h - 1)
    y = y*2 -1
    x_channel = x.view(1, -1).repeat(h, 1).unsqueeze_(2)
    y_channel = y.view(-1, 1).repeat(1, w).unsqueeze_(2)

    grid = torch.cat([x_channel, y_channel], 2)
    
    return grid

    
class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, sn=False):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        spectral_norm= nn.utils.spectral_norm(self.conv) if sn else None

        if spectral_norm is not None:
            self.conv = spectral_norm

        self.norm=nn.InstanceNorm2d(out_features, affine=True) if norm else None
       
        self.pool = pool

    def forward(self, x):
        output = self.conv(x)

        normed=self.norm(output) if self.norm else None
        if normed is not None:
            output = normed
        output = F.leaky_relu(output, 0.2)

        pooled=F.avg_pool2d(output, (2, 2)) if self.pool else None
        if pooled is not None:
            output = pooled
        return output


class Discriminator(nn.Module):

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=4, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(Discriminator, self).__init__()
        
        
        list_of_down_blocks=[DownBlock2d(num_channels + num_kp * use_kp if n == 0 else min(max_features, block_expansion * (2 ** n)),
                            min(max_features, block_expansion * (2 ** (n + 1))),
                            norm=(n != 0), kernel_size=4, pool=(n != num_blocks - 1), sn=sn) for n in range(num_blocks) ]

        self.down_blocks = nn.ModuleList(list_of_down_blocks)

        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)

        spectral_norm= nn.utils.spectral_norm(self.conv) if sn else None

        if spectral_norm is not None:
            self.conv = spectral_norm

        self.use_kp = use_kp
        self.kp_variance = kp_variance

    def forward(self, x, kp=None):
       
        output = x
        
        gaussian_heatmaps= gaussian_heatmaps_from_keypoints(kp, x.shape[2:], self.kp_variance) if self.use_kp else None
        if gaussian_heatmaps is not None:
            output = torch.cat([output, gaussian_heatmaps], dim=1)
           
        list_of_feature_maps = []
        i=0
        while i <len(self.down_blocks):
            list_of_feature_maps.append(self.down_blocks[i](output))
            output = list_of_feature_maps[-1]
            i+=1

        prediction_map = self.conv(output)
        
        return list_of_feature_maps, prediction_map


class MultiScaleDiscriminator(nn.Module):

    def __init__(self, scales=(), **kwargs):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.scales = scales
        discriminators_dictionary = {}
        i=0
        while i<len(scales):
            discriminators_dictionary[str(scales[i]).replace('.', '-')] = Discriminator(**kwargs)
            i+=1
        
        self.discs = nn.ModuleDict(discriminators_dictionary)

    def forward(self, x, kp=None):
       
        
        dictionary_of_output = {}
        for s in self.discs:
            
            modified_scale = str(s).replace('-', '.')    
            key = 'prediction_' + modified_scale
            list_of_feature_maps, prediction_map = self.discs[s](x[key], kp)
            
            dictionary_of_output['feature_maps_' + modified_scale] = list_of_feature_maps
            dictionary_of_output['prediction_map_' + modified_scale] = prediction_map
            
        return dictionary_of_output


