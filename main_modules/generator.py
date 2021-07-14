import torch
from torch import nn
import torch.nn.functional as F
from modules.dense_motion import DenseMotionNetwork
from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d


class DownBlock2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        out = self.pool(out)
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
        out = F.relu(self.norm(out))
        return out

class SameBlock2d(nn.Module):

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        return out


class ResBlock2d(nn.Module):
    

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = F.relu(self.norm1(x))
        out = self.conv1(out)
        out = F.relu(self.norm2(out))
        out = self.conv2(out)
        out =out + x
        return out

class OcclusionAwareGenerator(nn.Module):
    
    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

       
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))


        list_of_down_blocks=[DownBlock2d(min(max_features, block_expansion * (2 ** n)), min(max_features, block_expansion * (2 ** (n + 1))),
         kernel_size=(3, 3), padding=(1, 1)) for n in range(num_down_blocks) ]

        self.down_blocks = nn.ModuleList(list_of_down_blocks)



        self.dense_motion_network= None if dense_motion_params is None  else DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)


        self.estimate_occlusion_map = estimate_occlusion_map

        self.bottleneck = torch.nn.Sequential()
        n=0
        while  n < num_bottleneck_blocks:
            self.bottleneck.add_module('r' + str(n), ResBlock2d(min(max_features, block_expansion * (2 ** num_down_blocks)),
             kernel_size=(3, 3), padding=(1, 1)))
            n+=1

        list_of_up_blocks=[UpBlock2d(min(max_features, block_expansion * (2 ** (num_down_blocks - n))),
         min(max_features, block_expansion * (2 ** (num_down_blocks - n - 1))), kernel_size=(3, 3), padding=(1, 1)) for n in range(num_down_blocks) ]

        self.up_blocks = nn.ModuleList(list_of_up_blocks)

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
       
        self.num_channels = num_channels

    
    def forward(self, source_image, kp_driving, kp_source):
        
        #applying sameblock then downblocks and the out is 1*256*64*64
        output= self.first(source_image)

        n=0
        while n<len(self.down_blocks):
            output = self.down_blocks[n](output)
            n+=1

       
        # apply transformation and occlusion map
        dictionary_output = {}
        if self.dense_motion_network is not None:
            dictionary_of_dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)

            dictionary_output['mask'] = dictionary_of_dense_motion['mask']
            dictionary_output['sparse_deformed'] = dictionary_of_dense_motion['sparse_deformed']
            deformation = dictionary_of_dense_motion['deformation']

            dictionary_output["deformed"] = self.apply_deformation(source_image, deformation)

            output = self.apply_deformation(output, deformation)
           
            map_of_occlusion=dictionary_of_dense_motion['occlusion_map'] if 'occlusion_map' in dictionary_of_dense_motion else None

            if map_of_occlusion is not None:
                dictionary_output['occlusion_map'] = map_of_occlusion
                
                if output.shape[2] != map_of_occlusion.shape[2] or output.shape[3] != map_of_occlusion.shape[3]:
                    map_of_occlusion = F.interpolate(map_of_occlusion, size=output.shape[2:], mode='bilinear')
                
                output = output * map_of_occlusion
                
            
        # apply bottleneck and up block 
        output = self.bottleneck(output)
        
        n=0
        while n<len(self.up_blocks):
            output = self.up_blocks[n](output)
            n+=1
        
        output =F.sigmoid( self.final(output))
        
        dictionary_output["prediction"] = output

        return dictionary_output

    def apply_deformation(self, inpt, deformation):
        if deformation.shape[1] != inpt.shape[2] or deformation.shape[2] != inpt.shape[3]:#deform height and weidth and input height and weidth
    
            permuted_deformation = deformation.permute(0, 3, 1, 2)
            interpolated_deformation = F.interpolate(permuted_deformation, size=(inpt.shape[2], inpt.shape[3]), mode='bilinear')
            reversed_interpolated_deformation = interpolated_deformation.permute(0, 2, 3, 1)

        return F.grid_sample(inpt, reversed_interpolated_deformation if deformation.shape[1] != inpt.shape[2] or deformation.shape[2] != inpt.shape[3] else deformation)

