from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d


class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            ## Initialize the weights/bias with identity transformation
            self.jacobian.weight.data.zero_()
            
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        print("start")
        shape = heatmap.shape #torch.Size([1, 10, 58, 58])
        print(shape)
        heatmap = heatmap.unsqueeze(-1) # torch.Size([1, 10, 58, 58, 1])
        print(heatmap.shape)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0) # torch.Size([58, 58, 2]).unsqueeze_(0).unsqueeze_(0)      = torch.Size([1, 1, 58, 58, 2]) 
        print("hey")
        print(grid.shape, heatmap.shape)
        print((heatmap * grid).shape)
        print("hey again")
        print(heatmap * grid)
      
        value = (heatmap * grid).sum(dim=(2, 3)) # torch.Size([1, 10, 58, 58, 1]) * torch.Size([1, 1, 58, 58, 2]) = torch.Size([1, 10, 58, 58, 2])..sum(dim=(2, 3)) = torch.Size([1, 10, 2])
        print(value.shape)
        kp = {'value': value}

        return kp

    def forward(self, x): #torch.Size([1, 3, 256, 256]
        if self.scale_factor != 1:
            print("before",x.shape) 
            x = self.down(x)  #torch.Size([1, 3, 64, 64])
            print("after",x.shape)

        feature_map = self.predictor(x) # torch.Size([1, 35, 64, 64])
        print("feature_map", feature_map.shape)
        prediction = self.kp(feature_map) # torch.Size([1, 10, 58, 58])
        print("prediction",prediction.shape)

        final_shape = prediction.shape # torch.Size([1, 10, 58, 58])
        
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)  #torch.Size([1, 10, 3364])
        print(heatmap.shape)
        
        heatmap = F.softmax(heatmap / self.temperature, dim=2) #Softmax(xi​)=exp(xi​)​ / ∑j​ exp(xj​)
        #It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
        heatmap = heatmap.view(*final_shape) #torch.Size([1, 10, 58, 58])
        print(heatmap.shape) #torch.Size([1, 10, 2])
        out = self.gaussian2kp(heatmap) 

        if self.jacobian is not None:
            print(feature_map.shape) #torch.Size([1, 35, 64, 64])
            jacobian_map = self.jacobian(feature_map) #torch.Size([1, 40, 58, 58])
            print("1",(jacobian_map.shape))
            print("jsize",jacobian_map.size())
            jacobian_map = jacobian_map.view(final_shape[0], self.num_jacobian_maps, 4, final_shape[2], final_shape[3])# torch.Size([1, 10, 4, 58, 58])
            print("2",jacobian_map.shape)
            heatmap = heatmap.unsqueeze(2) #torch.Size([1, 10, 1, 58, 58])
            print("3",heatmap.shape)

            jacobian = heatmap * jacobian_map #torch.Size([1, 10, 4, 58, 58])
            print("4",jacobian.shape)
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1)#torch.Size([1, 10, 4, 3364])
            print("5",jacobian.shape)
            jacobian = jacobian.sum(dim=-1) #torch.Size([1, 10, 4])
            print("6",jacobian.shape)
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)#torch.Size([1, 10, 2, 2])
            print("7",jacobian.shape)
            out['jacobian'] = jacobian

        return out
