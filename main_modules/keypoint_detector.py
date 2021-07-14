from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, create_mesh, AntiAliasInterpolation2d


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

        self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
        self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                    out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
        # Initialize the weights/bias with identity transformation
        self.jacobian.weight.data.zero_()
        self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        self.temperature = temperature
        self.scale_factor = scale_factor
        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

   
        
    def heatmap2keypoints(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        #print("start")
        shape = heatmap.shape #torch.Size([1, 10, 58, 58])
        #print(shape)
        heatmap = heatmap.unsqueeze(-1) # torch.Size([1, 10, 58, 58, 1])
        #print(heatmap.shape)
        mesh =  create_mesh(shape[3], shape[2],  heatmap.type())  # torch.Size([58, 58, 2]).unsqueeze_(0).unsqueeze_(0)      = torch.Size([1, 1, 58, 58, 2]) 
        #print("hey")
        #print(mesh.shape, heatmap.shape)
        #print((heatmap * mesh).shape)
        #print("hey again")
        #print(heatmap * mesh)
      
        value = (heatmap * mesh).sum(dim=(2, 3)) # torch.Size([1, 10, 58, 58, 1]) * torch.Size([1, 1, 58, 58, 2]) = torch.Size([1, 10, 58, 58, 2])..sum(dim=(2, 3)) = torch.Size([1, 10, 2])
        #print(value.shape)
        return value

    def heatmapFormation(self, feature_map):
        prediction = self.kp(feature_map) # torch.Size([1, 10, 58, 58])
        #print("prediction",prediction.shape)

        final_shape = prediction.shape # torch.Size([1, 10, 58, 58])
        
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)  #torch.Size([1, 10, 3364])
        #print(heatmap.shape)
        
        heatmap = F.softmax(heatmap / self.temperature, dim=2) #Softmax(xi​)=exp(xi​)​ / ∑j​ exp(xj​)
        #It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
        heatmap = heatmap.view(*final_shape) #torch.Size([1, 10, 58, 58])
        #It is applied to all slices along dim, and will re-scale them so that the elements lie in the range [0, 1] and sum to 1.
        return heatmap, final_shape


    def jakobianFormation(self, feature_map, shape, heatmap ):
        jacobian_map = self.jacobian(feature_map) #torch.Size([1, 40, 58, 58])
        #print("1",(jacobian_map.shape))
        #print("jsize",jacobian_map.size())
        jacobian_map = jacobian_map.view(shape[0], self.num_jacobian_maps, 4, shape[2], shape[3])# torch.Size([1, 10, 4, 58, 58])
        #print("2",jacobian_map.shape)
        heatmap = heatmap.unsqueeze(2) #torch.Size([1, 10, 1, 58, 58])
        

        jacobian = heatmap * jacobian_map #torch.Size([1, 10, 4, 58, 58])
        #print("4",jacobian.shape)
        jacobian = jacobian.view(shape[0], shape[1], 4, -1)#torch.Size([1, 10, 4, 3364])
        #print("5",jacobian.shape)
        jacobian = jacobian.sum(dim=-1) #torch.Size([1, 10, 4])
        #print("6",jacobian.shape)
        jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)#torch.Size([1, 10, 2, 2])
        #print("7",jacobian.shape)
        return jacobian

    def forward(self, x): #torch.Size([1, 3, 256, 256]
       
        #print("before",x.shape) 
        x = self.down(x)  #torch.Size([1, 3, 64, 64])
        #print("after",x.shape)

        feature_map = self.predictor(x) # torch.Size([1, 35, 64, 64])
        heatmap,shape = self.heatmapFormation(feature_map)
        #print(heatmap.shape)
        #print(heatmap.shape) #torch.Size([1, 10, 2])
        keypoints = self.heatmap2keypoints(heatmap) 
        jacobian = self.jakobianFormation( feature_map, shape, heatmap)
        kp = {}
        kp['value'] = keypoints
        kp['jacobian'] = jacobian
        return kp
