from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, make_coordinate_grid, kp2gaussian


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, num_channels, estimate_occlusion_map=False,
                 scale_factor=1, kp_variance=0.01):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp + 1) * (num_channels + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_kp + 1, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    
    def create_heatmap(self, source_image, kp_driving, kp_source):
        """
        Eq 6. in the paper 
        """
        print("eq6")
        spatial_size = source_image.shape[2:]
        print(spatial_size)
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=source_image.shape[2:], kp_variance=self.kp_variance) 
        print(gaussian_driving.shape)#torch.Size([1, 10, 64, 64])
        gaussian_source = kp2gaussian(kp_source, spatial_size=source_image.shape[2:], kp_variance=self.kp_variance)
        print(gaussian_source.shape) #torch.Size([1, 10, 64, 64])
        heatmap = gaussian_driving - gaussian_source
        print(heatmap.shape)

        #adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, heatmap.shape[2], heatmap.shape[3]).type(heatmap.type())
        print("0",zeros.shape) #torch.Size([1, 1, 64, 64])
        heatmap = torch.cat([zeros, heatmap], dim=1)
        print("1",heatmap.shape) #torch.Size([1, 11, 64, 64])
        heatmap = heatmap.unsqueeze(2)
        print("2",heatmap.shape) #torch.Size([1, 11, 1, 64, 64])
        return heatmap


    def z_tdr(self, identity_grid, kp_driving, h, w, bs):
        z = identity_grid.view(1, 1, h, w, 2)
        tdr = kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        return z - tdr
    
    def jk(self, kp_source, kp_driving, h, w):
        jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
        jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
        return jacobian
    
    def jkDotZ_tdr(self, coordinate_grid, jacobian):
        coordinate_grid = torch.matmul(jacobian, coordinate_grid.unsqueeze(-1)) # jk * (Z - TDR)
        coordinate_grid = coordinate_grid.squeeze(-1)
        return coordinate_grid
             
    
        
    def create_local_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        print("Eq 4")
        bs, _, h, w = source_image.shape
        identity_grid = make_coordinate_grid((h, w), type=kp_source['value'].type())
        print("1",identity_grid.shape) #torch.Size([64, 64, 2])
        coordinate_grid =  self.z_tdr(identity_grid, kp_driving,  h, w, bs)
        print(coordinate_grid.shape)  #torch.Size([1, 10, 64, 64, 2])
        if 'jacobian' in kp_driving:
            
            jacobian = self.jk(kp_source, kp_driving,  h, w)  # jk
            print("6",jacobian.shape) #torch.Size([1, 10, 64, 64, 2, 2])
            coordinate_grid = self.jkDotZ_tdr(coordinate_grid, jacobian)
            print("8",coordinate_grid.shape) #torch.Size([1, 10, 64, 64, 2])
        
        tsr = kp_source['value'].view(bs, self.num_kp, 1, 1, 2)
        driving_to_source = coordinate_grid + tsr #TSR + jk * (Z - TDR)

        print("9",driving_to_source.shape) #torch.Size([1, 10, 64, 64, 2])
        #adding background feature
        identity_grid = identity_grid.view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        print("10",identity_grid.shape) #torch.Size([1, 1, 64, 64, 2])
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)
        print("11",sparse_motions.shape) #torch.Size([1, 11, 64, 64, 2])
        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        print("eq7")
        print("0", source_image.shape) #torch.Size([1, 3, 64, 64])
        
        bs, _, h, w = source_image.shape
        unsqueezed_src = source_image.unsqueeze(1).unsqueeze(1) #torch.Size([1, 1, 1, 3, 64, 64])
        repeated_src = unsqueezed_src.repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        print("1", repeated_src.shape) #torch.Size([1, 11, 1, 3, 64, 64])
        repeated_src = repeated_src.view(bs * (self.num_kp + 1), -1, h, w)
        print("2", repeated_src.shape) #torch.Size([11, 3, 64, 64])
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        print("3",sparse_motions.shape) #torch.Size([11, 64, 64, 2])
        sparse_deformed = F.grid_sample(repeated_src, sparse_motions)
        print("4", sparse_deformed.shape) #torch.Size([11, 3, 64, 64])
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        print("5",sparse_deformed.shape) #torch.Size([1, 11, 3, 64, 64])
        return sparse_deformed
    
   
    
    def get_masked_sparse_motion(self, mask, sparse_motion):
        mask = mask.unsqueeze(2)
        print("5", mask.shape)#torch.Size([1, 11, 1, 64, 64])
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        print("6", sparse_motion.shape) #torch.Size([1, 11, 2, 64, 64])
        deformation = (sparse_motion * mask).sum(dim=1)
        print("7", deformation.shape)#torch.Size([1, 2, 64, 64])
        deformation = deformation.permute(0, 2, 3, 1)
        print("8", deformation.shape) #torch.Size([1, 64, 64, 2])
        return deformation
    
    
    def get_mask(self, heatmap_representation, deformed_source,bs,  h, w):
        input = torch.cat([heatmap_representation, deformed_source], dim=2)
        print("0", input.shape) # torch.Size([1, 11, 4, 64, 64])
        input = input.view(bs, -1, h, w)
        print("1", input.shape) # torch.Size([1, 44, 64, 64])
        prediction = self.hourglass(input)
        print("2", prediction.shape) #torch.Size([1, 108, 64, 64])
        mask = self.mask(prediction)
        print("3", mask.shape) #torch.Size([1, 11, 64, 64])
        mask = F.softmax(mask, dim=1)
        return mask, prediction
    
    def get_occlusion_map(self, prediction):
        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        return occlusion_map

        
    def forward(self, source_image, kp_driving, kp_source):
        print("forward")
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            
        
        bs, _, h, w = source_image.shape
        print(bs,h,w)

        out_dict = dict()
        heatmap_representation = self.create_heatmap(source_image, kp_driving, kp_source)
        sparse_motion = self.create_local_motions(source_image, kp_driving, kp_source)
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = deformed_source # the source image moved due to the 
        
        mask , prediction = self.get_mask(heatmap_representation, deformed_source, bs,  h, w)
        print("4", mask.shape)#torch.Size([1, 11, 64, 64])
        out_dict['mask'] = mask 
        deformation = self.get_masked_sparse_motion( mask, sparse_motion)
       
        out_dict['deformation'] = deformation

       
        if self.occlusion:
            occlusion_map = self.get_occlusion_map( prediction)
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
    