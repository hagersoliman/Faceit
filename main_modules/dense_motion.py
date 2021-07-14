from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, AntiAliasInterpolation2d, localized_heatmap, create_mesh


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
        self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))    
        self.num_kp = num_kp
        self.scale_factor = scale_factor
        self.kp_variance = kp_variance
        self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    
    def create_heatmap(self, kp_driving, kp_source, w, h):
        """
        Eq 6. in the paper 
        """
        #print("eq6")
        bG = torch.zeros(1, 1, w, h).type(kp_driving['value'].type())
        driving_heatmap = localized_heatmap(kp_driving, self.kp_variance, w, h) #torch.Size([1, 10, 64, 64])
        source_heatmap = localized_heatmap(kp_source, self.kp_variance, w, h) #torch.Size([1, 10, 64, 64])
        heatmap = driving_heatmap - source_heatmap
        heatmap = torch.cat([bG, heatmap], dim=1).unsqueeze(2)
        return heatmap

    def z_tdr(self, mesh, kp_driving, h, w, bs):
        z = mesh.view(1, 1, h, w, 2)
        tdr = kp_driving['value'].view(bs, self.num_kp, 1, 1, 2)
        return z - tdr
    
    def jk(self, kp_source, kp_driving, h, w):
        #print("hey",kp_driving['jacobian'].shape)
        #print("heyy",kp_source['jacobian'].shape)
        jacobian = torch.matmul(kp_source['jacobian'], torch.inverse(kp_driving['jacobian']))
        jacobian = jacobian.unsqueeze(-3).unsqueeze(-3)
        jacobian = jacobian.repeat(1, 1, h, w, 1, 1)
        return jacobian
    
    def jkDotZ_tdr( self, mesh, jacobian):
        mesh = torch.matmul(jacobian, mesh.unsqueeze(-1)) # jk * (Z - TDR)
        mesh = mesh.squeeze(-1)
        return mesh
             
    
        
    def create_local_motions(self, source_image, kp_driving, kp_source):
        """
        Eq 4. in the paper T_{s<-d}(z)
        """
        #print("Eq 4")
        bs, _, h, w = source_image.shape
        ref_mesh = create_mesh(w, h, kp_source['value'].type()) #torch.Size([64, 64, 2])
        #print("1",ref_mesh.shape) 
        z_tdr =  self.z_tdr(ref_mesh, kp_driving,  h, w, bs)
        #print(z_tdr.shape)  #torch.Size([1, 10, 64, 64, 2])
        
        jacobian = self.jk(kp_source, kp_driving,  h, w)  # jk
        #print("6",jacobian.shape) #torch.Size([1, 10, 64, 64, 2, 2])
        dot_out = self.jkDotZ_tdr(z_tdr, jacobian)
        #print("8",dot_out.shape) #torch.Size([1, 10, 64, 64, 2])
        
        tsr = kp_source['value'].view(bs, self.num_kp, 1, 1, 2)
        driving_to_source = dot_out + tsr #TSR + jk * (Z - TDR)

        #print("9",driving_to_source.shape) #torch.Size([1, 10, 64, 64, 2])
        #adding background feature
        ref_mesh = ref_mesh.view(1, 1, h, w, 2).repeat(bs, 1, 1, 1, 1)
        #print("10",ref_mesh.shape) #torch.Size([1, 1, 64, 64, 2])
        sparse_motions = torch.cat([ref_mesh, driving_to_source], dim=1)
        #print("11",sparse_motions.shape) #torch.Size([1, 11, 64, 64, 2])
        return sparse_motions

    def create_wrapped_source_image(self, source_image, sparse_motions):
        """
        Eq 7. in the paper \hat{T}_{s<-d}(z)
        """
        #print("eq7")
        #print("0", source_image.shape) #torch.Size([1, 3, 64, 64])
        
        bs, _, h, w = source_image.shape
        unsqueezed_src = source_image.unsqueeze(1).unsqueeze(1) #torch.Size([1, 1, 1, 3, 64, 64])
        repeated_src = unsqueezed_src.repeat(1, self.num_kp + 1, 1, 1, 1, 1)
        #print("1", repeated_src.shape) #torch.Size([1, 11, 1, 3, 64, 64])
        repeated_src = repeated_src.view(bs * (self.num_kp + 1), -1, h, w)
        #print("2", repeated_src.shape) #torch.Size([11, 3, 64, 64])
        sparse_motions = sparse_motions.view((bs * (self.num_kp + 1), h, w, -1))
        #print("3",sparse_motions.shape) #torch.Size([11, 64, 64, 2])
        sparse_deformed = F.grid_sample(repeated_src, sparse_motions)
        #print("4", sparse_deformed.shape) #torch.Size([11, 3, 64, 64])
        sparse_deformed = sparse_deformed.view((bs, self.num_kp + 1, -1, h, w))
        #print("5",sparse_deformed.shape) #torch.Size([1, 11, 3, 64, 64])
        return sparse_deformed
    
   
    
    def get_masked_sparse_motion(self, mask, sparse_motion):
        mask = mask.unsqueeze(2)
        #print("5", mask.shape)#torch.Size([1, 11, 1, 64, 64])
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        #print("6", sparse_motion.shape) #torch.Size([1, 11, 2, 64, 64])
        deformation = (sparse_motion * mask).sum(dim=1)
        #print("7", deformation.shape)#torch.Size([1, 2, 64, 64])
        deformation = deformation.permute(0, 2, 3, 1)
        #print("8", deformation.shape) #torch.Size([1, 64, 64, 2])
        return deformation
    
    
    def get_mask(self, heatmap, wraped_source,bs,  h, w):
        input = torch.cat([heatmap, wraped_source], dim=2)
        #print("0", input.shape) # torch.Size([1, 11, 4, 64, 64])
        input = input.view(bs, -1, h, w)
        #print("1", input.shape) # torch.Size([1, 44, 64, 64])
        prediction = self.hourglass(input)
        #print("2", prediction.shape) #torch.Size([1, 108, 64, 64])
        mask = self.mask(prediction)
        #print("3", mask.shape) #torch.Size([1, 11, 64, 64])
        mask = F.softmax(mask, dim=1)
        return mask, prediction
    
    def get_occlusion_map(self, prediction):
        occlusion_map = torch.sigmoid(self.occlusion(prediction))
        return occlusion_map

        
    def forward(self, source_image, kp_driving, kp_source):
        #print("forward")
        if self.scale_factor != 1:
            source_image = self.down(source_image)
            
        
        bs, _, h, w = source_image.shape
        #print(bs,h,w)

        out_dict = dict()
        heatmap = self.create_heatmap( kp_driving, kp_source,  w, h)
        sparse_motion = self.create_local_motions(source_image, kp_driving, kp_source)
        wrapped_source = self.create_wrapped_source_image(source_image, sparse_motion)
        out_dict['sparse_deformed'] = wrapped_source # the source image moved due to the  
        mask , prediction = self.get_mask(heatmap, wrapped_source, bs,  h, w)
        #print("4", mask.shape)#torch.Size([1, 11, 64, 64])
        out_dict['mask'] = mask 
        deformation = self.get_masked_sparse_motion( mask, sparse_motion)
        out_dict['deformation'] = deformation
        occlusion_map = self.get_occlusion_map( prediction)
        out_dict['occlusion_map'] = occlusion_map

        return out_dict



