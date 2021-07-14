from torch import nn
import torch
import torch.nn.functional as F

from torchvision import models
import numpy as np
from torch.autograd import grad
from torchvision.models import vgg19
import math


def make_coordinate_grid(spatial_size, type):
    
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)
    
    x = (2 * (x / (w - 1)) - 1)
    
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)
   
    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    
    return meshed

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



class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        features = vgg19(pretrained=True).features

        layers = [(0, 2), (2, 7), (7, 12), (12, 21), (21, 30)]

        # self.slices =  [nn.Sequential(),
        #                 nn.Sequential(),
        #                 nn.Sequential(),
        #                 nn.Sequential(),
        #                 nn.Sequential()]

        self.slices = []
        self.slice1 = nn.Sequential()
        self.slices.append(self.slice1)
        self.slice2 = nn.Sequential()
        self.slices.append(self.slice2)
        self.slice3 = nn.Sequential()
        self.slices.append(self.slice3)
        self.slice4 = nn.Sequential()
        self.slices.append(self.slice4)
        self.slice5 = nn.Sequential()
        self.slices.append(self.slice5)
        
        for i in range(len(layers)):
          for j in range(*layers[i]):
            self.slices[i].add_module(str(j), features[j])

        # for i in range(len(layers)):
        # 	self.slices.append(nn.Sequential(*features[layers[i][0]: layers[i][1]]))
        
        self.mean = nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))), requires_grad=False)
        self.std = nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))), requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = []
        X = (X - self.mean) / self.std
        next_input = X
        for i in range(len(self.slices)):
            next_input = self.slices[i](next_input)
            out.append(next_input)

        # print("out is ..........")
        # print(out)
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type()).unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
            # print("control things:................")
            # print(self.control_points.size())
            # print(self.control_params.size())
        else:
            self.tps = False

    def transform_frame(self, frame):
        # print("frame:............")
        # print(frame.size())
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        #warp
        theta = (self.theta.type(grid.type())).unsqueeze(1)
        transformed = (torch.matmul(theta[:, :, :, :2], grid.unsqueeze(-1)) + theta[:, :, :, 2:]).squeeze(-1)
        if self.tps:
            control_points = self.control_points.type(grid.type())
            control_params = self.control_params.type(grid.type())
            distances = torch.abs(grid.view(grid.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)).sum(-1)

            result = (distances ** 2) * torch.log(distances + 1e-6) * control_params
            transformed += result.sum(dim=2).view(self.bs, grid.shape[1], 1)
        grid = transformed.view(self.bs, frame.shape[2], frame.shape[3], 2)
        # print("grid:............")
        # print(grid.size())
        # print(F.grid_sample(frame, grid, padding_mode="reflection").size())
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def jacobian(self, coordinates):
        theta = (self.theta.type(coordinates.type())).unsqueeze(1)
        transformed = (torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]).squeeze(-1)
        # print("transformed:.............")
        # print(transformed.size())
        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = torch.abs(coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)).sum(-1)

            result = (distances ** 2) * torch.log(distances + 1e-6) * control_params
            transformed += result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)

        new_coordinates = transformed

        grad_x = grad(new_coordinates[..., 0].sum(), coordinates, create_graph=True)
        grad_y = grad(new_coordinates[..., 1].sum(), coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.scales = train_params['scales']
        self.disc_scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        origin_pyramid = self.pyramid(x['driving'])
        generated_pyramid = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            total_value = 0
            for scale in self.scales:
                wanted_key = 'prediction_' + str(scale)
                x_vgg = self.vgg(generated_pyramid[wanted_key])
                y_vgg = self.vgg(origin_pyramid[wanted_key])

                for i in range(len(self.loss_weights['perceptual'])):
                    weight = self.loss_weights['perceptual'][i]
                    val_abs = torch.abs(x_vgg[i] - y_vgg[i].detach())
                    value = val_abs.mean()
                    total_value += weight * value
                loss_values['perceptual'] = total_value

        if self.loss_weights['generator_gan'] != 0:
            key_points = {key: value.detach() for key, value in kp_driving.items()}
            generated_discriminator_maps = self.discriminator(generated_pyramid, kp=key_points)
            origin_discriminator_maps = self.discriminator(origin_pyramid, kp=key_points)
            total_value = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                calc_origin_disc = (1 - generated_discriminator_maps[key]) ** 2
                value = calc_origin_disc.mean()
                total_value += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = total_value

            if sum(self.loss_weights['feature_matching']) != 0:
                total_value = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(origin_discriminator_maps[key], generated_discriminator_maps[key])):
                        weight = self.loss_weights['feature_matching'][i]
                        if weight == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        total_value += weight * value
                    loss_values['feature_matching'] = total_value

        equivariance_value = self.loss_weights['equivariance_value']
        if (equivariance_value + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_frame = transform.transform_frame(x['driving'])
            # print("transformed...............", transformed_frame.size())
            transformed_kp = self.kp_extractor(transformed_frame)

            generated['transformed_frame'] = transformed_frame
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if equivariance_value != 0:
                #wrap
                theta = (transform.theta.type(transformed_kp['value'].type())).unsqueeze(1)
                transformed = (torch.matmul(theta[:, :, :, :2], transformed_kp['value'].unsqueeze(-1)) + theta[:, :, :, 2:]).squeeze(-1)
                if transform.tps:
                    control_points = transform.control_points.type(transformed_kp['value'].type())
                    control_params = transform.control_params.type(transformed_kp['value'].type())
                    distances = torch.abs(transformed_kp['value'].view(transformed_kp['value'].shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)).sum(-1)

                    result = (distances ** 2) * torch.log(distances + 1e-6) * control_params
                    transformed += result.sum(dim=2).view(transform.bs, transformed_kp['value'].shape[1], 1)

                value = torch.abs(kp_driving['value'] - transformed).mean()
                loss_values['equivariance_value'] = equivariance_value * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                
                value = torch.matmul(normed_driving, jacobian_transformed)

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.discriminator = discriminator
        self.train_params = train_params
        self.loss_weights = train_params['loss_weights']
        self.scales = self.discriminator.scales

        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()


    def forward(self, x, generated):
        origin_pyramid = self.pyramid(x['driving'])
        generated_pyramide = self.pyramid(generated['prediction'].detach())
        kp_driving = generated['kp_driving']

        key_points = {key: value.detach() for key, value in kp_driving.items()}
        origin_discriminator_maps = self.discriminator(origin_pyramid, kp=key_points)
        generated_discriminator_maps= self.discriminator(generated_pyramide, kp=key_points)

        total_loss = 0
        # print("i'm dead ---------------------------------------------------")
        for scale in self.scales:
            calc_origin_disc = (1 - origin_discriminator_maps['prediction_map_%s' % scale]) ** 2
            calc_generated_disc = generated_discriminator_maps['prediction_map_%s' % scale] ** 2
            value = calc_origin_disc + calc_generated_disc
            # print("vaaaaaaaaaaaaaaaaaaalue==========================================")
            # print(value.size())
            total_loss += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values = {}
        loss_values['disc_gan'] = total_loss

        return loss_values
