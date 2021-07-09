from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision.models import vgg19
import numpy as np
from torch.autograd import grad
from torchvision.models import vgg19

class Vgg19(nn.Module):
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


class ImagePyramide(nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        scale_down = {}
        for lvl in scales:
            scale_down[str(lvl).translate({ord('.'): None})] = AntiAliasInterpolation2d(num_channels, lvl)
        self.scale_down = nn.ModuleDict(scale_down)

    def forward(self, x):
        predicted = {}
        for lvl, down_module in self.scale_down.items():
            predicted['prediction_' + str(lvl).translate({ord('.'): None})] = down_module(x)
        # print("predicted :............")
        # print(predicted)
        return predicted


class Transform:
    """
    Random tps transformation for equivariance constraints. See Sec 3.3
    """
    def __init__(self, bs, **kwargs):
        # tensor of random numbers with mean = 0
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type()).unsqueeze(0)
            self.control_params = torch.normal(mean=0, std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0).view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type()).unsqueeze(1)
        transformed = (torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]).squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)
            result = (distances ** 2) * torch.log(distances + 1e-6) * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
        return transformed + result

    def jacobian(self, old_coordinates):
        coordinates = self.warp_coordinates(old_coordinates)
        grad_x = grad(coordinates[..., 0].sum(), old_coordinates, create_graph=True)
        grad_y = grad(coordinates[..., 1].sum(), old_coordinates, create_graph=True)
        jacobian = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        return jacobian


def detach_kp(kp):
    return {key: value.detach() for key, value in kp.items()}


class GeneratorFullModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, discriminator, kp_extractor, train_params): # order changed
        #super(GeneratorFullModel, self).__init__()
        self.generator, self.discriminator = generator, discriminator
        self.kp_extractor, self.train_params = kp_extractor, train_params

        """self.pyramid = ImagePyramide(train_params['scales'], generator.num_channels)
    		
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()"""
        
        self.pyramid = self.pyramid.cuda() if torch.cuda.is_available() else ImagePyramide(train_params['scales'], generator.num_channels)
    		

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19() if sum(self.loss_weights['perceptual']) != 0 else False
        if self.vgg and torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

    def forward(self, x):
        kp_source = self.kp_extractor(x['source'])
        kp_driving = self.kp_extractor(x['driving'])

        generated = self.generator(x['source'], kp_source=kp_source, kp_driving=kp_driving)
        generated.update({'kp_source': kp_source, 'kp_driving': kp_driving})

        loss_values = {}

        pyramide_real, pyramide_generated = self.pyramid(x['driving']), self.pyramid(generated['prediction'])

        if self.vgg:
            value_total = 0
            for scale in train_params['scales']:
                x_vgg, y_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)]), self.vgg(pyramide_real['prediction_' + str(scale)])
                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value_total += (weight * torch.abs(x_vgg[i] - y_vgg[i].detach()).mean())
                loss_values['perceptual'] = value_total

        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated, kp=detach_kp(kp_driving))
            discriminator_maps_real = self.discriminator(pyramide_real, kp=detach_kp(kp_driving))
            value_total = 0
            for scale in self.discriminator.scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.discriminator.scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        if (self.loss_weights['equivariance_value'] + self.loss_weights['equivariance_jacobian']) != 0:
            transform = Transform(x['driving'].shape[0], **self.train_params['transform_params'])
            transformed_kp = self.kp_extractor(transform.transform_frame(x['driving']))

            generated['transformed_frame'] = transform.transform_frame(x['driving'])
            generated['transformed_kp'] = transformed_kp

            ## Value loss part
            if self.loss_weights['equivariance_value'] != 0:
                value = torch.abs(kp_driving['value'] - transform.warp_coordinates(transformed_kp['value'])).mean()
                loss_values['equivariance_value'] = self.loss_weights['equivariance_value'] * value

            ## jacobian loss part
            if self.loss_weights['equivariance_jacobian'] != 0:
                """jacobian_transformed = torch.matmul(transform.jacobian(transformed_kp['value']), 
                																					transformed_kp['jacobian'])

                normed_driving = torch.inverse(kp_driving['jacobian'])
                normed_transformed = jacobian_transformed
                value = torch.matmul(normed_driving, normed_transformed)"""

                normed_driving = torch.inverse(kp_driving['jacobian'])
                value = torch.matmul(normed_driving, torch.matmul(transform.jacobian(transformed_kp['value']), transformed_kp['jacobian']))

                eye = torch.eye(2).view(1, 1, 2, 2).type(value.type())

                value = torch.abs(eye - value).mean()
                loss_values['equivariance_jacobian'] = self.loss_weights['equivariance_jacobian'] * value

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Merge all discriminator related updates into single model for better multi-gpu usage
    """

    def __init__(self, kp_extractor, generator, discriminator, train_params):
        #super(DiscriminatorFullModel, self).__init__()
        self.generator, self.discriminator = generator, discriminator
        self.kp_extractor, self.train_params = kp_extractor, train_params
        
        """self.pyramid = ImagePyramide(self.discriminator.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()"""
        
        self.pyramid = self.pyramid.cuda() if torch.cuda.is_available() else ImagePyramide(self.discriminator.scales, generator.num_channels)

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        kp_driving = generated['kp_driving']
        discriminator_maps_generated = self.discriminator(self.pyramid(generated['prediction'].detach()), kp=detach_kp(kp_driving))
        discriminator_maps_real = self.discriminator(self.pyramid(x['driving']), kp=detach_kp(kp_driving))

        loss_values = {}
        value_total = 0
        for scale in self.discriminator.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
      
      
