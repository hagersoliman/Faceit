import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull


def load_kp_detector_chkpt(config, checkpoint):
  kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
  kp_detector.cuda()
  kp_detector.load_state_dict(checkpoint['kp_detector'])
  kp_detector = DataParallelWithCallback(kp_detector)
  kp_detector.eval()
  return kp_detector

def load_generator_chkpt(config, checkpoint):
  generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
  generator.cuda()
  generator.load_state_dict(checkpoint['generator'])
  generator = DataParallelWithCallback(generator)
  generator.eval()
  return generator

def load_checkpoints(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.load(f)
    checkpoint = torch.load(checkpoint_path)

    generator = load_generator_chkpt(config, checkpoint)
    kp_detector = load_kp_detector_chkpt(config, checkpoint)

    return generator, kp_detector

def compute_relative_movement(kp_source, kp_driving,kp_driving_initial,kp_new, adapt_scale):
  adapt_movement_scale = 1
  if adapt_scale:
    source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
  
  kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
  kp_value_diff *= adapt_movement_scale
  value = kp_value_diff + kp_source['value']
  return value    

def compute_relative_jacobians(kp_source, kp_driving,kp_driving_initial,kp_new ):

  jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
  jacobians = torch.matmul(jacobian_diff, kp_source['jacobian'])
  return jacobians

def get_new_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,use_relative_movement=False, use_relative_jacobian=False):
    
    kp_new = {k: v for k, v in kp_driving.items()}
    kp_new['value'] = compute_relative_movement(kp_source, kp_driving,kp_driving_initial,kp_new, adapt_movement_scale)
    if use_relative_jacobian:
      kp_new['jacobian'] = compute_relative_jacobians(kp_source, kp_driving,kp_driving_initial,kp_new )
    return kp_new

def reconstruct_frame(source_image, kp_driving_initial, driving_image, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
  source_frame = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
  driving_frame = torch.tensor(driving_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) 

  source_frame = source_frame.cuda()
  driving_frame = driving_frame.cuda()

  kp_source = kp_detector(source_frame)
  kp_driving = kp_detector(driving_frame)

  kp_norm = get_new_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
  out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
  generated_frame = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
  return generated_frame


def vid2vid(src_path, video_path,new_vidoe_path, config_path,checkpoint_path, relative=True, adapt_movement_scale=True ):
  generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',checkpoint_path='vox-cpk.pth.tar')
  source_image = imageio.imread('black_women_src.jpg')
  reader = imageio.get_reader('black_women_crop.mp4')
  fps = reader.get_meta_data()['fps']
  driving_video = []
  for im in reader:
    driving_video.append(im)

  driving_initial = torch.tensor( resize(driving_video[0], (256, 256))[..., :3][np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)  

  kp_driving_initial = kp_detector(driving_initial)
  source_image = resize(source_image, (256, 256))[..., :3]

  new_video = []

  for im in reader:
    driving_image = resize(im, (256, 256))[..., :3]
    new_frame = reconstruct_frame(source_image, kp_driving_initial, driving_image, generator, kp_detector, relative=True, adapt_movement_scale=True)
    new_video.append(new_frame)
  imageio.mimsave(new_vidoe_path, [img_as_ubyte(frame) for frame in new_video], fps=fps)

  reader.close()
#vid2vid('black_women_src.jpg','black_women_crop.mp4','newVideo2.mp4','config/vox-256.yaml','vox-cpk.pth.tar',True, True)
