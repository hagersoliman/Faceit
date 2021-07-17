import os
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import glob
import numbers
import random
import PIL
from skimage.transform import resize, rotate
from skimage.util import pad
import torchvision
import warnings
from skimage import img_as_ubyte, img_as_float


class ColorAugmentation(object):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_factors(self, brightness, contrast, saturation, hue):
        
        bf = random.uniform(max(0, 1 - brightness), 1 + brightness) if brightness > 0 else None
        cf = random.uniform(max(0, 1 - contrast), 1 + contrast) if contrast > 0 else None      
        sf = random.uniform(max(0, 1 - saturation), 1 + saturation) if saturation > 0 else None
        hf = random.uniform(-hue, hue) if hue > 0 else None
       
        return bf, cf, sf, hf

    
    def __call__(self, clip):
       
        
        brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_factors(self.brightness, self.contrast, self.saturation, self.hue)

        list_of_transforms = []
        if brightness_factor is not None:
            list_of_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness_factor))
        if saturation_factor is not None:
            list_of_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation_factor))
        if hue_factor is not None:
            list_of_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue_factor))
        if contrast_factor is not None:
            list_of_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast_factor))

        random.shuffle(list_of_transforms)

        list_of_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + list_of_transforms + [np.array,img_as_float]
        
        
        color_augmented_clip = []
        for img in clip:
            color_augmented_img = img
            for func in list_of_transforms:
                color_augmented_img = func(color_augmented_img)
            color_augmented_clip.append(color_augmented_img.astype('float32'))

        return color_augmented_clip

class Flipping(object):
    def __init__(self, time_flip=False, horizontal_flip=False):
        self.flip_on_time = time_flip
        self.flip_horizontally = horizontal_flip

    def __call__(self, clip):
        if random.random() < 0.5 and self.flip_on_time:
            clip.reverse()
            return clip
        if random.random() < 0.5 and self.flip_horizontally:
            fliped_clip=[]
            for img in clip:
                fliped_clip.append(np.fliplr(img))
            return fliped_clip

        return clip

class Transformations:
    def __init__(self, resize_param=None, rotation_param=None, flip_param=None, crop_param=None, jitter_param=None):
        self.transforms = []

        if flip_param is not None:
            self.transforms.append(Flipping(**flip_param))

        if jitter_param is not None:
            self.transforms.append(ColorAugmentation(**jitter_param))

    def __call__(self, clip):
        for t in self.transforms:
            clip = t(clip)
        return clip

class RepeatDataset(Dataset):

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
       
        return self.dataset[idx % self.dataset.__len__()]



def get_video(name, frame_shape):

    if os.path.isdir(name):
        #this means is test and we will load the entire video 
        names_of_frames = sorted(os.listdir(name))
        numpy_video = np.array([img_as_float32(io.imread(os.path.join(name, names_of_frames[idx]))) for idx in range(len(names_of_frames))])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        img = io.imread(name)

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = gray2rgb(img)

        if img.shape[2] == 4:
            img = img[..., :3]

        numpy_video = np.moveaxis(img_as_float32(img), 1, 0)
        numpy_video = numpy_video.reshape((-1,) + frame_shape)
        numpy_video = np.moveaxis(numpy_video, 1, 2)

    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        numpy_video = img_as_float32(video)
   
    return numpy_video


class DatasetFramesAndAugmentation(Dataset):

    def __init__(self, root_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.id_sampling = id_sampling
        self.is_train=is_train
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            if id_sampling:
                train_videos=[]
                train_videos_initial=os.listdir(os.path.join(root_dir, 'train'))
                for video in train_videos_initial:
                    if video.endswith("mp4"): 
                        train_videos.append(os.path.basename(video).split('#')[0])
                
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        
        self.videos = train_videos if is_train else test_videos
       
        self.transformations=Transformations(**augmentation_params) if is_train else None
       

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            
            name = self.videos[idx]
            list_of_pathes=[]
            for i in glob.glob(os.path.join(self.root_dir, name + '*')):
                
                if i.endswith("mp4"):
                    list_of_pathes.append(i)
           
            path = np.random.choice(list_of_pathes)
            
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)
            

        #read images from directory
        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            
            frames_decoded=[x.decode() for x in frames]
            index_of_frames = np.sort(np.random.choice(len(frames_decoded), replace=True, size=2))
            
            video_array = [img_as_float32(io.imread(os.path.join(path,frames_decoded[i]))) for i in index_of_frames]
           
        else:
            video_array = get_video(path, frame_shape=self.frame_shape)
            index_of_frames = np.sort(np.random.choice(len(video_array), replace=True, size=2)) if self.is_train else range(len(video_array))
            video_array = video_array[index_of_frames]

        if self.transformations is not None:
            video_array = self.transformations(video_array)

        output_dictionary = {}
        if self.is_train:
            
            output_dictionary['driving'] = np.array(video_array[1], dtype='float32').transpose((2, 0, 1))
            output_dictionary['source'] = np.array(video_array[0], dtype='float32').transpose((2, 0, 1))
           
        else:
            output_dictionary['video'] = np.array(video_array, dtype='float32').transpose((3, 0, 1, 2))

        output_dictionary['name'] = os.path.basename(path)

        return output_dictionary


