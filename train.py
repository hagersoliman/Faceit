from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger

from main_modules.model import GeneratorFullModel, DiscriminatorFullModel
from main_modules.generator import OcclusionAwareGenerator
from main_modules.discriminator import MultiScaleDiscriminator
from main_modules.keypoint_detector import KPDetector

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater,FramesDataset

import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy



def train(config, generator, discriminator, kp_detector, checkpoint, log_dir, dataset, device_ids):

    train_params = config['train_params']

    generator_learning_rate=train_params['lr_generator']
    discriminator_learning_rate=train_params['lr_discriminator']
    kp_detector_learning_rate=train_params['lr_kp_detector']

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate, betas=(0.5, 0.999))
    kp_detector_optimizer = torch.optim.Adam(kp_detector.parameters(), lr=kp_detector_learning_rate, betas=(0.5, 0.999))

    start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      generator_optimizer, discriminator_optimizer,
                                      None if kp_detector_learning_rate == 0 else kp_detector_optimizer) if checkpoint is not None else 0

    scheduler_generator = MultiStepLR(generator_optimizer, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(discriminator_optimizer, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(kp_detector_optimizer, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (kp_detector_learning_rate != 0))

    
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
        
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                
                generator_losses, generated = generator_full(x)

                loss=0
                for val in generator_losses.values():
                    loss += val.mean()
                
                loss.backward()
                generator_optimizer.step()
                generator_optimizer.zero_grad()
                kp_detector_optimizer.step()
                kp_detector_optimizer.zero_grad()
                
                if train_params['loss_weights']['generator_gan'] != 0:
                    discriminator_optimizer.zero_grad()
                    discriminator_losses = discriminator_full(x, generated)

                    loss=0
                    for val in discriminator_losses.values():
                        loss += val.mean()

                    loss.backward()
                    discriminator_optimizer.step()
                    discriminator_optimizer.zero_grad()
                else:
                    discriminator_losses = {}
                
                generator_losses.update(discriminator_losses)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in generator_losses.items()}

                logger.log_iter(losses=losses)

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': generator_optimizer,
                                     'optimizer_discriminator': discriminator_optimizer,
                                     'optimizer_kp_detector': kp_detector_optimizer}, inp=x, out=generated)




if __name__ == "__main__":
    
    
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")

    arguments = parser.parse_args()
    
    with open(arguments.config) as f:
        configuration = yaml.load(f)

    directory_of_logs=os.path.join(*os.path.split(arguments.checkpoint)[:-1]) if arguments.checkpoint is not None else os.path.join(arguments.log_dir,
     os.path.basename(arguments.config).split('.')[0]) +' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())
 
    
    generator = OcclusionAwareGenerator(**configuration['model_params']['generator_params'],**configuration['model_params']['common_params'])
    generator.to(arguments.device_ids[0]) if torch.cuda.is_available() else None
        
   
    discriminator = MultiScaleDiscriminator(**configuration['model_params']['discriminator_params'],**configuration['model_params']['common_params'])
    discriminator.to(arguments.device_ids[0]) if torch.cuda.is_available() else None
    

    keypoint_detector = KPDetector(**configuration['model_params']['kp_detector_params'],**configuration['model_params']['common_params'])
    keypoint_detector.to(arguments.device_ids[0]) if torch.cuda.is_available() else None


    dataset = FramesDataset(is_train=True, **configuration['dataset_params'])
    
    
    os.makedirs(directory_of_logs) if not os.path.exists(directory_of_logs) else None

    
    copy(arguments.config, directory_of_logs) if not os.path.exists(os.path.join(directory_of_logs, os.path.basename(arguments.config))) else None

    
    train(configuration, generator, discriminator, keypoint_detector, arguments.checkpoint, directory_of_logs, dataset, arguments.device_ids)
    
