from PULSE_network_ilo import PULSE
from torch.utils.data import Dataset, DataLoader
from torch import isnan
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import argparse
import re
import numpy as np
import time

class Images(Dataset):
    def __init__(self, root_dir, duplicates, save_best):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png")) + list(self.root_path.glob("*.jpg"))
        self.duplicates = duplicates # Number of times to duplicate the image in the dataset to produce multiple HR images
        self.save_best = save_best

    def __len__(self):
        return self.duplicates*len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx//self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if(self.duplicates == 1 or self.save_best):
            return image,img_path.stem
        else:
            return image,img_path.stem+f"_{(idx % self.duplicates)+1}"

parser = argparse.ArgumentParser(description='PULSE')

#I/O arguments
parser.add_argument('-input_dir', type=str, default='input', help='input data directory')
parser.add_argument('-output_dir', type=str, default='runs', help='output data directory')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')
parser.add_argument('-facebank_dir', type=str, default='data/facebank', help='facebank directory')
parser.add_argument('-duplicates', type=int, default=1, help='How many HR images to produce for every image in the input directory')
parser.add_argument('-batch_size', type=int, default=1, help='Batch size to use during optimization')

#PULSE arguments
parser.add_argument('-seed', type=int, default=0, help='manual seed to use')
parser.add_argument('-loss_str', type=str, default="100*L2+0.05*GEOCROSS+3*ARCFACE", help='Loss function to use')
parser.add_argument('-eps', type=float, default=2e-3, help='Target for downscaling loss (L2)')
parser.add_argument('-noise_type', type=str, default='trainable', help='zero, fixed, or trainable')
parser.add_argument('-num_trainable_noise_layers', type=int, default=5, help='Number of noise layers to optimize')
parser.add_argument('-tile_latent', action='store_true', help='Whether to forcibly tile the same latent 18 times')
parser.add_argument('-bad_noise_layers', type=str, default="17", help='List of noise layers to zero out to improve image quality')
parser.add_argument('-opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
parser.add_argument('-learning_rate', type=float, default=5e-3, help='Learning rate to use during optimization')
parser.add_argument('-steps', type=int, default=100, help='Number of optimization steps')
parser.add_argument('-lr_schedule', type=str, default='linear1cycledrop', help='fixed, linear1cycledrop, linear1cycle')
parser.add_argument('-save_intermediate', action='store_true', help='Whether to store and save intermediate HR and LR images during optimization')
parser.add_argument('-ref_name', type=str, default=None, help='Name of the reference person')
parser.add_argument('-concept_name', type=str, default=None, help='Name of the concept')
parser.add_argument('-concept_weight', type=float, default=0, help='Weight of the concept vector')
parser.add_argument('-save_best', action='store_true', help='Whether only saving the best results if duplicate')
parser.add_argument('-alternating_optim', action='store_true', help='Whether only saving the best results if duplicate')

#New arguments
parser.add_argument('-num_layers_arcface', type=int, default=4, help='Number of intermediate layers of Arcface network to use in loss')
parser.add_argument('-arcface_use_ref', action='store_true', help='Whether to use reference image or low res image as comparison in Arcface loss')
parser.add_argument('-use_spherical', action='store_true', help='Whether to use spherical optimizer')
parser.add_argument('-latent_radius', type=float, default=96, help='Size of the hypersphere latents are pulled to')
parser.add_argument('-use_initialization', type=str, default='random', help='random, sibling_original, sibling_reference')

kwargs = vars(parser.parse_args())


dataset = Images(kwargs["input_dir"], duplicates=kwargs["duplicates"], save_best = kwargs["save_best"])
out_path = Path(kwargs["output_dir"])
out_path.mkdir(parents=True, exist_ok=True)
dataloader = DataLoader(dataset, batch_size=kwargs["batch_size"])

toPIL = torchvision.transforms.ToPILImage()

use_file_name = (kwargs['ref_name']==None)
losses = {}

start = time.perf_counter()

for ref_im, ref_im_name in dataloader:
    if use_file_name:
        name = ref_im_name[0]
        if name[0]<='9' and name[0]>='0':
            kwargs['ref_name'] = name[:name.find('_')]
        else:
            kwargs['ref_name'] = re.search('[a-z]+_[a-z]+',name).group()
    print(kwargs['ref_name'])
    kwargs['ref_im'] = ref_im
    model = PULSE(**kwargs)
    model.optimize_mapper(**kwargs)
    HR0, HR1, HR2 = model.generate_image()
    toPIL(HR0[0].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name}_HR0.png")
    toPIL(HR1[0].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name}_HR1.png")
    toPIL(HR2[0].cpu().detach().clamp(0, 1)).save(out_path / f"{ref_im_name}_HR2.png")
        
end = time.perf_counter()
print(f"Total time = {end-start} s")
