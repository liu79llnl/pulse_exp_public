from stylegan import G_synthesis,G_mapping
from lpips.lpips import LPIPS

from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
from loss import LossBuilder
from functools import partial
from utils import load_facebank,prepare_facebank
from config import get_config
from pathlib import Path
from bicubic import BicubicDownSample

import dnnlib
import legacy

from pixel2style2pixel.models.psp import pSp
from Learner import face_learner
from insightface.recognition.arcface_torch.backbones import get_model as get_arcface_model

from argparse import Namespace
import sys
import os
sys.path.append('/usr/xtmp/jwl50/pulse-ae/pixel2style2pixel')

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchvision.models import vgg16

def remove_loss(loss_str,s):
    s_0 = loss_str.find(s)
    s_plus = max(0,loss_str.rfind('+',0,s_0))
    s_1 = s_0+len(s)
    return loss_str[:s_plus]+loss_str[s_1:]

class PULSE(torch.nn.Module):
    def __init__(self, seed, cache_dir, ref_name, ref_im, loss_str, eps, num_layers_arcface=4, arcface_use_ref=False, latent_radius=96, verbose=True, output_dir='/usr/xtmp/jwl50/pulse-ae/mapping_distribution/', **kwargs):
        super(PULSE, self).__init__()
        self.verbose = verbose
        self.output_dir = output_dir

        #Set seed
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        #Set arcface identity
        conf = get_config(False)
        conf.facebank_path = Path(kwargs['facebank_dir'])
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok = True)
        target_faces, names = load_facebank(conf)
        idx = np.where(names == ref_name)[0][0]
        target_face = target_faces[idx-1:idx]
        target_face = torch.tensor(target_face, requires_grad=False).cuda()
        
        #Load networks
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        with dnnlib.util.open_url(network_pkl) as fp:
            self.stylegan_D = legacy.load_network_pkl(fp)['D'].eval().requires_grad_(False).cuda() # type: ignore
        for p in self.stylegan_D.parameters():
            p.requires_grad = False
        
        #Load psp network
        psp_ckpt = torch.load('pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt', map_location='cpu')
        psp_opts = psp_ckpt['opts']
        psp_opts['checkpoint_path'] = 'pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt'
        psp_opts['learn_in_w'] = False
        psp_opts['output_size'] = 1024
        psp_opts = Namespace(**psp_opts)
        self.psp = pSp(psp_opts).eval().cuda()
        for p in self.psp.parameters():
            p.requires_grad = False

        #Load LPIPS network
        self.LPIPS = LPIPS(net_type='alex').cuda().eval()
        self.lpips_transform = transforms.Resize((256,256))

        #Load VGG network
        self.vgg = vgg16(pretrained=True).cuda().eval()
        self.vgg.classifier = self.vgg.classifier[:-1]

        #Set up learned mapping network
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        self.embed_d = 10
        self.fc1 = torch.nn.Linear(512, self.embed_d).cuda()
        self.fc1.requires_grad = True
        self.fc2 = torch.nn.Linear(self.embed_d, 512).cuda()
        self.fc2.requires_grad = True
        #self.fc = torch.nn.Linear(512, 512).cuda()
        self.learned_mapper = torch.nn.Sequential(
               self.fc1,
               self.fc2,
               self.lrelu
               )

        #Compute gaussian fit for initial mapping network and distribution of vgg features
        w_avg_samples = 10000
        z_samples = torch.randn(w_avg_samples, 18, 512).cuda()
        w_samples = self.psp.decoder.style(z_samples)
        w_avg = torch.mean(w_samples, dim=[0, 1])
        w_std = torch.std(w_samples, dim=[0, 1])

        self.gaussian_fit = {"mean": w_avg, "std": w_std}
        try:
            torch.save(self.gaussian_fit, "/usr/xtmp/jwl50/pulse-ae/gaussian_fit.pt")
            if self.verbose:
                print("\tSaved \"gaussian_fit.pt\"")
        except:
            pass

        if not(os.path.exists("vgg_fit.pt")):
            vgg_features = torch.empty((len(w_samples), 4096)).cuda().detach()
            for i in range(len(w_samples)):
                print(w_samples[i].shape)
                w_sample_expanded = w_samples[i].expand(18, -1)
                gen_im, _ = self.psp.decoder([(self.psp.latent_avg+w_sample_expanded).unsqueeze(0)],
                                            input_is_latent=True,
                                            randomize_noise=True,
                                            return_latents=False)
                gen_im = (gen_im+1)/2
                print(self.vgg(gen_im).shape)
                vgg_features[i] = self.vgg(gen_im).detach()
            vgg_features_avg = torch.mean(vgg_features, dim=0)
            vgg_features_std = torch.std(vgg_features, dim=0)

            self.vgg_fit = {"mean": vgg_features_avg, "std": vgg_features_std}
            torch.save(self.vgg_fit, "vgg_fit.pt")
            if self.verbose:
                print("\tSaved \"vgg_fit.pt\"")

        self.vgg_fit = torch.load('/usr/xtmp/jwl50/pulse-ae/vgg_fit.pt', map_location='cuda')
        self.vgg_fit['std'] = torch.clamp(self.vgg_fit['std'], min=torch.mean(self.vgg_fit['std'], dim=0))

        #Set up loss constants
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)

        #Set up arcface networks
        self.conf = conf
        self.facenet = face_learner(conf, True)
        self.facenet.threshold = 1.5
        if conf.device.type == 'cpu':
            self.facenet.load_state(conf, 'cpu_final.pth', True, True)
        else:
            self.facenet.load_state(conf, 'final.pth', True, True)

        self.num_layers_arcface = num_layers_arcface
        if self.num_layers_arcface == None:
            #Only last layer of arcface
            self.facenet.model.eval()
        elif self.num_layers_arcface == 1 or self.num_layers_arcface == 4:
            #Multiple layers
            name = "r100"
            weight = "/usr/xtmp/jl888/pulse-ae/saved_models/ms1mv3_arcface_r100/backbone.pth"
            self.facenet.model = get_arcface_model(name, fp16=False)
            self.facenet.model.load_state_dict(torch.load(weight))
            self.inter_features = {}
            def get_features(name):
                def hook(model, input, output):
                    self.inter_features[name] = output
                return hook
            self.facenet.model.layer1.register_forward_hook(get_features('layer1'))
            self.facenet.model.layer2.register_forward_hook(get_features('layer2'))
            self.facenet.model.layer3.register_forward_hook(get_features('layer3'))
            self.facenet.model.layer4.register_forward_hook(get_features('layer4'))
            self.facenet.model.eval().requires_grad_(False).to(conf.device)
            for param in self.facenet.model.parameters():
                param.requires_grad = False

            self.facenet_model_copy = get_arcface_model(name, fp16=False)
            self.facenet_model_copy.load_state_dict(torch.load(weight))
            self.inter_features_copy = {}
            def get_features_copy(name):
                def hook(model, input, output):
                    self.inter_features_copy[name] = output
                return hook
            self.facenet_model_copy.layer1.register_forward_hook(get_features_copy('layer1'))
            self.facenet_model_copy.layer2.register_forward_hook(get_features_copy('layer2'))
            self.facenet_model_copy.layer3.register_forward_hook(get_features_copy('layer3'))
            self.facenet_model_copy.layer4.register_forward_hook(get_features_copy('layer4'))
            self.facenet_model_copy.eval().requires_grad_(False).to(conf.device)
            for param in self.facenet_model_copy.parameters():
                param.requires_grad = False

        self.ref_im = ref_im.cuda()
        self.target_face = target_face

        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]

        self.eps = eps
        self.arcface_use_ref = arcface_use_ref
        self.latent_radius = latent_radius

    def _loss_l2(self, gen_im_lr, **kwargs):
        return (((gen_im_lr - self.ref_im)).pow(2).mean((1, 2, 3)).sum())

    def _loss_latentnorm(self, latent, **kwargs):
        #Loss in initial Gaussian distribution
        latent_z = torch.nn.functional.leaky_relu(latent, negative_slope=5) / self.gaussian_fit['std']
        return torch.abs(torch.clamp(torch.norm(latent_z), min=self.latent_radius)-self.latent_radius)

    def _loss_pairnorm(self, latent1, latent2, **kwargs):
        return torch.sigmoid(-torch.norm((torch.nn.functional.leaky_relu(latent1, negative_slope=5) - torch.nn.functional.leaky_relu(latent2, negative_slope=5))/self.gaussian_fit['std']))

    def _loss_lpips(self, gen_im, **kwargs):
        return self.LPIPS(self.lpips_transform(gen_im), self.lpips_transform(self.ref_im))

    def _loss_vgg(self, gen_im, **kwargs):
        return torch.norm((self.vgg(gen_im) - self.vgg_fit['mean'])/self.vgg_fit['std'])

    def _loss_arcface(self, gen_im, **kwargs):
        a = 124
        b = 901

        face = (F.interpolate(gen_im[:,:,a:b,a:b], [112,112], mode = 'bicubic').clamp(0,1)-0.5)/0.5

        if self.arcface_use_ref:
            target_face_112 = self.target_face.cuda().requires_grad_(False).clone().sub_(0.5).div_(0.5)
        else:
            target_face_112 = (F.interpolate(F.interpolate(self.ref_im, [1024, 1024])[:,:,a:b,a:b], [112,112], mode='bicubic')).cuda().sub_(0.5).div_(0.5)

        if self.num_layers_arcface == None:
            #Only last layer of arcface
            score = self.facenet.infer(self.conf, face, self.target_face)
            return score.sum()
        elif self.num_layers_arcface == 1 or self.num_layers_arcface == 4:
            #Last 4
            face_emb = self.facenet.model(face)
            face_inter_emb1, face_inter_emb2, face_inter_emb3, face_inter_emb4 = self.inter_features['layer1'], self.inter_features['layer2'], self.inter_features['layer3'], self.inter_features['layer4']
            target_emb = self.facenet_model_copy(target_face_112)
            target_inter_emb1, target_inter_emb2, target_inter_emb3, target_inter_emb4 = self.inter_features_copy['layer1'], self.inter_features_copy['layer2'], self.inter_features_copy['layer3'], self.inter_features_copy['layer4']

            score = (face_emb - target_emb).pow(2).mean()
            score1 = (face_inter_emb1 - target_inter_emb1).pow(2).mean()
            score2 = (face_inter_emb2 - target_inter_emb2).pow(2).mean()
            score3 = (face_inter_emb3 - target_inter_emb3).pow(2).mean()
            score4 = (face_inter_emb4 - target_inter_emb4).pow(2).mean()

            if self.num_layers_arcface == 1:
                return score
            elif self.num_layers_arcface == 4:
                return (score+score1+score2+score3+score4)/4

    def forward(self, latent_in, epoch_in, epoch_total, **kwargs):
        latent_in1 = self.learned_mapper(latent_in).expand(18, -1)
        #latent_in1 = self.psp.decoder.style(latent_in).expand(18, -1)

        gen_im1, _ = self.psp.decoder([(self.psp.latent_avg+latent_in1).unsqueeze(0)],
                                        input_is_latent=True,
                                        randomize_noise=True,
                                        return_latents=False)
        gen_im1 = (gen_im1+1)/2

        loss = 0
        var_dict = {'latent': latent_in1,
                    'gen_im_lr': self.D(gen_im1),
                    'ref_im': self.ref_im,
                    'gen_im': gen_im1,
                    'target_face': self.target_face,
                    }
        loss_fun_dict = {
            'L2': self._loss_l2,
            'ARCFACE': self._loss_arcface,
            'LATENTNORM': self._loss_latentnorm,
            'LPIPS': self._loss_lpips,
            'VGG': self._loss_vgg,
        }
        for weight, loss_type in self.parsed_loss:
            if float(weight) != 0:
                temp = loss_fun_dict[loss_type](**var_dict)
                loss += float(weight)*temp
                print("{}: {}".format(loss_type, temp))

        return loss

    def optimize_mapper(self, steps, opt_name, lr_schedule, use_spherical, learning_rate, **kwargs):
        #Set up optimizer
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
            'stepwise': lambda x: 10**(-x//1000)
        }

        opt_func = opt_dict[opt_name]
        if use_spherical:
            opt = SphericalOptimizer(opt_func, self.learned_mapper.parameters(), lr=learning_rate, betas = (0.9, 0.999))
        else:
            opt = opt_func(self.learned_mapper.parameters(), lr=learning_rate, betas = (0.9, 0.999))
    
        schedule_func = schedule_dict[lr_schedule]
        if use_spherical:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)

        num_latent_ins = 1
        self.latent_ins = [None for _ in range(num_latent_ins)]
        for i in range(len(self.latent_ins)):
            self.latent_ins[i] = torch.unsqueeze(torch.normal(torch.zeros_like(self.gaussian_fit['mean']), self.gaussian_fit['std']), dim=0).cuda()

        for j in range(steps):
            print("Step {}".format(j))

            for i in range(len(self.latent_ins)):
                print("Optimizing latent {}".format(i))
                latent_in = self.latent_ins[i]
                for pulse_run_step in range(1):
                    loss = self.forward(latent_in, j, steps)
                    loss.backward()

            opt.step()
            if use_spherical:
                opt.opt.zero_grad()
            else:
                opt.zero_grad()
            scheduler.step()

    def generate_image(self):
        latent_in = torch.unsqueeze(torch.zeros_like(self.gaussian_fit['mean']), dim=0).cuda()
        latent_in = self.learned_mapper(latent_in).expand(18, -1)
        #latent_in = self.psp.decoder.style(latent_in).expand(18, -1)
        gen_im, _ = self.psp.decoder([(self.psp.latent_avg+latent_in).unsqueeze(0)],
                                        input_is_latent=True,
                                        randomize_noise=True,
                                        return_latents=False)
        gen_im1 = (gen_im+1)/2

        for i in range(len(self.latent_ins)):
            latent_in = self.latent_ins[i]
            latent_in = self.learned_mapper(latent_in).expand(18, -1)
            #latent_in = self.psp.decoder.style(latent_in).expand(18, -1)
            gen_im, _ = self.psp.decoder([(self.psp.latent_avg+latent_in).unsqueeze(0)],
                                            input_is_latent=True,
                                            randomize_noise=True,
                                            return_latents=False)
            gen_im2 = (gen_im+1)/2
            toPIL = torchvision.transforms.ToPILImage()
            toPIL(gen_im2[0].cpu().detach().clamp(0, 1)).save(os.path.join(self.output_dir, '{}.png'.format(i)))

        return gen_im1.clone().cpu().detach().clamp(0, 1), gen_im2.clone().cpu().detach().clamp(0, 1), gen_im3.clone().cpu().detach().clamp(0, 1)
