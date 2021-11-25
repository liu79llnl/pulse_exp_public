import torch
import torch.nn.functional as F
from bicubic import BicubicDownSample
from Learner import face_learner
import warnings
import numpy as np
#from lpips import lpips
from torchvision import transforms
from torchvision.models import vgg16

import sys
from insightface.recognition.arcface_torch.backbones import get_model

#import dnnlib
#import legacy

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, target_face, loss_str, conf, eps, stylegan_D, prev_embeds=[], num_layers_arcface=4, arcface_use_ref=False, latent_radius=96, hr_sibling=None):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2]==ref_im.shape[3]
        im_size = ref_im.shape[2]
        factor=1024//im_size
        assert im_size*factor==1024
        self.D = BicubicDownSample(factor=factor)

        self.conf = conf
        self.facenet = face_learner(conf, True)
        self.facenet.threshold = 1.5
        if conf.device.type == 'cpu':
            self.facenet.load_state(conf, 'cpu_final.pth', True, True)
        else:
            self.facenet.load_state(conf, 'final.pth', True, True)

        self.num_layers_arcface = num_layers_arcface
        self.arcface_use_ref = arcface_use_ref

        if self.num_layers_arcface == None:
            #Only last layer of arcface
            self.facenet.model.eval()

        if self.num_layers_arcface == 1 or self.num_layers_arcface == 4:
            #Multiple layers
            name = "r100"
            weight = "/usr/xtmp/jl888/pulse-ae/saved_models/ms1mv3_arcface_r100/backbone.pth"
            self.facenet.model = get_model(name, fp16=False)
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

            self.facenet_model_copy = get_model(name, fp16=False)
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

        self.ref_im = ref_im
        print(ref_im.shape)
        
        self.target_face = target_face
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.eps = eps

        #Compute image Laplacian
        ref_im_gray =  ref_im[:,0,:,:]*0.299+ref_im[:,1,:,:]*0.5870+ref_im[:,2,:,:]*0.1140
        # laplacian = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        # self.laplacian_weights = F.conv2d(ref_im_gray.unsqueeze(0), laplacian.cuda(), padding=1)
        # self.laplacian_weights = torch.cat([self.laplacian_weights]*3, 1).cuda().abs()
        # print(torch.norm(self.laplacian_weights))
        # print(torch.norm(torch.ones_like(self.laplacian_weights)))
        # #use laplacian in L2 loss
        # alpha = 0
        # self.laplacian_weights = alpha/torch.norm(self.laplacian_weights)*self.laplacian_weights+(1-alpha)/torch.norm(torch.ones_like(self.laplacian_weights))*torch.ones_like(self.laplacian_weights).cuda()
        # self.laplacian_weights /= torch.norm(self.laplacian_weights)
        # #no laplacian in L2 loss
        # #self.laplacian_weights = torch.ones_like(self.laplacian_weights).cuda()

        # #update laplacian in future
        # self.laplacian_update = None
        
        # Load networks.
        self.stylegan_D = stylegan_D

        #self.lpips = lpips.PerceptualLoss(model="net-lin", net="vgg",
        #                                  use_gpu=True)
        """
        network_pkl = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
        print('Loading networks from "%s"...' % network_pkl)
        with dnnlib.util.open_url(network_pkl) as fp:
            self.stylegan_D = legacy.load_network_pkl(fp)['D'].eval().requires_grad_(False).cuda() # type: ignore        
        """
        self.prev_embeds = prev_embeds

        self.gaussian_fit = torch.load("gaussian_fit.pt")
        self.latent_radius = latent_radius

        self.hr_sibling = hr_sibling

        #Load VGG network
        self.vgg = vgg16(pretrained=True).cuda().eval()
        self.vgg.classifier = self.vgg.classifier[:-1]
        
        self.vgg_fit = torch.load('vgg_fit.pt', map_location='cuda')
        self.vgg_fit['std'] = torch.clamp(self.vgg_fit['std'], min=torch.mean(self.vgg_fit['std'], dim=0))

    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im_lr, ref_im, **kwargs):
        """
        if (ref_im[:,0,:,:] == ref_im[:,1,:,:]).float().mean()==1:
            gen_im_lr_gray = gen_im_lr[:,0,:,:]*0.299+gen_im_lr[:,1,:,:]*0.5870+gen_im_lr[:,2,:,:]*0.1140
            ref_im_gray =  ref_im[:,0,:,:]*0.299+ref_im[:,1,:,:]*0.5870+ref_im[:,2,:,:]*0.1140
            # print(gen_im_lr_gray.shape)
            return ((gen_im_lr_gray - ref_im_gray).pow(2).mean((1, 2)).clamp(min=self.eps).sum())
        else:
            return ((gen_im_lr - ref_im).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())
        """
        #self.laplacian_update = ((gen_im_lr - self.ref_im)*torch.ones_like(self.laplacian_weights)).pow(2).detach().clone()
        #self.laplacian_update /= torch.norm(self.laplacian_update)
        #self.update_laplacian()
        #print(self.laplacian_weights)

        # return (((gen_im_lr - ref_im)*self.laplacian_weights).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())
        #return (((gen_im_lr - ref_im)).pow(2).mean((1, 2, 3)).clamp(min=self.eps).sum())
        return (((gen_im_lr - ref_im)).pow(2).mean((1, 2, 3)).sum())

    def update_laplacian(self):
        self.laplacian_weights = 0.99*self.laplacian_weights + 0.01*self.laplacian_update
        self.laplacian_weights /= torch.norm(self.laplacian_weights)

    def initialize_laplacian(self):
        ref_im_gray =  self.ref_im[:,0,:,:]*0.299+self.ref_im[:,1,:,:]*0.5870+self.ref_im[:,2,:,:]*0.1140
        laplacian = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).unsqueeze(0).unsqueeze(0)
        self.laplacian_weights = F.conv2d(ref_im_gray.unsqueeze(0), laplacian.cuda(), padding=1)
        self.laplacian_weights = 0*self.laplacian_weights+1*torch.ones_like(self.laplacian_weights).cuda()
        self.laplacian_weights = torch.cat([self.laplacian_weights]*3, 1).cuda().abs()
        self.laplacian_weights /= torch.norm(self.laplacian_weights)
        
    def _loss_l1(self, gen_im_lr, ref_im, **kwargs):
        return 10*(((gen_im_lr - ref_im)*self.laplacian_weights).abs().mean((1, 2, 3)).clamp(min=self.eps).sum())

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if(latent.shape[1] == 1):
            return 0
        else:
            X = latent.view(-1, 1, 18, 512)
            Y = latent.view(-1, 18, 1, 512)
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*512).mean((1, 2))/8.).sum()
            return D
    
    def _loss_latentnorm(self, latent, **kwargs):
        #Loss using entire norm
        #print(torch.norm(latent.squeeze(0), p=1, dim=1))
        #return torch.abs(torch.norm(latent)-self.latent_radius)

        #Loss using norms of each style vector
        #style_norms = torch.norm(latent.squeeze(0), p=1, dim=0)
        #return torch.norm(style_norms-self.latent_radius*torch.ones_like(style_norms), p=1)

        #Loss in initial Gaussian distribution
        latent_z = torch.nn.functional.leaky_relu(latent, negative_slope=5) / self.gaussian_fit['std']
        return torch.abs(torch.norm(latent_z)-self.latent_radius)
        # return torch.abs(torch.norm(latent)-self.latent_radius)

    def _loss_vgg(self, gen_im, **kwargs):
        return torch.norm((self.vgg(gen_im) - self.vgg_fit['mean'])/self.vgg_fit['std'])
    
    def _loss_arcface(self, gen_im, target_face, ref_im, **kwargs):
        a = 124
        b = 901

        if True: #target_face.shape[0] > 112:
            face = (F.interpolate(gen_im[:,:,a:b,a:b], [112,112], mode = 'bicubic').clamp(0,1)-0.5)/0.5
        else:
            face = (F.interpolate(F.interpolate(gen_im[:,:,a:b,a:b], [ref_im.shape[0], ref_im.shape[0]], mode='bicubic'), [112,112], mode = 'bicubic').clamp(0,1)-0.5)/0.5
        #print(target_face.shape)
        if self.arcface_use_ref:
            #target_face_112 = F.interpolate(target_face, [112,112], mode='bicubic').cuda().sub_(0.5).div_(0.5)
            target_face_112 = target_face.cuda().requires_grad_(False).clone().sub_(0.5).div_(0.5)
        else:
            #target_face_112 = F.interpolate(ref_im, [112,112], mode='bicubic').cuda().sub_(0.5).div_(0.5)
            target_face_112 = (F.interpolate(F.interpolate(ref_im, [1024, 1024])[:,:,a:b,a:b], [112,112], mode='bicubic')).cuda().sub_(0.5).div_(0.5)
        #print(torch.max(target_face_112))

        if self.num_layers_arcface == None:
            #Only last layer of arcface
            score = self.facenet.infer(self.conf, face, target_face)
            return score.sum()

        elif self.num_layers_arcface == 1 or self.num_layers_arcface == 4:
            #Last 4
            face_emb = self.facenet.model(face)
            face_inter_emb1, face_inter_emb2, face_inter_emb3, face_inter_emb4 = self.inter_features['layer1'], self.inter_features['layer2'], self.inter_features['layer3'], self.inter_features['layer4']

            #target_face_112 = F.interpolate(ref_im, [112,112], mode='bicubic')
            #print(torch.max(target_face_112))
            #target_face_tmp = target_face_112.cuda().sub_(0.5).div_(0.5)
            #target_emb = self.facenet_model_copy(target_face_tmp)
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

    def _loss_discriminator(self, gen_im, **kwargs):
        score = self.stylegan_D(gen_im, None)
        return torch.abs(score[0][0])

    def _loss_uniqueness(self, gen_im, **kwargs):
        if len(self.prev_embeds) == 0:
            return torch.tensor(0)[()].cuda()
        a = 124
        b = 901
        face = (F.interpolate(gen_im[:,:,a:b,a:b], [112,112], mode = 'bicubic').clamp(0,1)-0.5)/0.5
        face_embed = self.facenet.model(face)
        score = sum(list(map(lambda x : F.cosine_similarity(face_embed, x).abs(), self.prev_embeds)))
        
        return score.item()

    def _loss_lpips(self, gen_im, hr_sibling, **kwargs):
        gen_im_LPIPS = (F.interpolate(gen_im, [256,256], mode='bicubic'))
        ref_im_LPIPS = (F.interpolate(hr_sibling, [256,256], mode='bicubic'))
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        score = self.lpips(normalize(gen_im_LPIPS),
                           normalize(ref_im_LPIPS)).mean()
        return score.item()

    def forward(self, latent, gen_im):
        var_dict = {'latent': latent,
                    'gen_im_lr': self.D(gen_im),
                    'ref_im': self.ref_im,
                    'gen_im': gen_im,
                    'target_face': self.target_face,
                    'hr_sibling': self.hr_sibling,
                    }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'ARCFACE': self._loss_arcface,
            'DISCRIMINATOR': self._loss_discriminator,
            'UNIQUENESS': self._loss_uniqueness,
            #'LPIPS': self._loss_lpips,
            'LATENTNORM': self._loss_latentnorm,
            'VGG': self._loss_vgg,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        #print(losses)
        return loss, losses