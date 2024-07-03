import numpy as np
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import os
import wandb
os.environ['WANDB_API_KEY'] = '6a4e3c8ae2f28b0b79564b0d3290599456d0c71f'
import torch.nn as nn

from beats.BEATs import BEATs, BEATsConfig

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_nse', type=float, default=1.0, help='weight for noise reconstruction loss: nse(B(G(X)), N(Y)')
        parser.add_argument('--noise_emb_type', type=str, default='none', help='use either none or BEATs')
        parser.add_argument('--inject_type', type=str, default='FiLM', help='way to inject noise embedding into generator')
        parser.add_argument('--inject_layers', type=str, default='11,12,14,16,18,20', help='which layers to inject noise embeddings')
        parser.add_argument('--num_inject_layers', type=int, default='6', help='how many layers to inject noise embeddings')
        parser.add_argument('--discriminator', type=int, default=1, help='use discriminator(1) or not(0)')
        parser.add_argument('--gaussian_stddev', type=float, default=0.0, help='standard deviation of Gaussiam noise')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'nse']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.inject_type = self.opt.inject_type
        self.inject_layers = [int(i) for i in self.opt.inject_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G & N
            self.model_names = ['G']

        # Pre-trained audio classification model BEATs
        BEATs_path ='/share/nas169/jethrowang/NADA-GAN/BEATs/BEATs_finetuned_trg_noisy_10epochs.pt'
        checkpoint = torch.load(BEATs_path)
        cfg = BEATsConfig(checkpoint['cfg'])
        self.netB = BEATs(cfg).to(self.device)
        self.netB.load_state_dict(checkpoint['model'], strict=False)
        self.netB.eval()

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        # for layer_id, layer in enumerate(self.netG.model):
            # print(f'Layer ID: {layer_id}, Layer: {layer}')

        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.real_C = self.real_C[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_C = input['C'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
        
        # prepare noise features for loss_nse
        # print(f'real_B: {self.real_B.shape}')
        self.real_B_2d = self.real_B.view(self.real_B.size(0), self.real_B.size(1) * self.real_B.size(2) * self.real_B.size(3))
        # print(f'real_B_2d: {self.real_B_2d.shape}')
        self.trg_noise_feat, _, _ = self.netB(self.real_B_2d)
        # print(f'trg_noise_feat: {self.trg_noise_feat.shape}')

        # prepare noise embeddings for generator
        if self.opt.noise_emb_type == 'BEATs':
            self.trg_noise_emb = self.trg_noise_feat

        if self.opt.state == "Test":
            self.trg_noise_emb = util.add_gaussian_noise(self.trg_noise_emb, stddev=self.opt.gaussian_stddev, seed=42)
            # self.trg_noise_emb = util.add_mixture_gaussian_noise(self.trg_noise_emb, seed=42)

        # self.fake = self.netG(self.real)
        # self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.noise_emb_type is not None:
            self.fake_B = self.netG(self.real_A, noise_emb_type=self.opt.noise_emb_type, noise_emb=self.trg_noise_emb, inject_layers=self.inject_layers)
        else:
            self.fake_B = self.netG(self.real_A)

        if self.opt.nce_idt:
            # self.idt_B = self.fake[self.real_A.size(0):]
            if self.opt.noise_emb_type is not None:
                self.idt_B = self.netG(self.real_B, noise_emb_type=self.opt.noise_emb_type, noise_emb=self.trg_noise_emb, inject_layers=self.inject_layers)
            else:
                self.idt_B = self.netG(self.real_B)
    
    def collect_D_results(self, y_true, y_pred):
        # real
        real_output = self.netD(self.real_B)
        real_labels = torch.ones(self.real_B.size(0)).to(self.device)
        y_true.extend(real_labels.cpu().numpy())
        y_pred.extend(real_output.cpu().detach().numpy())

        # fake
        fake = self.fake_B.detach()
        fake_output = self.netD(fake)
        fake_labels = torch.zeros(fake.size(0)).to(self.device)
        y_true.extend(fake_labels.cpu().numpy())
        y_pred.extend(fake_output.cpu().detach().numpy())
    

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        if self.opt.discriminator == 1:
            fake = self.fake_B.detach()
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD(fake)
            self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
            # Real
            self.pred_real = self.netD(self.real_B)
            loss_D_real = self.criterionGAN(self.pred_real, True)
            self.loss_D_real = loss_D_real.mean()
            
            # wandb.log({'loss_D_fake': self.loss_D_fake.item() * 0.5, 'loss_D_real': self.loss_D_real.item() * 0.5})
        else:
            self.loss_D_fake = torch.tensor(0.0, requires_grad=True)
            self.loss_D_real = torch.tensor(0.0, requires_grad=True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN, NCE, nse, ASR loss for the generator"""
        fake = self.fake_B

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.lambda_nse > 0.0:
            fake_2d = fake.view(fake.size(0), fake.size(1) * fake.size(2) * fake.size(3))
            pred_noise_feat, _, _ = self.netB(fake_2d)
            self.loss_nse = torch.mean(torch.abs(pred_noise_feat - self.trg_noise_feat)) * self.opt.lambda_nse
        else:
            self.loss_nse = 0.0

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_nse

        # if self.opt.lambda_nse > 0.0:
            # wandb.log({'loss_G_GAN': self.loss_G_GAN.item(), 'loss_NCE_both': loss_NCE_both.item(), 'loss_nse': self.loss_nse.item()})
        # else:
            # wandb.log({'loss_G_GAN': self.loss_G_GAN.item(), 'loss_NCE_both': loss_NCE_both.item()})

        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, layers=self.nce_layers, encode_only=True, noise_emb_type=self.opt.noise_emb_type, noise_emb=self.trg_noise_emb, inject_layers=self.inject_layers)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, layers=self.nce_layers, encode_only=True, noise_emb_type=self.opt.noise_emb_type, noise_emb=self.trg_noise_emb, inject_layers=self.inject_layers)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    