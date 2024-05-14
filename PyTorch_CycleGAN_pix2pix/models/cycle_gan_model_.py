import torch
import torch.nn as nn
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg
from PIL import Image
from torchvision.models.resnet import ResNet50_Weights
import scipy.ndimage as nd
import torch.nn.functional as F
from .loss import GradientPenaltyLoss, HFENLoss, TVLoss, GPLoss, ElasticLoss, RelativeL1, L1CosineSim, ColorLoss, GradientLoss
from collections import OrderedDict

# VGG 19 layers to listen to
vgg_layer19 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'conv_3_4': 16, 'pool_3': 18, 'conv_4_1': 19, 'conv_4_2': 21, 'conv_4_3': 23, 'conv_4_4': 25, 'pool_4': 27, 'conv_5_1': 28, 'conv_5_2': 30, 'conv_5_3': 32, 'conv_5_4': 34, 'pool_5': 36
}
vgg_layer_inv19 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'conv_3_4', 18: 'pool_3', 19: 'conv_4_1', 21: 'conv_4_2', 23: 'conv_4_3', 25: 'conv_4_4', 27: 'pool_4', 28: 'conv_5_1', 30: 'conv_5_2', 32: 'conv_5_3', 34: 'conv_5_4', 36: 'pool_5'
}
# VGG 16 layers to listen to
vgg_layer16 = {
    'conv_1_1': 0, 'conv_1_2': 2, 'pool_1': 4, 'conv_2_1': 5, 'conv_2_2': 7, 'pool_2': 9, 'conv_3_1': 10, 'conv_3_2': 12, 'conv_3_3': 14, 'pool_3': 16, 'conv_4_1': 17, 'conv_4_2': 19, 'conv_4_3': 21, 'pool_4': 23, 'conv_5_1': 24, 'conv_5_2': 26, 'conv_5_3': 28, 'pool_5': 30
}
vgg_layer_inv16 = {
    0: 'conv_1_1', 2: 'conv_1_2', 4: 'pool_1', 5: 'conv_2_1', 7: 'conv_2_2', 9: 'pool_2', 10: 'conv_3_1', 12: 'conv_3_2', 14: 'conv_3_3', 16: 'pool_3', 17: 'conv_4_1', 19: 'conv_4_2', 21: 'conv_4_3', 23: 'pool_4', 24: 'conv_5_1', 26: 'conv_5_2', 28: 'conv_5_3', 30: 'pool_5'
}


# +
class StyleLoss_(nn.Module):
    def __init__(self, feature_extractor):
        super(StyleLoss_, self).__init__()
        self.feature_extractor = feature_extractor.eval()
        self.criterion = nn.MSELoss()

    def forward(self, fake, target):
        fake_features = self.feature_extractor(fake).detach()
        target_features = self.feature_extractor(target).detach()
        return self.criterion(fake_features, target_features)

# def get_resnet50_feature_extractor(device):
#     resnet50 = models.resnet50(weights='DEFAULT').to(device)
# #     feature_extractor = nn.Sequential(*list(resnet50.children())[:-2])  # Remove the last 2 layers (avgpool and fc)
#     for param in feature_extractor.parameters():
#         param.requires_grad = False
#     return feature_extractor


def get_vgg19_feature_extractor(device):
    vgg19 = models.vgg19(weights="DEFAULT").to(device)
    feature_extractor = nn.Sequential(*list(vgg19.children())[:-2])  # Remove the last 2 layers (Linear and Avg pool)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    return feature_extractor

def get_resnext_feature_extractor(device):
    resnext = models.resnext101_64x4d(weights="DEFAULT").to(device)
    feature_extractor = nn.Sequential(*list(resnext.children())[:-2])  # Remove the last 2 layers (Linear and Avg pool)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    return feature_extractor


# +
def gram_matrix(input):
    """
    Compute the Gram matrix of a batch of feature maps.
    """
    num_channels, height, width = input.size()
    features = input.view(num_channels, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(num_channels * height * width)

def get_style_reconstruction_loss(style_reference_image, generated_image, device):
    """
    Compute the style reconstruction loss between a style reference image and a generated image.

    Args:
        style_reference_image: Style reference image tensor.
        generated_image: Generated image tensor.
        device: Device to perform computations.

    Returns:
        style_loss: Style reconstruction loss.
    """
    # Get VGG-19 feature extractor
    feature_extractor = get_vgg19_feature_extractor(device)

    # Set model to evaluation mode
    feature_extractor.eval()

    # Extract feature maps from the style reference image and the generated image
    with torch.no_grad():
        style_reference_features = feature_extractor(style_reference_image)
        generated_features = feature_extractor(generated_image)

    # Compute the Gram matrices for the style reference image and the generated image
    style_reference_grams = [gram_matrix(feature_map) for feature_map in style_reference_features]
    generated_grams = [gram_matrix(feature_map) for feature_map in generated_features]

    # Compute the style reconstruction loss
    style_loss = sum(F.mse_loss(generated_gram, style_reference_gram) 
                     for generated_gram, style_reference_gram in zip(generated_grams, style_reference_grams))

    return style_loss


# +
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss (fake,real)')
            parser.add_argument('--lambda_ltr', type=float, default=0.0, help='weight for TRL loss')
            parser.add_argument('--lambda_tv', type=float, default=0.0, help='weight for Total Variation loss')
            parser.add_argument('--lambda_ganA', type=float, default=0.0, help='weight for GAN A loss')
            parser.add_argument('--lambda_ganB', type=float, default=0.0, help='weight for GAN B loss')
            parser.add_argument('--lambda_MSE', type=float, default=0.0, help='weight for MSE loss')
            parser.add_argument('--lambda_style_rec', type=float, default=0.0, help='weight for Style reconstruction loss')
            parser.add_argument('--lambda_l1', type=float, default=0.0, help='weight for L1 Loss between fake and real')
            parser.add_argument('--lambda_D_adv', type=float, default=0.5, help='weight for adversarial loss in Ds')
            parser.add_argument('--lambda_l1_D', type=float, default=0.0, help='weight for L1 loss in Ds')
            parser.add_argument('--lambda_con_vgg_19',type=float,default=0.0,help="weight for contextual loss vgg19-based")
            parser.add_argument('--lambda_charbonnier',type=float,default=0.0,help="weight for charbonnier loss")
            parser.add_argument('--lambda_MSP',type=float,default=0.0,help="weight for MSP loss")
            
            


        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','ltr_A','ltr_B','style_A', 'style_B', 'tv_A','tv_B','L1_A','L1_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        if self.isTrain and self.opt.lambda_ltr > 0.0:  
            visual_names_A.append('ltr_B')
            visual_names_B.append('ltr_A')
#         if self.isTrain and self.opt.lambda_style > 0.0:  
#             visual_names_A.append('style_B')
#             visual_names_B.append('style_A')
        if self.isTrain and self.opt.lambda_tv > 0.0:  
            visual_names_A.append('tv_B')
            visual_names_B.append('tv_A')
#         if self.isTrain and self.opt.lambda_l1 > 0.0:  
#             visual_names_A.append('L1_B')
#             visual_names_B.append('L1_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = nn.MSELoss()
            self.criterionL1=nn.L1Loss()
            self.criterionL1_D=nn.L1Loss()
            self.criterionTv =  RelativeL1()
            self.criterionLtr = GradientLoss()
#             self.criterionStyle=StyleLoss(device='cuda:{}'.format(self.gpu_ids[0]) if self.gpu_ids else torch.device('cpu'))#net = vgg16
            self.criterionContextual_vgg_19=Contextual_Loss(layers_weights=vgg_layer19)#net=vgg19
            self.criterionCharbonnier=CharbonnierLoss()
            self.criterionMSP=MultiscalePixelLoss()
            #------------
#             resnet_feature_extractor = get_resnet50_feature_extractor(torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu'))
#             vgg19_feature_extractor = get_vgg19_feature_extractor(torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu'))
            resnext_feature_extractor=get_resnext_feature_extractor(torch.device("cuda:0"))
            self.style_loss = StyleLoss_(resnext_feature_extractor)
            
            #------------
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        lambda_D_adv=self.opt.lambda_D_adv
        lambda_l1_D=self.opt.lambda_l1_D
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #Content loss/L1 Loss between fake_B and real_B and fake_A and real_A
        if lambda_l1_D>0:
            self.L1_D=self.criterionL1_D(fake,real)*lambda_l1_D

        else:
            self.L1_D=0
            
        
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * lambda_D_adv + self.L1_D
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_style=self.opt.lambda_style
        lambda_ltr=self.opt.lambda_ltr
        lambda_tv=self.opt.lambda_tv
        lambda_ganA=self.opt.lambda_ganA
        lambda_ganB=self.opt.lambda_ganB
        lambda_MSE=self.opt.lambda_MSE
        lambda_style_rec=self.opt.lambda_style_rec
        lambda_l1=self.opt.lambda_l1
        lambda_contextual_vgg19=self.opt.lambda_con_vgg_19
        lambda_charbonnier=self.opt.lambda_charbonnier
        lambda_MSP=self.opt.lambda_MSP
        
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        if lambda_ganA>0:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)*lambda_ganA
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            
        # GAN loss D_B(G_B(B))
        if lambda_ganB>0:
                    self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)*lambda_ganB
        else:
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        #Style Loss
        if lambda_style>0:
            self.loss_style_A=self.style_loss(self.fake_A,self.real_A) * lambda_style
            self.loss_style_B=self.style_loss(self.fake_B,self.real_B) * lambda_style
        else:
            self.loss_style_A=0
            self.loss_style_B=0

#         if lambda_style>0:
#             self.loss_style_A=self.criterionStyle(self.fake_A,self.real_A)*lambda_style
#             self.loss_style_B=self.criterionStyle(self.fake_B,self.real_B)*lambda_style
#         else:
#             self.loss_style_A=0
#             self.loss_style_B=0
        
        
        #Contextual Loss
        if lambda_contextual_vgg19>0:
            self.loss_con_A=self.criterionContextual_vgg_19(self.fake_A,self.real_A)*lambda_contextual_vgg19
            self.loss_con_B=self.criterionContextual_vgg_19(self.fake_B,self.real_B)*lambda_contextual_vgg19
        else:
            self.loss_con_A=0
            self.loss_con_B=0
        
        
        #Style Reconstruction Loss
        if lambda_style_rec>0:
            self.loss_style_rec_A=get_style_reconstruction_loss(self.real_A,self.fake_A,torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu'))
            self.loss_style_rec_B=get_style_reconstruction_loss(self.real_B,self.fake_B,torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu'))
        else:
            self.loss_style_rec_A=0
            self.loss_style_rec_B=0
        
        # Texture_Regularization_Loss 
        #Features 
        #device = torch.device("cpu")
        if lambda_ltr > 0:
            self.ltr_A = self.netG_A(self.real_B)
            self.loss_ltr_A = torch.nn.functional.mse_loss(self.ltr_A, self.real_B) * lambda_ltr
            
            self.ltr_B = self.netG_B(self.real_A)
            self.loss_ltr_B = torch.nn.functional.mse_loss(self.ltr_B, self.real_A) * lambda_ltr
        else:
            self.loss_ltr_A = 0
            self.loss_ltr_B = 0
        
        
        if lambda_trl>0:
            #device = torch.device("cpu")
            a = self.real_A.cpu().detach().numpy()
            fakeA = self.fake_A.cpu().detach().numpy()
            LoG_A = torch.from_numpy(nd.gaussian_laplace(a , 2))
            LoG_PA = torch.from_numpy(nd.gaussian_laplace(fakeA, 2))
            self.tdr_A = torch.mean((LoG_A - LoG_PA )** 2)

            B = self.real_B.cpu().detach().numpy()
            fakeB = self.fake_B.cpu().detach().numpy()
            LoG_B = torch.from_numpy(nd.gaussian_laplace(B, 2))
            LoG_PB = torch.from_numpy(nd.gaussian_laplace(fakeB, 2))
            self.tdr_B = torch.mean((LoG_B - LoG_PB )** 2)
            self.tot_trl_loss= lambda_trl * (self.tdr_A + self.tdr_B)
        else:
            self.tot_trl_loss=0
        
        #Total Variation Loss
        if lambda_tv > 0:
            
            self.tv_B = self.netG_A(self.fake_B)
            self.loss_tv_B = self.criterionTv(self.tv_B, self.real_B) * lambda_tv
            self.tv_A = self.netG_B(self.fake_A)
            self.loss_tv_A = self.criterionTv(self.tv_A, self.real_A) * lambda_tv
            
        else: 
            self.loss_tv_A = 0
            self.loss_tv_B = 0
        
        
        #MSE Loss/Content Loss between fake_B and real_B and fake_A and real_A
        if lambda_MSE>0:
            self.loss_MSE_A=self.criterionMSE(self.fake_A,self.real_A)*lambda_MSE
            self.loss_MSE_B=self.criterionMSE(self.fake_B,self.real_B)*lambda_MSE
        else:
            self.loss_MSE_A=0
            self.loss_MSE_B=0
        
        #Content loss/L1 Loss between fake_B and real_B and fake_A and real_A
        if lambda_l1>0:
            self.loss_L1_A=self.criterionL1(self.fake_A,self.real_A)*lambda_l1
            self.loss_L1_B=self.criterionL1(self.fake_B,self.real_B)*lambda_l1
        else:
            self.loss_L1_A=0
            self.loss_L1_B=0
        #Charbonnier Loss (L1 Loss with sqrt)
        if lambda_charbonnier>0:
            self.loss_charbonnier_A=self.criterionCharbonnier(self.fake_A,self.real_A)*lambda_charbonnier
            self.loss_charbonnier_B=self.criterionCharbonnier(self.fake_B,self.real_B)*lambda_charbonnier
        else:
            self.loss_charbonnier_A=0
            self.loss_charbonnier_B=0
        
        #Multiscale Pixel Loss
        if lambda_MSP>0:
            self.loss_MSP_A=self.criterionMSP(self.fake_A,self.real_A)
            self.loss_MSP_B=self.criterionMSP(self.fake_B,self.real_B)
        else:
            self.loss_MSP_A=0
            self.loss_MSP_B=0
        
        # combined loss and calculate gradients
        self.loss_G = self.loss_MSP_A + self.loss_MSP_B+ self.loss_charbonnier_A + self.loss_charbonnier_B + self.loss_con_A+self.loss_con_B+self.loss_ltr_A + self.loss_ltr_B + self.loss_L1_A + self.loss_L1_B + self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_style_A + self.loss_style_B  + self.loss_tv_B + self.loss_tv_A + self.loss_MSE_A + self.loss_MSE_B + self.loss_style_rec_A + self.loss_style_rec_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


# -

class RelativeL1(nn.Module):
    '''
    Comparing to the regular L1, introducing the division by |c|+epsilon
    better models the human vision system’s sensitivity to variations
    in the dark areas. (where epsilon = 0.01, to prevent values of 0 in the
    denominator)
    '''
    def __init__(self, eps=.01, reduction='mean'):
        super().__init__()
        self.criterion = torch.nn.L1Loss(reduction=reduction)
        self.eps = eps

    def forward(self, fake, target):
        base = target + self.eps
        return self.criterion(fake/base, target/base)


class MultiscalePixelLoss(nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss(), scale = 5):
        super(MultiscalePixelLoss, self).__init__()
        self.criterion = loss_f
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)
        self.weights = [1, 0.5, 0.25, 0.125, 0.125]
        self.weights = self.weights[:scale]

    def forward(self, fake, target, mask=None):
        loss = 0
        if mask is not None:
            mask = mask.expand(-1, fake.size()[1], -1, -1)
        for i in range(len(self.weights)):
            if mask is not None:
                loss += self.weights[i] * self.criterion(fake * mask, target * mask)
            else:
                loss += self.weights[i] * self.criterion(fake, target)
            if i != len(self.weights) - 1:
                fake = self.downsample(fake)
                target = self.downsample(target)
                if mask is not None:
                    mask = self.downsample(mask)
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)


# +
# class VGG16(torch.nn.Module):
#     def __init__(self,device="cuda:0"):
#         super().__init__()

#         vgg16 = models.vgg16(weights='DEFAULT').to(device)
#         features = nn.Sequential(*list(vgg16.children()))
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()

#         for x in range(4):
#             self.slice1.add_module(str(x), features[x])
#         for x in range(4, 9):
#             self.slice2.add_module(str(x), features[x])
#         for x in range(9, 16):
#             self.slice3.add_module(str(x), features[x])
#         for x in range(16, 23):
#             self.slice4.add_module(str(x), features[x])

#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         h = self.slice1(x)
#         h_relu1_2 = h
#         h = self.slice2(h)
#         h_relu2_2 = h
#         h = self.slice3(h)
#         h_relu3_3 = h
#         h = self.slice4(h)
#         h_relu4_3 = h

#         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
#         out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
#         return out

def gram_matrix(y):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class StyleLoss(nn.Module):
    def __init__(self,device):
        super().__init__()

        self.add_module('vgg', VGG_Model(vgg_layer19))
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        style_loss = 0.0
        for x_feat, y_feat in zip(x_vgg, y_vgg):
            style_loss += self.criterion(gram_matrix(x_feat), gram_matrix(y_feat))

        return style_loss


# -

class VGG_Model(nn.Module):
    """
        A VGG model with listerners in the layers.
        Will return a dictionary of outputs that correspond to the
        layers set in "listen_list".
    """
    def __init__(self, listen_list=None, net='vgg19', use_input_norm=True, z_norm=False,device=torch.device("cuda:0")):
        super(VGG_Model, self).__init__()
        #vgg = vgg16(pretrained=True)
        if net == 'vgg19':
            vgg_net = vgg.vgg19(weights='DEFAULT').to(device)
            vgg_layer = vgg_layer19
            self.vgg_layer_inv = vgg_layer_inv19
        elif net == 'vgg16':
            vgg_net = vgg.vgg16(weights='DEFAULT').to(device)
            vgg_layer = vgg_layer16
            self.vgg_layer_inv = vgg_layer_inv16
        self.vgg_model = vgg_net.features
        self.use_input_norm = use_input_norm
        # image normalization
        if self.use_input_norm:
            if z_norm: # if input in range [-1,1]
                mean = torch.tensor(
                    [[[0.485-1]], [[0.456-1]], [[0.406-1]]], requires_grad=False).to(device)
                std = torch.tensor(
                    [[[0.229*2]], [[0.224*2]], [[0.225*2]]], requires_grad=False).to(device)
            else: # input in range [0,1]
                mean = torch.tensor(
                    [[[0.485]], [[0.456]], [[0.406]]], requires_grad=False).to(device)
                std = torch.tensor(
                    [[[0.229]], [[0.224]], [[0.225]]], requires_grad=False).to(device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        vgg_dict = vgg_net.state_dict()
        vgg_f_dict = self.vgg_model.state_dict()
        vgg_dict = {k: v for k, v in vgg_dict.items() if k in vgg_f_dict}
        vgg_f_dict.update(vgg_dict)
        # no grad
        for p in self.vgg_model.parameters():
            p.requires_grad = False
        if listen_list == []:
            self.listen = []
        else:
            self.listen = set()
            for layer in listen_list:
                self.listen.add(vgg_layer[layer])
        self.features = OrderedDict()

    def forward(self, x):
        if self.use_input_norm:
            
            x = (x - self.mean.detach()) / self.std.detach()

        for index, layer in enumerate(self.vgg_model):
            x = layer(x)
            if index in self.listen:
                self.features[self.vgg_layer_inv[index]] = x
        return self.features


# +
DIS_TYPES = ['cosine', 'l1', 'l2']

class Contextual_Loss(nn.Module):
    '''
    Contextual loss for unaligned images (https://arxiv.org/abs/1803.02077)

    https://github.com/roimehrez/contextualLoss
    https://github.com/S-aiueo32/contextual_loss_pytorch
    https://github.com/z-bingo/Contextual-Loss-PyTorch

    layers_weights: is a dict, e.g., {'conv_1_1': 1.0, 'conv_3_2': 1.0}
    crop_quarter: boolean
    '''
    def __init__(self, layers_weights, crop_quarter=False, max_1d_size=100,
            distance_type: str = 'cosine', b=1.0, band_width=0.5,
            use_vgg: bool = True, net: str = 'vgg19', calc_type: str =  'regular'):
        super(Contextual_Loss, self).__init__()

        assert band_width > 0, 'band_width parameter must be positive.'
        assert distance_type in DIS_TYPES,\
            f'select a distance type from {DIS_TYPES}.'

        listen_list = []
        self.layers_weights = {}
        try:
            listen_list = layers_weights.keys()
            self.layers_weights = layers_weights
        except:
            pass

        self.crop_quarter = crop_quarter
        self.distanceType = distance_type
        self.max_1d_size = max_1d_size
        self.b = b
        self.band_width = band_width #self.h = h, #sigma

        if use_vgg:
            self.vgg_model = VGG_Model(listen_list=listen_list, net=net)

        if calc_type == 'bilateral':
            self.calculate_loss = self.bilateral_CX_Loss
        elif calc_type == 'symetric':
            self.calculate_loss = self.symetric_CX_Loss
        else: #if calc_type == 'regular':
            self.calculate_loss = self.calculate_CX_Loss

    def forward(self, images, gt):
        device = images.device

        if hasattr(self, 'vgg_model'):
            assert images.shape[1] == 3 and gt.shape[1] == 3,\
                'VGG model takes 3 channel images.'

            loss = 0
            vgg_images = self.vgg_model(images)
            vgg_images = {k: v.clone().to(device) for k, v in vgg_images.items()}
            vgg_gt = self.vgg_model(gt)
            vgg_gt = {k: v.to(device) for k, v in vgg_gt.items()}

            for key in self.layers_weights.keys():
                if self.crop_quarter:
                    vgg_images[key] = self._crop_quarters(vgg_images[key])
                    vgg_gt[key] = self._crop_quarters(vgg_gt[key])

                N, C, H, W = vgg_images[key].size()
                if H*W > self.max_1d_size**2:
                    vgg_images[key] = self._random_pooling(vgg_images[key], output_1d_size=self.max_1d_size)
                    vgg_gt[key] = self._random_pooling(vgg_gt[key], output_1d_size=self.max_1d_size)

                loss_t = self.calculate_loss(vgg_images[key], vgg_gt[key])
                loss += loss_t * self.layers_weights[key]
                # del vgg_images[key], vgg_gt[key]
        #TODO: without VGG it runs, but results are not looking right
        else:
            if self.crop_quarter:
                images = self._crop_quarters(images)
                gt = self._crop_quarters(gt)

            N, C, H, W = images.size()
            if H*W > self.max_1d_size**2:
                images = self._random_pooling(images, output_1d_size=self.max_1d_size)
                gt = self._random_pooling(gt, output_1d_size=self.max_1d_size)

            loss = self.calculate_loss(images, gt)
        return loss

    @staticmethod
    def _random_sampling(tensor, n, indices):
        N, C, H, W = tensor.size()
        S = H * W
        tensor = tensor.view(N, C, S)
        device=tensor.device
        if indices is None:
            indices = torch.randperm(S)[:n].contiguous().type_as(tensor).long()
            indices = indices.clamp(indices.min(), tensor.shape[-1]-1) #max = indices.max()-1
            indices = indices.view(1, 1, -1).expand(N, C, -1)
        indices = indices.to(device)

        res = torch.gather(tensor, index=indices, dim=-1)
        return res, indices

    @staticmethod
    def _random_pooling(feats, output_1d_size=100):
        single_input = type(feats) is torch.Tensor

        if single_input:
            feats = [feats]

        N, C, H, W = feats[0].size()
        feats_sample, indices = Contextual_Loss._random_sampling(feats[0], output_1d_size**2, None)
        res = [feats_sample]

        for i in range(1, len(feats)):
            feats_sample, _ = Contextual_Loss._random_sampling(feats[i], -1, indices)
            res.append(feats_sample)

        res = [feats_sample.view(N, C, output_1d_size, output_1d_size) for feats_sample in res]

        if single_input:
            return res[0]
        return res

    @staticmethod
    def _crop_quarters(feature_tensor):
        N, fC, fH, fW = feature_tensor.size()
        quarters_list = []
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., 0:round(fH / 2), round(fW / 2):])
        quarters_list.append(feature_tensor[..., round(fH / 2):, 0:round(fW / 2)])
        quarters_list.append(feature_tensor[..., round(fH / 2):, round(fW / 2):])

        feature_tensor = torch.cat(quarters_list, dim=0)
        return feature_tensor

    @staticmethod
    def _create_using_L2(I_features, T_features):
        """
        Calculating the distance between each feature of I and T
        :param I_features:
        :param T_features:
        :return: raw_distance: [N, C, H, W, H*W], each element of which is the distance between I and T at each position
        """
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)
        #
        square_I = torch.sum(Ivecs*Ivecs, dim=1, keepdim=False)
        square_T = torch.sum(Tvecs*Tvecs, dim=1, keepdim=False)
        # raw_distance
        raw_distance = []
        for i in range(N):
            Ivec, Tvec, s_I, s_T = Ivecs[i, ...], Tvecs[i, ...], square_I[i, ...], square_T[i, ...]
            # matrix multiplication
            AB = Ivec.permute(1, 0) @ Tvec
            dist = s_I.view(-1, 1) + s_T.view(1, -1) - 2*AB
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        raw_distance = torch.clamp(raw_distance, 0.0)
        return raw_distance

    @staticmethod
    def _create_using_L1(I_features, T_features):
        assert I_features.size() == T_features.size()
        N, C, H, W = I_features.size()

        Ivecs = I_features.view(N, C, -1)
        Tvecs = T_features.view(N, C, -1)

        raw_distance = []
        for i in range(N):
            Ivec, Tvec = Ivecs[i, ...], Tvecs[i, ...]
            dist = torch.sum(
                torch.abs(Ivec.view(C, -1, 1) - Tvec.view(C, 1, -1)), dim=0, keepdim=False
            )
            raw_distance.append(dist.view(1, H, W, H*W))
        raw_distance = torch.cat(raw_distance, dim=0)
        return raw_distance

    @staticmethod
    def _create_using_dotP(I_features, T_features):
        assert I_features.size() == T_features.size()
        # prepare feature before calculating cosine distance
        # mean shifting by channel-wise mean of `y`.
        mean_T = T_features.mean(dim=(0, 2, 3), keepdim=True)
        I_features = I_features - mean_T
        T_features = T_features - mean_T

        # L2 channelwise normalization
        I_features = F.normalize(I_features, p=2, dim=1)
        T_features = F.normalize(T_features, p=2, dim=1)

        N, C, H, W = I_features.size()
        cosine_dist = []
        # work seperatly for each example in dim 1
        for i in range(N):
            # channel-wise vectorization
            T_features_i = T_features[i].view(1, 1, C, H*W).permute(3, 2, 0, 1).contiguous() # 1CHW --> 11CP, with P=H*W
            I_features_i = I_features[i].unsqueeze(0)
            dist = F.conv2d(I_features_i, T_features_i).permute(0, 2, 3, 1).contiguous()
            #cosine_dist.append(dist) # back to 1CHW
            #TODO: temporary hack to workaround AMP bug:
            cosine_dist.append(dist.to(torch.float32)) # back to 1CHW
        cosine_dist = torch.cat(cosine_dist, dim=0)
        cosine_dist = (1 - cosine_dist) / 2
        cosine_dist = cosine_dist.clamp(min=0.0)

        return cosine_dist

    #compute_relative_distance
    @staticmethod
    def _calculate_relative_distance(raw_distance, epsilon=1e-5):
        """
        Normalizing the distances first as Eq. (2) in paper
        :param raw_distance:
        :param epsilon:
        :return:
        """
        div = torch.min(raw_distance, dim=-1, keepdim=True)[0]
        relative_dist = raw_distance / (div + epsilon) # Eq 2
        return relative_dist

    def symetric_CX_Loss(self, I_features, T_features):
        loss = (self.calculate_CX_Loss(T_features, I_features) + self.calculate_CX_Loss(I_features, T_features)) / 2
        return loss #score

    def bilateral_CX_Loss(self, I_features, T_features, weight_sp: float = 0.1):
        def compute_meshgrid(shape):
            N, C, H, W = shape
            rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
            cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

            feature_grid = torch.meshgrid(rows, cols)
            feature_grid = torch.stack(feature_grid).unsqueeze(0)
            feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

            return feature_grid

        # spatial loss
        grid = compute_meshgrid(I_features.shape).to(T_features.device)
        raw_distance = Contextual_Loss._create_using_L2(grid, grid) # calculate raw distance
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_sp = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # feature loss
        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        dist_tilde = Contextual_Loss._calculate_relative_distance(raw_distance)
        exp_distance = torch.exp((self.b - dist_tilde) / self.band_width) # Eq(3)
        cx_feat = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)

        # combined loss
        cx_combine = (1. - weight_sp) * cx_feat + weight_sp * cx_sp
        k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)
        cx = k_max_NC.mean(dim=1)
        cx_loss = torch.mean(-torch.log(cx + 1e-5))
        return cx_loss

    def calculate_CX_Loss(self, I_features, T_features):
        device = I_features.device
        T_features = T_features.to(device)

        if torch.sum(torch.isnan(I_features)) == torch.numel(I_features) or torch.sum(torch.isinf(I_features)) == torch.numel(I_features):
            print(I_features)
            raise ValueError('NaN or Inf in I_features')
        if torch.sum(torch.isnan(T_features)) == torch.numel(T_features) or torch.sum(
                torch.isinf(T_features)) == torch.numel(T_features):
            print(T_features)
            raise ValueError('NaN or Inf in T_features')

        # calculate raw distances
        if self.distanceType == 'l1':
            raw_distance = Contextual_Loss._create_using_L1(I_features, T_features)
        elif self.distanceType == 'l2':
            raw_distance = Contextual_Loss._create_using_L2(I_features, T_features)
        else: # self.distanceType == 'cosine':
            raw_distance = Contextual_Loss._create_using_dotP(I_features, T_features)
        if torch.sum(torch.isnan(raw_distance)) == torch.numel(raw_distance) or torch.sum(
                torch.isinf(raw_distance)) == torch.numel(raw_distance):
            print(raw_distance)
            raise ValueError('NaN or Inf in raw_distance')

        # normalizing the distances
        relative_distance = Contextual_Loss._calculate_relative_distance(raw_distance)
        if torch.sum(torch.isnan(relative_distance)) == torch.numel(relative_distance) or torch.sum(
                torch.isinf(relative_distance)) == torch.numel(relative_distance):
            print(relative_distance)
            raise ValueError('NaN or Inf in relative_distance')
        del raw_distance

        #compute_sim()
        # where h>0 is a band-width parameter
        exp_distance = torch.exp((self.b - relative_distance) / self.band_width) # Eq(3)
        if torch.sum(torch.isnan(exp_distance)) == torch.numel(exp_distance) or torch.sum(
                torch.isinf(exp_distance)) == torch.numel(exp_distance):
            print(exp_distance)
            raise ValueError('NaN or Inf in exp_distance')
        del relative_distance

        # Similarity
        contextual_sim = exp_distance / torch.sum(exp_distance, dim=-1, keepdim=True) # Eq(4)
        if torch.sum(torch.isnan(contextual_sim)) == torch.numel(contextual_sim) or torch.sum(
                torch.isinf(contextual_sim)) == torch.numel(contextual_sim):
            print(contextual_sim)
            raise ValueError('NaN or Inf in contextual_sim')
        del exp_distance

        #contextual_loss()
        max_gt_sim = torch.max(torch.max(contextual_sim, dim=1)[0], dim=1)[0] # Eq(1)
        del contextual_sim
        CS = torch.mean(max_gt_sim, dim=1)
        CX_loss = torch.mean(-torch.log(CS)) # Eq(5)
        if torch.isnan(CX_loss):
            raise ValueError('NaN in computing CX_loss')
        return CX_loss

# +
# class CycleGANModel(BaseModel):
#     """
#     This class implements the CycleGAN model, for learning image-to-image translation without paired data.

#     The model training requires '--dataset_mode unaligned' dataset.
#     By default, it uses a '--netG resnet_9blocks' ResNet generator,
#     a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
#     and a least-square GANs objective ('--gan_mode lsgan').

#     CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
#     """
#     @staticmethod
#     def modify_commandline_options(parser, is_train=True):
#         """Add new dataset-specific options, and rewrite default values for existing options.

#         Parameters:
#             parser          -- original option parser
#             is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

#         Returns:
#             the modified parser.

#         For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
#         A (source domain), B (target domain).
#         Generators: G_A: A -> B; G_B: B -> A.
#         Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
#         Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
#         Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
#         Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
#         Dropout is not used in the original CycleGAN paper.
#         """
#         parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
#         if is_train:
#             parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
#             parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
#             parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

#         return parser

#     def __init__(self, opt):
#         """Initialize the CycleGAN class.

#         Parameters:
#             opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
#         """
#         BaseModel.__init__(self, opt)
#         # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
#         self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
#         # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
#         visual_names_A = ['real_A', 'fake_B', 'rec_A']
#         visual_names_B = ['real_B', 'fake_A', 'rec_B']
#         if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
#             visual_names_A.append('idt_B')
#             visual_names_B.append('idt_A')

#         self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
#         # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
#         if self.isTrain:
#             self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
#         else:  # during test time, only load Gs
#             self.model_names = ['G_A', 'G_B']

#         # define networks (both Generators and discriminators)
#         # The naming is different from those used in the paper.
#         # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
#         self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
#         self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
#                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

#         if self.isTrain:  # define discriminators
#             self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
#                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
#             self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
#                                             opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

#         if self.isTrain:
#             if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
#                 assert(opt.input_nc == opt.output_nc)
#             self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
#             self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
#             # define loss functions
#             self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
#             self.criterionCycle = torch.nn.L1Loss()
#             self.criterionIdt = torch.nn.L1Loss()
#             # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
#             self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#             self.optimizers.append(self.optimizer_G)
#             self.optimizers.append(self.optimizer_D)

#     def set_input(self, input):
#         """Unpack input data from the dataloader and perform necessary pre-processing steps.

#         Parameters:
#             input (dict): include the data itself and its metadata information.

#         The option 'direction' can be used to swap domain A and domain B.
#         """
#         AtoB = self.opt.direction == 'AtoB'
#         self.real_A = input['A' if AtoB else 'B'].to(self.device)
#         self.real_B = input['B' if AtoB else 'A'].to(self.device)
#         self.image_paths = input['A_paths' if AtoB else 'B_paths']

#     def forward(self):
#         """Run forward pass; called by both functions <optimize_parameters> and <test>."""
#         self.fake_B = self.netG_A(self.real_A)  # G_A(A)
#         self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
#         self.fake_A = self.netG_B(self.real_B)  # G_B(B)
#         self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

#     def backward_D_basic(self, netD, real, fake):
#         """Calculate GAN loss for the discriminator

#         Parameters:
#             netD (network)      -- the discriminator D
#             real (tensor array) -- real images
#             fake (tensor array) -- images generated by a generator

#         Return the discriminator loss.
#         We also call loss_D.backward() to calculate the gradients.
#         """
#         # Real
#         pred_real = netD(real)
#         loss_D_real = self.criterionGAN(pred_real, True)
#         # Fake
#         pred_fake = netD(fake.detach())
#         loss_D_fake = self.criterionGAN(pred_fake, False)
#         # Combined loss and calculate gradients
#         loss_D = (loss_D_real + loss_D_fake) * 0.5
#         loss_D.backward()
#         return loss_D

#     def backward_D_A(self):
#         """Calculate GAN loss for discriminator D_A"""
#         fake_B = self.fake_B_pool.query(self.fake_B)
#         self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

#     def backward_D_B(self):
#         """Calculate GAN loss for discriminator D_B"""
#         fake_A = self.fake_A_pool.query(self.fake_A)
#         self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

#     def backward_G(self):
#         """Calculate the loss for generators G_A and G_B"""
#         lambda_idt = self.opt.lambda_identity
#         lambda_A = self.opt.lambda_A
#         lambda_B = self.opt.lambda_B
#         # Identity loss
#         if lambda_idt > 0:
#             # G_A should be identity if real_B is fed: ||G_A(B) - B||
#             self.idt_A = self.netG_A(self.real_B)
#             self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
#             # G_B should be identity if real_A is fed: ||G_B(A) - A||
#             self.idt_B = self.netG_B(self.real_A)
#             self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
#         else:
#             self.loss_idt_A = 0
#             self.loss_idt_B = 0

#         # GAN loss D_A(G_A(A))
#         self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
#         # GAN loss D_B(G_B(B))
#         self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
#         # Forward cycle loss || G_B(G_A(A)) - A||
#         self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
#         # Backward cycle loss || G_A(G_B(B)) - B||
#         self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
#         # combined loss and calculate gradients
#         self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
#         self.loss_G.backward()

#     def optimize_parameters(self):
#         """Calculate losses, gradients, and update network weights; called in every training iteration"""
#         # forward
#         self.forward()      # compute fake images and reconstruction images.
#         # G_A and G_B
#         self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
#         self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
#         self.backward_G()             # calculate gradients for G_A and G_B
#         self.optimizer_G.step()       # update G_A and G_B's weights
#         # D_A and D_B
#         self.set_requires_grad([self.netD_A, self.netD_B], True)
#         self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
#         self.backward_D_A()      # calculate gradients for D_A
#         self.backward_D_B()      # calculate graidents for D_B
#         self.optimizer_D.step()  # update D_A and D_B's weights
