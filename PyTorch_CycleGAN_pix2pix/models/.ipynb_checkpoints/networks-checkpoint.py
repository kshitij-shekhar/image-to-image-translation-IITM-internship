import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler
#from transform_unet import TransUnet
#from models.unet_modules import UnetGenerator
# from torch.nn.utils.parametrizations import spectral_norm
# from resnet_modules import ResnetGenerator_new_cbam_down_only,ResnetGenerator_cbam
#from models.unet import UNet
#from smat_models.SmaAt_UNet import SmaAt_UNet
#from models.denseunet import DenseUnet

""
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=True, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    use_dropout = True

    
    if netG == "resnet_cbam_new_down": #cbam in main generator, after initial conv layer and in downsampling layers
        net=ResnetGenerator_new_cbam_down_only(input_nc, output_nc,use_attn=True, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9) 
    elif netG == "resnet_cbam_new_down_12_blocks": #cbam in main generator, after initial conv layer and in downsampling layers
        net=ResnetGenerator_new_cbam_down_only(input_nc, output_nc,use_attn=True, ngf=ngf, norm_layer=norm_layer, use_dropout=False, n_blocks=12)        
    elif netG == "resnet_cbam_9_blocks_fourth_placement": 
        net=ResnetGenerator_cbam(input_nc, output_nc,use_attn=False, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,cbam_placement="fourth")
    elif netG == 'resnet_cbam_9_blocks_different_cbam_placement': #different placement here means first placement
        net=ResnetGenerator_cbam(input_nc, output_nc,use_attn=False, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "unet_cbam_modified":
        net=Generator_unet_cls_cbam(input_nc,output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8)
    elif netG == "unet_PFA_3_layers_modified":
        net=Generator_unet_cls_PA(input_nc,output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8,PFA_layers=3)
    elif netG == "unet_PFA_4_layers_modified":
        net=Generator_unet_cls_PA(input_nc,output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8,PFA_layers=4)
    elif netG == "unet_PFA_5_layers_modified":
        net=Generator_unet_cls_PA(input_nc,output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8,PFA_layers=5)
    elif netG == "resnet_dsff": 
        net=ResNet_PFA(BasicBlock,input_nc, output_nc, ngf=0, norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout, n_blocks=0)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
        
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier no cbam
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_4_layers':  # default PatchGAN classifier no cbam
        net = NLayerDiscriminator(input_nc, ndf, n_layers=4, norm_layer=norm_layer)
    elif netD == 'basic_5_layers':  # default PatchGAN classifier no cbam
        net = NLayerDiscriminator(input_nc, ndf, n_layers=5, norm_layer=norm_layer)
    elif netD == 'pixel':     # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)

    return init_net(net, init_type, init_gain, gpu_ids)

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

##############################################################################
# Classes
# #############################################################################

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def get_target_tensor_with_type(self, prediction, target_is_real, image_type):
        if image_type == 'train_0':
            target_tensor = torch.tensor(0.0).to('cuda:0')
        elif image_type == 'train_1':
            target_tensor = torch.tensor(1.0).to('cuda:0')
        elif image_type == 'train_2':
            target_tensor = torch.tensor(2.0).to('cuda:0')
        elif image_type == 'train_3':
            target_tensor = torch.tensor(3.0).to('cuda:0')
        else:
            raise ValueError('Unsupported image type: %s' % image_type)
            
        if target_is_real:
            return target_tensor.expand_as(prediction)
        else:
            return -target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real,image_type=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if image_type==None:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            return loss
        else:
            if self.gan_mode in ['lsgan', 'vanilla']:
                target_tensor = self.get_target_tensor_with_type(prediction, target_is_real, image_type)
                loss = self.loss(prediction, target_tensor)
            elif self.gan_mode == 'wgangp':
                if target_is_real:
                    loss = -prediction.mean()
                else:
                    loss = prediction.mean()
            return loss
     
        
        
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)



""
class PixelDiscriminator(nn.Module):


    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


""
class Down(nn.Module):
    """Downscaling with conv with stride=2,instanceNorm, relu"""

    def __init__(self, in_features, out_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_features, out_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        # upsample
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.gate = nn.Sequential(
            nn.Conv2d(out_features, out_features//2, 1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        self.merge = nn.Sequential(
            nn.Conv2d(out_features+out_features//2, out_features, 1),
            nn.InstanceNorm2d(out_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

    def forward(self, x1, x2):

        x1 = self.up_conv(x1)

        x2 = self.gate(x2)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)

        return self.merge(x)


""
class ResidualBlock(nn.Module):
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Pathology_block(nn.Module):
    def __init__(self, in_features, out_features, n_residual_blocks, alt_leak=False, neg_slope=1e-2):
        super(Pathology_block, self).__init__()

        ext_model = [nn.Conv2d(in_features, out_features, 1),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]
        ext_model += [nn.ReflectionPad2d(1),
                       nn.Conv2d(out_features, out_features, 4, stride=2),
                       nn.InstanceNorm2d(out_features),
                       nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True)]

        for _ in range(n_residual_blocks):
            ext_model += [ResidualBlock(out_features, alt_leak, neg_slope)]
        self.extractor = nn.Sequential(*ext_model)

    def forward(self, x1, x2, x3):

        x1 = F.interpolate(x1, scale_factor=0.5)
        diffY1 = x2.size()[2] - x1.size()[2]
        diffX1 = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])
        x = torch.cat([x1, x2], dim=1)

        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        diffY2 = x2.size()[2] - x3.size()[2]
        diffX2 = x2.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])
        x = torch.cat([x, x3], dim=1)

        return self.extractor(x)


""
class channel_attention(nn.Module):
    def __init__(self,ch,ratio=8):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(
            nn.Linear(ch,ch//8,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//8,ch,bias=False)
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x1=self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1=self.mlp(x1)
        x2=self.max_pool(x).squeeze(-1).squeeze(-1)
        x2=self.mlp(x2)

        feats=x1+x2
        feats=self.sigmoid(feats).unsqueeze(-1).unsqueeze(-1)
        refined_feats=x*feats

        return refined_feats


""
class spatial_attention(nn.Module):
    def __init__(self,kernel=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel,padding=3,bias=False)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x1=torch.mean(x,dim=1,keepdim=True)#squishes channels to 1 channel consisting of the mean of all the channels
        x2=torch.max(x,dim=1,keepdim=True)[0]
        feats=torch.cat([x1,x2],dim=1)
        # print(feats.shape)
        feats=self.conv(feats)
        feats=self.sigmoid(feats)
        refined_feats=x*feats
        return refined_feats


""
class Generator_unet_cls_cbam(nn.Module):
    """Cbam in the downsampling layers(3)"""
    
    def __init__(self, input_nc, output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8, n_residual_blocks=8, alt_leak=False, neg_slope=1e-2):
        super(Generator_unet_cls, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        self.ca1=channel_attention(64)
        self.sa1=spatial_attention()
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        self.ca2=channel_attention(128)
        self.sa2=spatial_attention()
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)
        self.ca3=channel_attention(256)
        self.sa3=spatial_attention()

        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)
        # merge features [N 256 H/8 W/8]
        self.pathology_f = Pathology_block(448, 256, n_residual_blocks // 2, alt_leak, neg_slope)

        self.merge = nn.Sequential(nn.Conv2d(512, 256, 1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        
        
        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)
        # Upsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        # Upsampling [N 64 H/2 W/2]
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        # Upsampling [N 32 H W]
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        # Upsampling [N 3 H W]
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

#         self.out_cls = nn.Sequential(nn.Dropout(0.15),
#                                      nn.ReflectionPad2d(1),
#                                      nn.Conv2d(256, 1, 3),
#                                      nn.Sigmoid())

    def forward(self, x, mode='G'):
        # encoder
        x0 = self.inc(x)
        x1 = self.sa1(self.ca1(self.down1(x0)))
        x2 = self.sa2(self.ca2(self.down2(x1)))
        x3 = self.sa3(self.ca3(self.down3(x2)))

        # extract feature
        pathology_features = self.pathology_f(x1, x2, x3)
#         c_out = self.out_cls(pathology_features)
#         # Average pooling and flatten
#         c_out = F.avg_pool2d(c_out, c_out.size()[2:]).view(c_out.size()[0])
#         if mode == 'C':
#             return c_out
        latent_features = self.ext_f1(x3)
        features = torch.cat([latent_features, pathology_features], dim=1)
        features = self.merge(features)
        features = self.ext_f2(features)

        # decoder
        x = self.up1(features, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        outputs = self.outc(x)
        return outputs, latent_features, pathology_features

""
class Generator_unet_cls_PA(nn.Module):
    """ PFA blocks in the upsampling layers"""
    def __init__(self, input_nc, output_nc,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8, n_residual_blocks=8, alt_leak=False, neg_slope=1e-2,PFA_layers=1):
        super(Generator_unet_cls_PA, self).__init__()
        # Initial convolution block [N 32 H W]
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 32, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Downsampling [N 64 H/2 W/2]
        self.down1 = Down(32, 64, alt_leak, neg_slope)
        # Downsampling [N 128 H/4 W/4]
        self.down2 = Down(64, 128, alt_leak, neg_slope)
        # Downsampling [N 256 H/8 W/8]
        self.down3 = Down(128, 256, alt_leak, neg_slope)
        self.PFA_layers=PFA_layers
        # Residual blocks [N 256 H/8 W/8]
        res_ext_encoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_encoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f1 = nn.Sequential(*res_ext_encoder)
        # merge features [N 256 H/8 W/8]
        self.pathology_f = Pathology_block(448, 256, n_residual_blocks // 2, alt_leak, neg_slope)

        self.merge = nn.Sequential(nn.Conv2d(512, 256, 1),
                                   nn.InstanceNorm2d(256),
                                   nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        # Residual blocks [N 256 H/8 W/8]
        res_ext_decoder = []
        for _ in range(n_residual_blocks // 2):
            res_ext_decoder += [ResidualBlock(256, alt_leak, neg_slope)]
        self.ext_f2 = nn.Sequential(*res_ext_decoder)
        
        self.PFA_1=PA_Block(256,reduction=8)
        
        # Upsampling [N 128 H/4 W/4]
        self.up1 = Up(256, 128, alt_leak, neg_slope)
        if self.PFA_layers==2:
            self.PFA_2=PA_Block(128,reduction=8)
            
        # Upsampling [N 64 H/2 W/2]
        
        self.up2 = Up(128, 64, alt_leak, neg_slope)
        if self.PFA_layers==3:
            self.PFA_3=PA_Block(64,reduction=8)
            
        # Upsampling [N 32 H W]
        
        self.up3 = Up(64, 32, alt_leak, neg_slope)
        if self.PFA_layers==4:
            self.PFA_4=PA_Block(32,reduction=8)
            
        # Upsampling [N 3 H W]
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(32, output_nc, 7),
                                  nn.Tanh())

#         self.out_cls = nn.Sequential(nn.Dropout(0.15),
#                                      nn.ReflectionPad2d(1),
#                                      nn.Conv2d(256, 1, 3),
#                                      nn.Sigmoid())

    def forward(self, x, mode='G'):
        # encoder
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        # extract feature
        pathology_features = self.pathology_f(x1, x2, x3)
#         c_out = self.out_cls(pathology_features)
#         # Average pooling and flatten
#         c_out = F.avg_pool2d(c_out, c_out.size()[2:]).view(c_out.size()[0])
#         if mode == 'C':
#             return c_out
        latent_features = self.ext_f1(x3)
        features = torch.cat([latent_features, pathology_features], dim=1)
        features = self.merge(features)
        features = self.ext_f2(features)

        # decoder
        x=self.PFA_1(features)
        x = self.up1(features, x2)
        if self.PFA_layers==2:
            x=self.PFA_2(x)
        x = self.up2(x, x1)
        if self.PFA_layers==3:
            x=self.PFA_3(x)
        x = self.up3(x, x0)
        if self.PFA_layers==4:
            x=self.PFA_4(x)
        
        outputs = self.outc(x)
        return outputs, latent_features, pathology_features


""
class PA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(PA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.InstanceNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(2, 1, 3, padding=1, bias=False)

    def forward(self, x):
        _, c, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_c = torch.mean(x, dim=1, keepdim=True)
        x_c_max, _ = torch.max(x, dim=1, keepdim=True)
        x_c = torch.cat([x_c, x_c_max], dim=1)
        x_c = self.conv1(x_c)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)*self.sigmoid(x_c)
        return out

""
class BasicBlock(nn.Module):

    def __init__(self, in_channel=3, out_channel=3, stride=1, downsample=None, upsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 =nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False))
        self.bn1 = nn.InstanceNorm2d(out_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, bias=False))
        self.bn2 = nn.InstanceNorm2d(out_channel)
        self.downsample = downsample
        self.upsample = upsample

    def forward(self, x):
        identity = x
        if self.upsample is not None:
            x = self.upsample(x)
            identity = self.downsample(identity)
        if self.downsample is not None and self.upsample is None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


""
class ResNet_PFA(nn.Module):
    """From DSFF GAN Paper"""

    def __init__(self,
                 block=BasicBlock,input_nc=3,output_nc=3,ngf=0,norm_layer=0,use_dropout=False,n_blocks=8):
        super(ResNet_PFA, self).__init__()
        self.inc = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(input_nc, 16, 7),
                                 nn.InstanceNorm2d(32),
                                 nn.LeakyReLU(0.2, inplace=True) )


        # if bilinear, use the normal convolutions to reduce the number of channels
        self.out_seg_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 32, 3),
                                     nn.InstanceNorm2d(32),
                                     nn.LeakyReLU(0.2, inplace=True),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(32, 1, 3),
                                    nn.Sigmoid()
                                     )
        self.out_seg = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(1, 32, 3),
                                     nn.InstanceNorm2d(32),
                                     nn.LeakyReLU(0.2, inplace=True))


        
        self.layer1 = self._make_layer(block, 16,32, 0, stride=2)
        self.layer2 = self._make_layer(block, 32,64,0, stride=2)
        self.layer3 = self._make_layer(block, 64,128,0, stride=2)
        self.layer4 = self._make_layer(block, 128,128, 1, stride=1)
        self.layer9 = self._make_layer(block, 32, 32, 1, stride=1)
        self.layer8 = self._make_layer(block, 160, 128, 3, stride=1)
        self.layer5 = self._make_layer(block, 128,64, 0, stride=2,up=True)
        self.layer6 = self._make_layer(block, 64, 32, 0, stride=2,up=True)
        self.layer7 = self._make_layer(block, 32, 16, 0, stride=2,up=True)
        self.outc = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(16,output_nc , 7),
                                  nn.Tanh())
        self.merge1=merge(64)
        self.merge2 = merge(32)
        self.merge3 = merge(16)

        self.conv1=nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.Conv2d(128,64,3,1),
                            nn.InstanceNorm2d(64),
                            nn.LeakyReLU(0.2, inplace=True)
                                 )
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(64, 32, 3, 1),
                                   nn.InstanceNorm2d(32),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )

        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(32, 16, 3, 1),
                                   nn.InstanceNorm2d(16),
                                   nn.LeakyReLU(0.2, inplace=True)
                                   )
        self.drop=nn.Dropout(0.2)
    def _make_layer(self, block, in_channel,channel, block_num, stride=1,up=False,i=0):
        downsample = None
        upsample=None
        if  up==False :
            if stride!=1:
                downsample = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, channel , kernel_size=3, stride=2, bias=False),
                    nn.InstanceNorm2d(channel ),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            else:
                downsample = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channel, channel , kernel_size=3, stride=1, bias=False),
                    nn.InstanceNorm2d(channel ),
                    nn.LeakyReLU(0.2, inplace=True))
            layers = []
            
            layers.append(block(in_channel,channel,downsample=downsample,stride=stride,upsample=upsample))
                            
            
            for i in range( block_num):
                layers.append(block(channel, channel))
        if  up==True:
            upsample=nn.Sequential(nn.ConvTranspose2d(in_channel, in_channel, 3, stride=2, padding=1, output_padding=1),
                                   nn.InstanceNorm2d(in_channel),
                                   nn.LeakyReLU(0.2, inplace=True))
            downsample = nn.Sequential(
                nn.ConvTranspose2d(in_channel, channel, 3, stride, padding=1, output_padding=1),
                nn.InstanceNorm2d(channel),
            nn.LeakyReLU(0.2, inplace=True))
            layers = []
            layers.append(block(in_channel,
                                channel,
                                downsample=downsample,
                                upsample=upsample))
            for _ in range(i, block_num):
                layers.append(block(channel, channel ))
        return nn.Sequential(*layers)

    def forward(self, x,mode='G'):
        x = self.inc(x)  # [1,16,256,256]
        x1 = self.layer1(x)  # [1,32,128,128]
        x2 = self.layer2(x1)  # [1,64,64,64]
        x3 = self.layer3(x2)  # [1,128,32,32]]
        label_out=self.out_seg_conv(x3)
        if mode=='C':
            return label_out
        label_features=self.out_seg(label_out)
        features=self.layer4(x3)
        label_features=self.layer9(label_features)
        features=torch.cat((label_features,features),1)
        features=self.layer8(features)
        features=self.drop(features)
        x2 = self.merge1(x2, features)#[1,64,64,64]
        x4 = self.layer5(features)  # [1,64,64,64]
        x2=torch.cat((x2,x4),1)#[1,128,64,64]
        x2=self.conv1(x2)#[1,128,32,32]
        x1 = self.merge2(x1, x2)
        x4 = self.layer6(x2)  # [1,64,64,64]
        x1=torch.cat((x1,x4),1)
        x1=self.conv2(x1)
        x = self.merge3(x, x1)
        x4 = self.layer7(x1)  # [1,32,128,128]
        x=torch.cat((x,x4),1)
        x = self.conv3(x)
        x = self.outc(x)
        return x, features,label_out



class merge(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_features, alt_leak=False, neg_slope=1e-2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.gate = nn.Sequential(

            nn.Conv2d(in_features, in_features, 1),
            nn.InstanceNorm2d(in_features),
            nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))
        self.up= nn.Sequential(nn.ConvTranspose2d(in_features*2, in_features, 3, stride=2, padding=1, output_padding=1),
                               nn.InstanceNorm2d(in_features),
                               nn.LeakyReLU(neg_slope, inplace=True) if alt_leak else nn.ReLU(inplace=True))

        self.attention=PA_Block(in_features)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x1, x2):
        x1 = self.gate(x1)
        x3=self.up(x2)
        x2 = self.attention(self.relu(x1+x3))
        return x2

""
# class UnetGenerator_modified(nn.Module):
#     """Create a Unet-based generator"""

#     def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet generator
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
#                                 image of size 128x128 will become of size 1x1 # at the bottleneck
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer

#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
#         super(UnetGenerator, self).__init__()
#         self.down_conv_1=nn.Conv2d(input_nc,)
        
        
        
# #         # construct unet structure
# #         unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
# #         for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
# #             unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
# #         # gradually reduce the number of filters from ngf * 8 to ngf
# #         self.bottleneck_features=unet_block
# #         unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
# #         unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
# #         unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
# #         self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input),self.bottleneck_features
    
    
# class UnetSkipConnectionBlock(nn.Module):
#     """Defines the Unet submodule with skip connection.
#         X -------------------identity----------------------
#         |-- downsampling -- |submodule| -- upsampling --|
#     """

#     def __init__(self, outer_nc, inner_nc, input_nc=None,
#                  submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
#         """Construct a Unet submodule with skip connections.

#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)

#         if outermost:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
            
#             down = [downconv]
#             up = [uprelu, upconv, nn.Tanh()]
#             model = down + [submodule] + up
#         elif innermost:
#             upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv,upnorm]
#             model = down + up
#         else:
#             upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]

#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         if self.outermost:
#             return self.model(x)
#         else:   # add skip connections
#             return torch.cat([x, self.model(x)], 1)
