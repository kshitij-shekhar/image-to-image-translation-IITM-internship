import functools
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from .cbam_modules import *

class ResnetGenerator_new_cbam_down_only(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    
    This uses cbam in the main Generator(like supriti's model), but after the conv layer with the kernel size of 7,
    not before it. Cbam here is used only after the initial conv layer and in the downsampling layers. Both times
    it's used directly after a conv layer.
    
    No cbam is used in the Residual Block.
    
    """

    def __init__(self, input_nc, output_nc, use_attn=True,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_new_cbam_down_only, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.ca1 = channel_attention(ngf)
        self.sa1 = spatial_attention()
        
        self.ca2 = channel_attention(ngf*2)
        self.sa2 = spatial_attention()
        
        self.ca3 = channel_attention(ngf*4)
        self.sa3 = spatial_attention()
        if use_attn:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    self.ca1,
                    self.sa1,
                    norm_layer(ngf),
                    nn.ReLU(True)
                    #self.ca1,
                    ]
        else:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)
                    ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      self.ca2 if mult==1 else self.ca3,
                      self.sa2 if mult==1 else self.sa3,
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      #self.ca1,
#                       self.sa1,
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

# +
# class ResnetGenerator_new_cbam_down_only_modified(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    
#     This uses cbam in the main Generator(like supriti's model), but after the conv layer with the kernel size of 7,
#     not before it. Cbam here is used only after the initial conv layer and in the downsampling layers. Both times
#     it's used directly after a conv layer.
    
#     No cbam is used in the Residual Block.
    
#     """

#     def __init__(self, input_nc, output_nc, use_attn=True,ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
#         """Construct a Resnet-based generator

#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert(n_blocks >= 0)
#         super(ResnetGenerator_new_cbam_down_only, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm2d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm2d

#         self.ca1 = channel_attention(ngf)
#         self.sa1 = spatial_attention()
        
#         self.ca2 = channel_attention(ngf*2)
#         self.sa2 = spatial_attention()
        
#         self.ca3 = channel_attention(ngf*4)
#         self.sa3 = spatial_attention()
        
#         self.init_RPad=nn.ReflectionPad2d(3)
#         self.init_conv=nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)
#         self.init_norm=norm_layer(ngf)
#         self.init_relu=nn.ReLU(True)
        
#         self.down_conv1=nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)
#         self.down_norm2=norm_layer(ngf*2)
#         self.down_relu=nn.ReLU(True)
        
#         self.down_conv2=nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias)
#         self.down_norm2=norm_layer(ngf*4)
#         self.down_relu2=nn.ReLU(True)
        
#         self.resblock1=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock2=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock3=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock4=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock5=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock6=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock7=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock8=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
#         self.resblock9=ResnetBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
#         self.pathology_f=Pathology_block(ngf*7, ngf*4, 4)
        
#         self.up_conv1=nn.ConvTranspose2d(ngf * 4, int(ngf * 4 / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)
#         self.up_norm1=norm_layer(int(ngf * 4 / 2))  
#         self.up_relu1=nn.ReLU(True)
        
#         self.up_conv2=nn.ConvTranspose2d(ngf * 2, int(ngf * 2 / 2),kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias)
#         self.up_norm2=norm_layer(int(ngf*2/2))
#         self.up_relu2=nn.ReLU(True)
        
#         self.final_RPad=nn.ReflectionPad2d(3)
#         self.final_conv=nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)
#         self.tanh=nn.Tanh()
        
#         if use_attn:
#             model = [nn.ReflectionPad2d(3),
#                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                     self.ca1,
#                     self.sa1,
#                     norm_layer(ngf),
#                     nn.ReLU(True)
#                     #self.ca1,
#                     ]
        

#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i #1, 2
#             model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                       self.ca2 if mult==1 else self.ca3,
#                       self.sa2 if mult==1 else self.sa3,
#                       norm_layer(ngf * mult * 2),
#                       nn.ReLU(True)]

#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):       # add ResNet blocks

#             model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i) #4, 2
#             model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                          kernel_size=3, stride=2,
#                                          padding=1, output_padding=1,
#                                          bias=use_bias),
#                       #self.ca1,
# #                       self.sa1,
#                       norm_layer(int(ngf * mult / 2)),
#                       nn.ReLU(True)]
#         model += [nn.ReflectionPad2d(3)]
#         model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]

#         self.model = nn.Sequential(*model)

#     def forward(self, input):
        
    
        


# +
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


# -

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


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


# +
class ResnetGenerator_cbam(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, use_attn=False,ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',cbam_placement="first"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_cbam, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

#         self.ca1 = ChannelAttention(input_nc)
#         self.sa1 = SpatialAttention()
        if use_attn:
            model = [nn.ReflectionPad2d(3),
                    self.ca1,
                    self.sa1,
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)
                    #self.ca1,
                    ]
        else:
            model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                    norm_layer(ngf),
                    nn.ReLU(True)
                    ]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            if cbam_placement=="first":
                model += [ResnetBlock_with_cbam(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            elif cbam_placement=="second":
                model += [ResnetBlock_with_cbam_2(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            elif cbam_placement=="third":
                model += [ResnetBlock_with_cbam_3(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            elif cbam_placement=="fourth":
                model += [ResnetBlock_with_cbam_4(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


# -

class ResnetBlock_with_cbam_4(nn.Module):
    """No channel attention in the either cbam placement : only spatial attention"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_with_cbam_4, self).__init__()
#         self.ca=channel_attention(dim)
        self.sa=spatial_attention()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        #add cbam
#         conv_block += [self.ca]
        conv_block += [self.sa]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        #add cbam
#         conv_block += [self.ca]
        conv_block += [self.sa]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

# +
# class PixelAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(PixelAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
        
#         # Project features to query, key, and value
#         query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (N, H*W, C/2)
#         key = self.key_conv(x).view(batch_size, -1, height * width)  # (N, C/2, H*W)
#         value = self.value_conv(x).view(batch_size, -1, height * width)  # (N, C, H*W)
        
#         # Compute attention scores
#         energy = torch.bmm(query, key)  # (N, H*W, H*W)
#         attention = torch.softmax(energy, dim=-1)  # row-wise softmax
        
#         # Apply attention to values
#         attentioned_value = torch.bmm(value, attention.permute(0, 2, 1))  # (N, C, H*W)
#         attentioned_value = attentioned_value.view(batch_size, channels, height, width)  # (N, C, H, W)
        
#         # Apply residual connection and scaling
#         out = self.gamma * attentioned_value + x
#         return out


class PixelAttention(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y


# -

class ContentEncoder(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,32,kernel_size=7)
        self.ca1=ChannelAttention(32)
        self.pa1=PixelAttention(32)
        self.conv2=nn.Conv2d(1,32)
