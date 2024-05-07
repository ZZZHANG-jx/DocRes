from math import log
import torch
import torch.nn as nn
from torch.nn import init
import functools
from model.cbam import CBAM
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0,stride=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down_single(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up_single(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = SingleConv(in_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=4, stride=2,padding=1, bias=True)
    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        # input is BCHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DoubleConv(in_channels, out_channels)
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=4, stride=2,padding=1, bias=True)
    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        # input is BCHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.hardtanh = nn.Hardtanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1):
        x = self.conv(x1)
        # x = self.sigmoid(x)
        # x = self.hardtanh(x)
        # x = (x+1)/2
        return x
class GiemaskGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(GiemaskGenerator, self).__init__()
        self.init_channel =32
        self.inc = DoubleConv(3,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, 1)
        self.up1_1 = Up_single(self.init_channel*32, self.init_channel*16)
        self.up2_1 = Up_single(self.init_channel*16, self.init_channel*8)
        self.up3_1 = Up_single(self.init_channel*8, self.init_channel*4)
        self.up4_1 = Up_single(self.init_channel*4,self.init_channel*2)
        self.up5_1 = Up_single(self.init_channel*2, self.init_channel)
        self.outc_1 = OutConv(self.init_channel, 1)
  #      self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)


        x_1 = self.up1_1(x6, x5)
        x_1 = self.up2_1(x_1, x4)
        x_1 = self.up3_1(x_1, x3)
        x_1 = self.up4_1(x_1, x2)
        x_1 = self.up5_1(x_1, x1)
        mask = self.outc_1(x_1)

        x = self.up1(x6, x5)
#        x = self.dropout(x)
        x = self.up2(x, x4)
#        x = self.dropout(x)
        x = self.up3(x, x3)
#        x = self.dropout(x)
        x = self.up4(x, x2)
#        x = self.dropout(x)
        x = self.up5(x, x1)
#        x = self.dropout(x)
        depth = self.outc(x)
        return depth,mask
    """Create a Unet-based generator"""
class Giemask2Generator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Giemask2Generator, self).__init__()
        self.init_channel =32
        self.inc = DoubleConv(3,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, 1)
        self.up1_1 = Up_single(self.init_channel*32, self.init_channel*16)
        self.up2_1 = Up_single(self.init_channel*16, self.init_channel*8)
        self.up3_1 = Up_single(self.init_channel*8, self.init_channel*4)
        self.up4_1 = Up_single(self.init_channel*4,self.init_channel*2)
        self.up5_1 = Up_single(self.init_channel*2, self.init_channel)
        self.outc_1 = OutConv(self.init_channel, 1)
        self.outc_2 = OutConv(self.init_channel, 1)
  #      self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)


        x_1 = self.up1_1(x6, x5)
        x_1 = self.up2_1(x_1, x4)
        x_1 = self.up3_1(x_1, x3)
        x_1 = self.up4_1(x_1, x2)
        x_1 = self.up5_1(x_1, x1)
        mask = self.outc_1(x_1)
        edge = self.outc_2(x_1)

        x = self.up1(x6, x5)
#        x = self.dropout(x)
        x = self.up2(x, x4)
#        x = self.dropout(x)
        x = self.up3(x, x3)
#        x = self.dropout(x)
        x = self.up4(x, x2)
#        x = self.dropout(x)
        x = self.up5(x, x1)
#        x = self.dropout(x)
        depth = self.outc(x)
        return depth,mask,edge
    """Create a Unet-based generator"""
class GieGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(GieGenerator, self).__init__()
        self.init_channel =32
        self.inc = DoubleConv(input_nc,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, 2)
  #      self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
#        x = self.dropout(x)
        x = self.up2(x, x4)
#        x = self.dropout(x)
        x = self.up3(x, x3)
#        x = self.dropout(x)
        x = self.up4(x, x2)
#        x = self.dropout(x)
        x = self.up5(x, x1)
#        x = self.dropout(x)
        logits1 = self.outc(x)
        return logits1


class GiecbamGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(GiecbamGenerator, self).__init__()
        self.init_channel =32
        self.inc = DoubleConv(input_nc,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)
        self.cbam = CBAM(gate_channels=self.init_channel*32)
        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, 2)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x6 = self.cbam(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.dropout(x)
        logits1 = self.outc(x)
        return logits1




class Gie2headGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Gie2headGenerator, self).__init__()
        self.init_channel =32
        self.inc = DoubleConv(input_nc,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1_1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2_1 = Up(self.init_channel*16, self.init_channel*8)
        self.up3_1 = Up(self.init_channel*8, self.init_channel*4)
        self.up4_1 = Up(self.init_channel*4,self.init_channel*2)
        self.up5_1 = Up(self.init_channel*2, self.init_channel)
        self.outc_1 = OutConv(self.init_channel, 1)

        self.up1_2 = Up(self.init_channel*32, self.init_channel*16)
        self.up2_2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3_2 = Up(self.init_channel*8, self.init_channel*4)
        self.up4_2 = Up(self.init_channel*4,self.init_channel*2)
        self.up5_2 = Up(self.init_channel*2, self.init_channel)
        self.outc_2 = OutConv(self.init_channel, 1)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x_1 = self.up1_1(x6, x5)
        x_1 = self.up2_1(x_1, x4)
        x_1 = self.up3_1(x_1, x3)
        x_1 = self.up4_1(x_1, x2)
        x_1 = self.up5_1(x_1, x1)
        logits_1 = self.outc_1(x_1)

        x_2 = self.up1_2(x6, x5)
        x_2 = self.up2_2(x_2, x4)
        x_2 = self.up3_2(x_2, x3)
        x_2 = self.up4_2(x_2, x2)
        x_2 = self.up5_2(x_2, x1)
        logits_2 = self.outc_2(x_2)

        logits = torch.cat((logits_1,logits_2),1)

        return logits



class BmpGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(BmpGenerator, self).__init__()
        self.init_channel =32
        self.output_nc = output_nc
        self.inc = DoubleConv(input_nc,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, self.output_nc)
  #      self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up1(x6, x5)
#        x = self.dropout(x)
        x = self.up2(x, x4)
#        x = self.dropout(x)
        x = self.up3(x, x3)
#        x = self.dropout(x)
        x = self.up4(x, x2)
#        x = self.dropout(x)
        x = self.up5(x, x1)
#        x = self.dropout(x)
        logits1 = self.outc(x)
        return logits1
class Bmp2Generator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Bmp2Generator, self).__init__()
        #gienet
        self.init_channel =32
        self.inc = DoubleConv(3,self.init_channel)
        self.down1 = Down(self.init_channel, self.init_channel*2)
        self.down2 = Down(self.init_channel*2, self.init_channel*4)
        self.down3 = Down(self.init_channel*4, self.init_channel*8)
        self.down4 = Down(self.init_channel*8, self.init_channel*16)
        self.down5 = Down(self.init_channel*16, self.init_channel*32)

        self.up1 = Up(self.init_channel*32, self.init_channel*16)
        self.up2 = Up(self.init_channel*16, self.init_channel*8)
        self.up3 = Up(self.init_channel*8, self.init_channel*4)
        self.up4 = Up(self.init_channel*4,self.init_channel*2)
        self.up5 = Up(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, 1)
        self.up1_1 = Up_single(self.init_channel*32, self.init_channel*16)
        self.up2_1 = Up_single(self.init_channel*16, self.init_channel*8)
        self.up3_1 = Up_single(self.init_channel*8, self.init_channel*4)
        self.up4_1 = Up_single(self.init_channel*4,self.init_channel*2)
        self.up5_1 = Up_single(self.init_channel*2, self.init_channel)
        self.outc_1 = OutConv(self.init_channel, 1)
        self.outc_2 = OutConv(self.init_channel, 1)

        #bpm net
        self.inc_b = DoubleConv(4,self.init_channel)
        self.down1_b = Down(self.init_channel, self.init_channel*2)
        self.down2_b = Down(self.init_channel*2, self.init_channel*4)
        self.down3_b = Down(self.init_channel*4, self.init_channel*8)
        self.down4_b = Down(self.init_channel*8, self.init_channel*16)
        self.down5_b = Down(self.init_channel*16, self.init_channel*32)

        self.up1_b = Up(self.init_channel*32, self.init_channel*16)
        self.up2_b = Up(self.init_channel*16, self.init_channel*8)
        self.up3_b = Up(self.init_channel*8, self.init_channel*4)
        self.up4_b = Up(self.init_channel*4,self.init_channel*2)
        self.up5_b = Up(self.init_channel*2, self.init_channel)
        self.outc_b = OutConv(self.init_channel, 2)
  #      self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        #gienet
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x_1 = self.up1_1(x6, x5)
        x_1 = self.up2_1(x_1, x4)
        x_1 = self.up3_1(x_1, x3)
        x_1 = self.up4_1(x_1, x2)
        x_1 = self.up5_1(x_1, x1)
        mask = self.outc_1(x_1)
        edge = self.outc_2(x_1)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        depth = self.outc(x)
        
        #bmpnet
        mask[mask>0.5]=1.
        mask[mask<=0.5]=0.
        image_cat_depth = torch.cat((input*mask,depth*mask),dim=1)
        x1_b = self.inc_b(image_cat_depth)
        x2_b = self.down1_b(x1_b)
        x3_b = self.down2_b(x2_b)
        x4_b = self.down3_b(x3_b)
        x5_b = self.down4_b(x4_b)
        x6_b = self.down5_b(x5_b)
        x_b = self.up1_b(x6_b, x5_b)
        x_b = self.up2_b(x_b, x4_b)
        x_b = self.up3_b(x_b, x3_b)
        x_b = self.up4_b(x_b, x2_b)
        x_b = self.up5_b(x_b, x1_b)
        bm = self.outc_b(x_b)
        # return depth,mask,edge,bm
        return bm
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

#class GieGenerator(nn.Module):
#    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                 norm_layer=nn.BatchNorm2d, use_dropout=False):
#        super(GieGenerator, self).__init__()
#
#        # construct unet structure
#        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
#        for i in range(num_downs - 5):
#            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
#        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
#        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
#
#        self.model = unet_block
#
#    def forward(self, input):
#        return self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
           # resize = nn.Upsample(scale_factor=2)
           # conv = nn.Conv2d(inner_nc,outer_nc,kernel_size=4,stride=2,padding=1,bias=use_bias)
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            #up = [uprelu, resize, conv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)



##===================================================================================================
class DilatedDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4,stride=1,dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4,stride=1,dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DilatedDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DilatedDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class DilatedUp(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = DilatedDoubleConv(in_channels, out_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4,stride=1,dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.deconv = nn.ConvTranspose2d(in_channels, out_channels,kernel_size=4, stride=2,padding=1, bias=True)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        # x1 = self.deconv(x1)
        # input is BCHW
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class DilatedSingleUnet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, biline=True, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DilatedSingleUnet, self).__init__()
        self.init_channel = 32
        self.inc = DilatedDoubleConv(input_nc,self.init_channel)
        self.down1 = DilatedDown(self.init_channel, self.init_channel*2)
        self.down2 = DilatedDown(self.init_channel*2, self.init_channel*4)
        self.down3 = DilatedDown(self.init_channel*4, self.init_channel*8)
        self.down4 = DilatedDown(self.init_channel*8, self.init_channel*16)
        self.down5 = DilatedDown(self.init_channel*16, self.init_channel*32)
        self.cbam = CBAM(gate_channels=self.init_channel*32)

        self.up1 = DilatedUp(self.init_channel*32, self.init_channel*16)
        self.up2 = DilatedUp(self.init_channel*16, self.init_channel*8)
        self.up3 = DilatedUp(self.init_channel*8, self.init_channel*4)
        self.up4 = DilatedUp(self.init_channel*4,self.init_channel*2)
        self.up5 = DilatedUp(self.init_channel*2, self.init_channel)
        self.outc = OutConv(self.init_channel, output_nc)
    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x6 = self.cbam(x6)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits1 = self.outc(x)
        return logits1