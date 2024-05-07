# Densenet decoder encoder with intermediate fully connected layers and dropout

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.autograd import gradcheck
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck
from torch.autograd import Function
import numpy as np


def add_coordConv_channels(t):
    n,c,h,w=t.size()
    xx_channel=np.ones((h, w))
    xx_range=np.array(range(h))
    xx_range=np.expand_dims(xx_range,-1)
    xx_coord=xx_channel*xx_range
    yy_coord=xx_coord.transpose()

    xx_coord=xx_coord/(h-1)
    yy_coord=yy_coord/(h-1)
    xx_coord=xx_coord*2 - 1
    yy_coord=yy_coord*2 - 1
    xx_coord=torch.from_numpy(xx_coord).float()
    yy_coord=torch.from_numpy(yy_coord).float()

    if t.is_cuda:
    	xx_coord=xx_coord.cuda()
    	yy_coord=yy_coord.cuda()

    xx_coord=xx_coord.unsqueeze(0).unsqueeze(0).repeat(n,1,1,1)
    yy_coord=yy_coord.unsqueeze(0).unsqueeze(0).repeat(n,1,1,1)

    t_cc=torch.cat((t,xx_coord,yy_coord),dim=1)

    return t_cc



class DenseBlockEncoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockEncoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers     = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.Conv2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

# Dense block in encoder.
class DenseBlockDecoder(nn.Module):
    def __init__(self, n_channels, n_convs, activation=nn.ReLU, args=[False]):
        super(DenseBlockDecoder, self).__init__()
        assert(n_convs > 0)

        self.n_channels = n_channels
        self.n_convs    = n_convs
        self.layers = nn.ModuleList()
        for i in range(n_convs):
            self.layers.append(nn.Sequential(
                    nn.BatchNorm2d(n_channels),
                    activation(*args),
                    nn.ConvTranspose2d(n_channels, n_channels, 3, stride=1, padding=1, bias=False),))

    def forward(self, inputs):
        outputs = []

        for i, layer in enumerate(self.layers):
            if i > 0:
                next_output = 0
                for no in outputs:
                    next_output = next_output + no
                outputs.append(next_output)
            else:
                outputs.append(layer(inputs))
        return outputs[-1]

class DenseTransitionBlockEncoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, mp, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockEncoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.mp             = mp
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.Conv2d(n_channels_in, n_channels_out, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(mp),
        )
    def forward(self, inputs):
        # print(inputs.shape,'222222222222222',self.main(inputs).shape)
        return self.main(inputs)


class DenseTransitionBlockDecoder(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, activation=nn.ReLU, args=[False]):
        super(DenseTransitionBlockDecoder, self).__init__()
        self.n_channels_in  = n_channels_in
        self.n_channels_out = n_channels_out
        self.main           = nn.Sequential(
                nn.BatchNorm2d(n_channels_in),
                activation(*args),
                nn.ConvTranspose2d(n_channels_in, n_channels_out, 4, stride=2, padding=1, bias=False),
        )
    def forward(self, inputs):
        # print(inputs.shape,'333333333333',self.main(inputs).shape)
        return self.main(inputs)

## Dense encoders and decoders for image of size 128 128
class waspDenseEncoder128(nn.Module):
    def __init__(self, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Tanh, f_args=[]):
        super(waspDenseEncoder128, self).__init__()
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.BatchNorm2d(nc),
                nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 64 x 64
                DenseBlockEncoder(ndf, 6),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args),

                # state size. (ndf*2) x 32 x 32
                DenseBlockEncoder(ndf*2, 12),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args),

                # state size. (ndf*4) x 16 x 16
                DenseBlockEncoder(ndf*4, 16),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*4) x 8 x 8
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 4 x 4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        input=add_coordConv_channels(input)
        output = self.main(input).view(-1,self.ndim)
        #print(output.size())
        return output

class waspDenseDecoder128(nn.Module):
    def __init__(self, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Hardtanh, f_args=[]):
        super(waspDenseDecoder128, self).__init__()
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*8),

            # state size. (ngf*4) x 8 x 8
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*4),

            # state size. (ngf*2) x 16 x 16
            DenseBlockDecoder(ngf*4, 12),
            DenseTransitionBlockDecoder(ngf*4, ngf*2),

            # state size. (ngf) x 32 x 32
            DenseBlockDecoder(ngf*2, 6),
            DenseTransitionBlockDecoder(ngf*2, ngf),

            # state size. (ngf) x 64 x 64
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 128 x 128
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
        # self.smooth=nn.Sequential(
        #     nn.Conv2d(nc, nc, 1, stride=1, padding=0, bias=False),
        #     f_activation(*f_args),
        # )
    def forward(self, inputs):
        # return self.smooth(self.main(inputs))
        return self.main(inputs)



## Dense encoders and decoders for image of size 512 512
class waspDenseEncoder512(nn.Module):
    def __init__(self, nc=1, ndf = 32, ndim = 128, activation=nn.LeakyReLU, args=[0.2, False], f_activation=nn.Tanh, f_args=[]):
        super(waspDenseEncoder512, self).__init__()
        self.ndim = ndim

        self.main = nn.Sequential(
                # input is (nc) x 128 x 128  > *4
                nn.BatchNorm2d(nc),
                nn.ReLU(True),
                nn.Conv2d(nc, ndf, 4, stride=2, padding=1),

                # state size. (ndf) x 64 x 64  > *4
                DenseBlockEncoder(ndf, 6),
                DenseTransitionBlockEncoder(ndf, ndf*2, 2, activation=activation, args=args),

                # state size. (ndf*2) x 32 x 32 > *4
                DenseBlockEncoder(ndf*2, 12),
                DenseTransitionBlockEncoder(ndf*2, ndf*4, 2, activation=activation, args=args),

                # state size. (ndf*4) x 16 x 16  > *4
                DenseBlockEncoder(ndf*4, 16),
                DenseTransitionBlockEncoder(ndf*4, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 8 x 8   *4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndf*8, 2, activation=activation, args=args),

                # state size. (ndf*8) x 4 x 4 > *4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndf*8, 4, activation=activation, args=args),
                f_activation(*f_args),

                # state size. (ndf*8) x 2 x 2 > *4
                DenseBlockEncoder(ndf*8, 16),
                DenseTransitionBlockEncoder(ndf*8, ndim, 4, activation=activation, args=args),
                f_activation(*f_args),
        )

    def forward(self, input):
        input=add_coordConv_channels(input)
        output = self.main(input).view(-1,self.ndim)
        # output = self.main(input).view(8,-1)
        # print(input.shape,'---------------------')
        #print(output.size())
        return output

class waspDenseDecoder512(nn.Module):
    def __init__(self, nz=128, nc=1, ngf=32, lb=0, ub=1, activation=nn.ReLU, args=[False], f_activation=nn.Tanh, f_args=[]):
        super(waspDenseDecoder512, self).__init__()
        self.main   = nn.Sequential(
            # input is Z, going into convolution
            nn.BatchNorm2d(nz),
            activation(*args),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),

            # state size. (ngf*8) x 4 x 4
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*8),

            # state size. (ngf*8) x 8 x 8
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*8),

            # state size. (ngf*4) x 16 x 16
            DenseBlockDecoder(ngf*8, 16),
            DenseTransitionBlockDecoder(ngf*8, ngf*4),

            # state size. (ngf*2) x 32 x 32
            DenseBlockDecoder(ngf*4, 12),
            DenseTransitionBlockDecoder(ngf*4, ngf*2),

            # state size. (ngf) x 64 x 64
            DenseBlockDecoder(ngf*2, 6),
            DenseTransitionBlockDecoder(ngf*2, ngf),

            # state size. (ngf) x 128 x 128
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size. (ngf) x 256 x 256
            DenseBlockDecoder(ngf, 6),
            DenseTransitionBlockDecoder(ngf, ngf),

            # state size (ngf) x 512 x 512
            nn.BatchNorm2d(ngf),
            activation(*args),
            nn.ConvTranspose2d(ngf, nc, 3, stride=1, padding=1, bias=False),
            f_activation(*f_args),
        )
        # self.smooth=nn.Sequential(
        #     nn.Conv2d(nc, nc, 1, stride=1, padding=0, bias=False),
        #     f_activation(*f_args),
        # )
    def forward(self, inputs):
        # return self.smooth(self.main(inputs))
        return self.main(inputs)


class dnetccnl(nn.Module):
    #in_channels -> nc      | encoder first layer
    #filters -> ndf    | encoder first layer
    #img_size(h,w) -> ndim
    #out_channels  -> optical flow (x,y)

    def __init__(self, img_size=448, in_channels=3, out_channels=2, filters=32,fc_units=100):
        super(dnetccnl, self).__init__()
        self.nc=in_channels
        self.nf=filters
        self.ndim=img_size
        self.oc=out_channels
        self.fcu=fc_units

        self.encoder=waspDenseEncoder128(nc=self.nc+2,ndf=self.nf,ndim=self.ndim)
        self.decoder=waspDenseDecoder128(nz=self.ndim,nc=self.oc,ngf=self.nf)
        # self.fc_layers= nn.Sequential(nn.Linear(self.ndim, self.fcu),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               nn.Linear(self.fcu,self.ndim),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               )

    def forward(self, inputs):

        encoded=self.encoder(inputs)
        encoded=encoded.unsqueeze(-1).unsqueeze(-1)
        decoded=self.decoder(encoded)
        # print torch.max(decoded)
        # print torch.min(decoded)
        # print(decoded.shape,'11111111111111111',encoded.shape)

        return decoded

class dnetccnl512(nn.Module):
    #in_channels -> nc      | encoder first layer
    #filters -> ndf    | encoder first layer
    #img_size(h,w) -> ndim
    #out_channels  -> optical flow (x,y)

    def __init__(self, img_size=448, in_channels=3, out_channels=2, filters=32,fc_units=100):
        super(dnetccnl512, self).__init__()
        self.nc=in_channels
        self.nf=filters
        self.ndim=img_size
        self.oc=out_channels
        self.fcu=fc_units

        self.encoder=waspDenseEncoder512(nc=self.nc+2,ndf=self.nf,ndim=self.ndim)
        self.decoder=waspDenseDecoder512(nz=self.ndim,nc=self.oc,ngf=self.nf)
        # self.fc_layers= nn.Sequential(nn.Linear(self.ndim, self.fcu),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               nn.Linear(self.fcu,self.ndim),
        #                               nn.ReLU(True),
        #                               nn.Dropout(0.25),
        #                               )

    def forward(self, inputs):

        encoded=self.encoder(inputs)
        encoded=encoded.unsqueeze(-1).unsqueeze(-1)
        decoded=self.decoder(encoded)
        # print torch.max(decoded)
        # print torch.min(decoded)
        # print(decoded.shape,'11111111111111111',encoded.shape)

        return decoded