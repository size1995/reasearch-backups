import mxnet as mx

use_gpu = True
ctx = mx.cpu()
from attention_block import self_attention_block
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout



            
class Res_Block(HybridBlock):
    def __init__(self, outer_channels, use_bias=False):
        super(Res_Block, self).__init__()
        with self.name_scope():
            conv1 = Conv2D(channels=outer_channels, kernel_size=3, strides=1, padding=1,
                           in_channels=outer_channels, use_bias=use_bias)    
            norm1 = BatchNorm(momentum=0.1, in_channels=outer_channels)
            relu1 = LeakyReLU(alpha=0.2)
            
            conv2 = Conv2D(channels=outer_channels, kernel_size=3, strides=1, padding=1,
                           in_channels=outer_channels, use_bias=use_bias)
            norm2 = BatchNorm(momentum=0.1, in_channels=outer_channels)
            relu2 = LeakyReLU(alpha=0.2)

            conv3 = Conv2D(channels=outer_channels, kernel_size=3, strides=1, padding=1,
                           in_channels=outer_channels, use_bias=use_bias)
            norm3 = BatchNorm(momentum=0.1, in_channels=outer_channels)
            relu3 = LeakyReLU(alpha=0.2)


            res_block = [conv1, norm1, relu1, conv2, norm2, relu2,conv3, norm3, relu3]

            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(outer_channels // 16, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(outer_channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
            
            self.res = HybridSequential()
            with self.res.name_scope():
                for block in res_block:
                    self.res.add(block)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.res(x)
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.se(w).reshape(shape=(0,0,1,1))
        
             
        x = x + residual
        return x,w


class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False, use_attention=True, use_resblock=True):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            self.innermost = innermost
            self.use_attention = use_attention
            if not self.outermost:
                res_block_1 = Res_Block(outer_channels=outer_channels)
                res_block_2 = Res_Block(outer_channels=inner_channels)
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)

            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                res_block_3 = Res_Block(outer_channels=inner_channels)
                res_block_4 = Res_Block(outer_channels=outer_channels)
                if use_resblock:
                    res1 = res_block_1
                    encoder = [en_relu, en_conv]
                    res2 = res_block_2
                    res3 = res_block_3
                    decoder = [de_relu, de_conv, de_norm]
                    res4 = res_block_4
                else:
                    encoder = [en_relu, en_conv]
                    decoder = [de_relu, de_conv, de_norm]



            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels)
                channel_trans=Conv2D(channels=1,in_channels=outer_channels,kernel_size=1, prefix='')

                if use_resblock:
                    res1 = None
                    encoder = [en_conv]
                    res2 = None
                    res3 = None
                    decoder = [de_relu, de_conv, de_norm,channel_trans]
                    res4 = None
                else:
                    encoder = [en_conv]
                    decoder = [de_relu, de_conv, de_norm,channel_trans]


            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                res_block_3 = Res_Block(outer_channels=inner_channels)
                res_block_4 = Res_Block(outer_channels=outer_channels)

                if use_resblock:
                    res1 = res_block_1
                    encoder = [en_relu, en_conv, en_norm]
                    res2 = res_block_2
                    res3 = res_block_3
                    decoder = [de_relu, de_conv, de_norm]
                    res4 = res_block_4
                else:
                    encoder = [en_relu, en_conv, en_norm]
                    decoder = [de_relu, de_conv, de_norm]


            if use_dropout:
                decoder += [Dropout(rate=0.5)]

            self.encoder = HybridSequential()
            with self.encoder.name_scope():
                for block in encoder:
                    self.encoder.add(block)

            self.inner_block = inner_block

            self.res1=res1
            self.res2=res2
            self.res3=res3
            self.res4=res4

            self.decoder = HybridSequential()
            with self.decoder.name_scope():
                for block in decoder:
                    self.decoder.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            x1=x
            y1= [F.zeros(shape=(1, 1, 1, 1))]
            x2 = self.encoder(x1)
            x3=x2
            y2= [F.zeros(shape=(1, 1, 1, 1))]
            x4, y_en, y_de = self.inner_block(x3)
            x5 = x4
            y3=[F.zeros(shape=(1, 1, 1, 1))]
            x6 = self.decoder(x5)
            x7=x6
            y4=[F.zeros(shape=(1, 1, 1, 1))]
            y_en=y_en+[[y1,y2]]
            y_de=y_de+[[y3,y4]]
            return x7, y_en, y_de
        if self.innermost:
            x1,y1=self.res1(x)
            x2 = self.encoder(x1)
            x3,y2 = self.res2(x2)
            x4 = x3
            y_en = [[y1,y2]]
            x5,y3 = self.res3(x4)
            x6=self.decoder(x5)
            x7,y4 = self.res4(x6)
            y_de= [[y3,y4]]
        else:
            x1,y1=self.res1(x)
            x2 = self.encoder(x1)
            x3,y2 = self.res2(x2)
            x4, y_en, y_de = self.inner_block(x3)
            x5, y3 = self.res3(x4)
            x6 = self.decoder(x5)
            x7,y4 = self.res4(x6)
            y_en=y_en+[[y1,y2]]
            y_de=y_de+[[y3,y4]]


        return x7+x, y_en,y_de


class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, ngf=32, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet1 = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True, use_attention=True, use_resblock=True)
        unet2 = UnetSkipUnit(ngf * 8, ngf * 8, unet1, use_dropout=use_dropout, use_attention=True, use_resblock=True)

        unet3 = UnetSkipUnit(ngf * 8, ngf * 4, unet2, use_attention=True, use_resblock=True)
        unet4 = UnetSkipUnit(ngf * 4, ngf * 2, unet3, use_attention=True, use_resblock=True)
        unet5 = UnetSkipUnit(ngf * 2, ngf * 1, unet4, use_attention=True, use_resblock=True)
        unet6 = UnetSkipUnit(ngf, in_channels, unet5, outermost=True, use_resblock=True)
        self.model = unet6

    def hybrid_forward(self, F, x):
        x1, ye,yd = self.model(x)
        return x1, ye,yd


def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))


def network_init(net):
    for param in net.collect_params().values():
        param_init(param)


def set_network():
    net = UnetGenerator(in_channels=3)
    net.initialize(init='Xavier')

    return net
