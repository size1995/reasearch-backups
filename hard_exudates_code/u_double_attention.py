import mxnet as mx
epochs = 100
batch_size = 10

use_gpu = True
ctx = mx.cpu()

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from attention_block import self_attention_block,CA_M,CA_M1,CA_M2,CA_M3
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np


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


            res_block = [conv1, norm1, relu1, conv2, norm2, relu2, conv3, norm3, relu3]
          
            self.res = HybridSequential()
            with self.res.name_scope():
                for block in res_block:
                    self.res.add(block)

    def hybrid_forward(self, F, x):
        residual = x
        x = self.res(x)
        x = x + residual
        return x

class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False,use_position_attention=False,use_channel_attention=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost

            res1=Res_Block(outer_channels=outer_channels)
            res2=Res_Block(outer_channels=inner_channels)
            res3=Res_Block(outer_channels=inner_channels)
            res4=Res_Block(outer_channels=outer_channels)
            attention_non=CA_M3()
            attention_position=CA_M2(in_channel=inner_channels)
            attention_channel=CA_M1()
            
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                if use_position_attention:
                    p_attention=attention_position
                else:
                    p_attention=attention_non
                if use_channel_attention:
                    c_attention=attention_channel
                else:
                    c_attention=attention_non
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [res1,en_relu, en_conv,p_attention,res2]
                decoder = [res3,de_relu, de_conv, de_norm,res4]
                model = encoder + decoder
            elif outermost:
                if use_position_attention:
                    p_attention=attention_position
                else:
                    p_attention=attention_non
                if use_channel_attention:
                    c_attention=attention_channel
                else:
                    c_attention=attention_non
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels)
                encoder = [en_conv,p_attention]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            else:
                if use_position_attention:
                    p_attention=attention_position
                else:
                    p_attention=attention_non
                if use_channel_attention:
                    c_attention=attention_channel
                else:
                    c_attention=attention_non
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [res1,en_relu, en_conv, en_norm,p_attention,res2]
                decoder = [res3,de_relu, de_conv, de_norm,res4]
                model = encoder + [inner_block] + decoder
            self.c_attention=c_attention
            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return self.c_attention(self.model(x)+x)

class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        #Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True,use_channel_attention=True)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet,use_channel_attention=True)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet,use_channel_attention=True)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))

def network_init(net):
    for param in net.collect_params().values():
        param_init(param)

def set_network():
    net=nn.HybridSequential()
    # Pixel2pixel networks
    netG = UnetGenerator(in_channels=3, num_downs=8)
    with net.name_scope():
        net.add(netG)
        net.add(nn.Conv2D(1,kernel_size=1,prefix=''))
        net.add(nn.BatchNorm())
    # Initialize parameters
    net.initialize()
    # trainer for the generator and the discriminator

    return net
