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

lr = 0.0002
beta1 = 0.5
lambda1 = 100

pool_size = 50

class Res_Block(HybridBlock):
    def __init__(self, outer_channels,use_bias=False):
        super(Res_Block, self).__init__()
        with self.name_scope():
            conv1 = Conv2D(channels=outer_channels, kernel_size=3, strides=1, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            relu1 = LeakyReLU(alpha=0.2)
            norm1 = BatchNorm(momentum=0.1, in_channels=outer_channels)
            conv2 = Conv2D(channels=outer_channels, kernel_size=3, strides=1, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            norm2 = BatchNorm(momentum=0.1, in_channels=outer_channels)
            relu2 = LeakyReLU(alpha=0.2)

            res_block=[conv1,norm1,relu1,conv2,norm2,relu2]
            self.res = HybridSequential()
            with self.res.name_scope():
                for block in res_block:
                    self.res.add(block)

    def hybrid_forward(self, F, x):
        residual=x
        x=self.res(x)
        x = x + residual
        return x
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False, use_attention=True, use_resblock=True):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            self.innermost=innermost
            self.use_attention = use_attention
            res_block=Res_Block(outer_channels=outer_channels)
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)

            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)
            
            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                res_block1=Res_Block(outer_channels=inner_channels)
                if use_resblock:
                    
                    encoder = [res_block, en_relu, en_conv]
                    decoder = [de_relu,res_block1, de_conv, de_norm]
                else:
                    encoder = [en_relu, en_conv]
                    decoder = [de_relu,de_conv, de_norm]
             
                attention = self_attention_block(in_channel=inner_channels)

                model = encoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                res_block1=Res_Block(outer_channels=inner_channels * 2)
                if use_resblock:
                    encoder = [res_block, en_conv]
                    decoder = [de_relu,res_block1, de_conv, Activation(activation='tanh')]
                else:
                    encoder = [en_conv]
                    decoder = [de_relu,de_conv, Activation(activation='tanh')]
                
                attention = self_attention_block(in_channel=inner_channels * 2)
                model = encoder + [inner_block]
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                res_block1=Res_Block(outer_channels=inner_channels * 2)
                if use_resblock:
                    encoder = [res_block, en_relu, en_conv, en_norm]
                    decoder = [de_relu,res_block1, de_conv, de_norm]
                else:
                    encoder = [en_relu, en_conv, en_norm]
                    decoder = [de_relu,de_conv, de_norm]
                
                attention = self_attention_block(in_channel=inner_channels * 2)

                model = encoder + [inner_block]
            if use_dropout:
                decoder += [Dropout(rate=0.5)]
			

			

            self.encoder = HybridSequential()
            with self.encoder.name_scope():
                for block in model:
                    self.encoder.add(block)
            self.inner_block=inner_block
            self.attention=attention
            self.decoder=HybridSequential()
            with self.decoder.name_scope():
                for block in decoder:
                    self.decoder.add()
    def hybrid_forward(self, F, x):
        x1=self.encoder(x)
        if self.innermost:
            x2=x1
            y=F.zeros(shape=(1,1,1,1))
        else:
            x2,y=self.inner_block(x1)
		
        x3,y1=self.attention(x2)
        x4=self.decoder(x3)

        if self.outermost:
            return x4,y1
        else:
            return F.concat(x4, x, dim=1),y1


class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=32, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet1 = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True, use_attention=True,use_resblock=True)

        unet2 = UnetSkipUnit(ngf * 8, ngf * 8, unet1, use_dropout=use_dropout, use_attention=True,use_resblock=True)

        unet3 = UnetSkipUnit(ngf * 8, ngf * 4, unet2, use_attention=True,use_resblock=True)
        unet4 = UnetSkipUnit(ngf * 4, ngf * 2, unet3, use_attention=True,use_resblock=True)
        unet5 = UnetSkipUnit(ngf * 2, ngf * 1, unet4, use_attention=True,use_resblock=True)
        unet6 = UnetSkipUnit(ngf, in_channels, unet5, outermost=True,use_resblock=True)

        with self.name_scope():
            self.unet1 = unet1
            self.unet2 = unet2
            self.unet3 = unet3
            self.unet4 = unet4
            self.unet5 = unet5
            self.unet6 = unet6

    def hybrid_forward(self, F, x):
        x1,y1=self.unet1(x)
        x2,y2=self.unet2(x1)
        x3,y3=self.unet3(x2)
        x4,y4=self.unet4(x3)
        x5,y5=self.unet5(x4)
        x6,y6=self.unet6(x5)
        return x6,y1,y2,y3,y4,y5,y6


def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx, force_reinit=True)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx, force_reinit=True)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx, force_reinit=True)
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape), force_reinit=True)


def network_init(net):
    for param in net.collect_params().values():
        param_init(param)


def set_network():
    net = nn.HybridSequential()
    netG = UnetGenerator(in_channels=3, num_downs=6)
    with net.name_scope():
        net.add(netG)
        net.add(nn.Conv2D(1, kernel_size=1, prefix=''))
        net.add(nn.BatchNorm())
    net.initialize()

    return net
