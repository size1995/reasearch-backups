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

class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False,use_attention=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            self.use_attention=use_attention
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)
            attention= self_attention_block(in_channel=outer_channels)
            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                if use_attention:
                    attention= [self_attention_block(in_channel=inner_channels)]
                    decoder=attention+decoder
                
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                if use_attention:
                    attention= [self_attention_block(in_channel=inner_channels* 2)]
                    decoder=attention+decoder
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                if use_attention:
                    attention= [self_attention_block(in_channel=inner_channels* 2)]
                    decoder=attention+decoder
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)
        

class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        #Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True,use_attention=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet,use_dropout=use_dropout,use_attention=True)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet, use_attention=True)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet,use_attention=True)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet,use_attention=True)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)

def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx,force_reinit=True)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx,force_reinit=True)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx,force_reinit=True)
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape),force_reinit=True)

def network_init(net):
    for param in net.collect_params().values():
        param_init(param)

def set_network():
    net=nn.HybridSequential()
    netG = UnetGenerator(in_channels=3, num_downs=8)
    with net.name_scope():
        net.add(netG)
        net.add(nn.Conv2D(1,kernel_size=1,prefix=''))
        net.add(nn.BatchNorm())
    net.initialize()

    return net
