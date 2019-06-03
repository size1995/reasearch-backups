import mxnet as mx
use_gpu = True
ctx = mx.cpu()
from attention_block import self_attention_block,CA_M,CA_M1,CA_M2,CA_M3
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout

class encoder(HybridBlock):
    def __init__(self,in_channel, out_channel):
        super(encoder,self).__init__()
        with self.name_scope():
            en_conv = Conv2D(channels=out_channel, kernel_size=4, strides=2,
                             padding=1,in_channels=in_channel)
            norm = BatchNorm(momentum=0.1, in_channels=out_channel)
            relu = LeakyReLU(alpha=0.2)
        encode=[en_conv,norm,relu]
        self.encoder = HybridSequential()
        with self.encoder.name_scope():
            for block in encode:
                self.encoder.add(block)
    def hybrid_forward(self,F,x):
        return self.encoder(x)
    
class decoder(HybridBlock):
    def __init__(self,in_channel, out_channel):
        super(decoder,self).__init__()
        with self.name_scope():
            de_conv=Conv2DTranspose(channels=out_channel, kernel_size=4, strides=2, padding=1,
                                          in_channels=in_channel)
            norm = BatchNorm(momentum=0.1, in_channels=out_channel)
            relu = LeakyReLU(alpha=0.2)
        decode=[de_conv,norm,relu]
        self.decoder = HybridSequential()
        with self.decoder.name_scope():
            for block in decode:
                self.decoder.add(block)
    def hybrid_forward(self,F,x):
        return self.decoder(x)

class net1(HybridBlock):
    def __init__(self):
        super(net1,self).__init__()
        self.net=HybridSequential()
        with self.net.name_scope():
            self.net.add(encoder(3,16))
            self.net.add(encoder(16,32))
            self.net.add(encoder(32,64))
            self.net.add(decoder(64,32))
            self.net.add(decoder(32,16))
            self.net.add(decoder(16,1))
    def hybrid_forward(self,F,x):
        return self.net(x) 

class net2(HybridBlock):
    def __init__(self):
        super(net2,self).__init__()
        self.net=HybridSequential()       
        with self.net.name_scope():
            
            self.net.add(encoder(3,16))
            self.net.add(encoder(16,32))
        self.att= CA_M2(32)
        self.net1=HybridSequential()
        with self.net1.name_scope():      
            self.net1.add(encoder(32,64))
            self.net1.add(decoder(64,32))
            self.net1.add(decoder(32,16))
            self.net1.add(decoder(16,1))

    def hybrid_forward(self,F,x):
        y1=self.net(x)
        y2=self.att(y1)
        y3=self.net1(y2)
        return y3


class net3(HybridBlock):
    def __init__(self):
        super(net3,self).__init__()     
        with self.name_scope():
            encoder1=encoder(3,16)
            encoder2=encoder(16,32)
            encoder3=encoder(32,64)
            decoder1=decoder(64,32)
            decoder2=decoder(32,16)
            decoder3=decoder(16,1)
            att2=CA_M2(32)
            att3=CA_M2(64)
            att4=CA_M2(32)
        blocks=[encoder1,encoder2,att2,encoder3,att3,decoder1,att4,decoder2,decoder3]
        self.net1=HybridSequential()
        with self.net1.name_scope():  
            for block in blocks:
                self.net1.add(block)

    def hybrid_forward(self,F,x):
        return self.net1(x)

def set_network1():
    net = net1()
    net.initialize(init='Xavier')
    return net
def set_network2():
    net = net2()
    net.initialize(init='Xavier')
    return net
def set_network3():
    net = net3()
    net.initialize(init='Xavier')
    return net      
            