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


class CA_M4(nn.HybridBlock):
    def __init__(self):
        super(CA_M4,self).__init__()
        self.gp = nn.GlobalAvgPool2D()
        self.bn=nn.BatchNorm()
    def hybrid_forward(self, F, x):
        x1=x.reshape((0,0,-1))
        
        num=x1.shape[2]
        
        x1_m=self.gp(x).reshape((0,0,1))
        
        cov1=F.batch_dot(x1-x1_m,(x1-x1_m).transpose((0,2,1)))/num
        rela=F.softmax(cov1,axis=-1)
        att_x=F.batch_dot(rela,x1)
        y=F.reshape_like(att_x,x)
        out=self.bn(x+y)
        return out,rela
#position attention
class CA_M5(nn.HybridBlock):
    def __init__(self,in_channel):
        super(CA_M5,self).__init__()
        self.in_channel=in_channel
        self.middel_channel=self.in_channel//2
        self.conv1=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn1=nn.BatchNorm()
        
        self.conv2=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn2=nn.BatchNorm()

        
        self.conv3=nn.Conv2D(in_channels=self.in_channel,channels=self.middel_channel,kernel_size=1,strides=1)
        self.bn3=nn.BatchNorm()
        
        self.conv_out=nn.Conv2D(in_channels=self.middel_channel,channels=self.in_channel,kernel_size=1,strides=1)
        
    def hybrid_forward(self, F, x):
        XA=self.conv1(x)
        XA=self.bn1(XA)
        X1=XA
        
        XB=self.conv2(x)
        XB=self.bn2(XB)

        
        XC=self.conv3(x)
        XC=self.bn3(XC)

        
        XA=XA.reshape((0,0,-1))
        XA=XA.transpose((0,2,1))#B*HW*C
        
        XB=XB.reshape((0,0,-1))#B*C*HW
        XC=XC.reshape((0,0,-1))#B*C*HW
        XC=XC.transpose((0,2,1))#B*HW*C
        att=F.batch_dot(XA,XB)
        
        att=F.softmax(att,axis=-1)
        
        X_ATT=F.batch_dot(att,XC)
        X_ATT=X_ATT.transpose((0,2,1))
        
        X_ATT=F.reshape_like(X_ATT,X1)
        X_ATT=self.conv_out(X_ATT)
        return X_ATT+x,att
            
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
                 use_dropout=False, use_bias=False, use_attention=True, use_resblock=True,use_p_at=False,use_c_at=False):
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
                if use_p_at:
                    
                    self.p_at=CA_M2(in_channel=inner_channels)
                else:
                    self.p_at=CA_M3()
                if use_c_at:
                    self.c_at=CA_M1()
                else:
                    self.c_at=CA_M3()
                res_block_3 = Res_Block(outer_channels=inner_channels)
                res_block_4 = Res_Block(outer_channels=outer_channels)
                if use_resblock:
                    res1 = res_block_1
                    encoder = [en_conv,en_norm,en_relu]
                    res2 = res_block_2
                    res3 = res_block_3
                    decoder = [de_conv,de_norm,de_relu]
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
                    encoder = [en_conv,en_norm,en_relu]
                    res2 = None
                    res3 = None
                    decoder = [ de_conv,de_norm,de_relu,channel_trans]
                    res4 = None
                else:
                    
                    encoder = [en_conv]
                    decoder = [de_relu, de_conv, de_norm,channel_trans]


            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                
                if use_p_at:
                    self.p_at=CA_M2(in_channel=inner_channels)
                else:
                    self.p_at=CA_M3()
                if use_c_at:
                    self.c_at=CA_M1()
                else:
                    self.c_at=CA_M3()
                    
                res_block_3 = Res_Block(outer_channels=inner_channels)
                res_block_4 = Res_Block(outer_channels=outer_channels)
      

                if use_resblock:
                    res1 = res_block_1
                    encoder = [en_conv, en_norm,en_relu]
                    res2 = res_block_2
                    res3 = res_block_3
                    decoder = [de_conv, de_norm,de_relu]
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
            x2 = self.encoder(x1)
            x3=x2
            x4= self.inner_block(x3)
            x5 = x4
            x6 = self.decoder(x5)
            x7=x6
            return x7
        if self.innermost:
            x1=self.res1(x)
            x2 = self.encoder(x1)
            x2 = self.p_at(x2)
            x3= self.res2(x2)
            x4 = x3 
            x5 = self.res3(x4)
            x6=self.decoder(x5)
            x7= self.res4(x6)
            out=x7+x
            out=self.c_at(out)
        else:
            x1=self.res1(x)
            x2 = self.encoder(x1)
            x2 = self.p_at(x2)
            x3= self.res2(x2)
            x4= self.inner_block(x3)
            x5= self.res3(x4)
            x6 = self.decoder(x5)
            x7= self.res4(x6)
            out=x7+x
            out=self.c_at(out)
        return out


class UnetGenerator(HybridBlock):
    def __init__(self, ngf=32, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet1 = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True, use_c_at=True, use_resblock=True)
        unet2 = UnetSkipUnit(ngf * 8, ngf * 8, unet1, use_dropout=use_dropout, use_attention=True, use_resblock=True)
        unet3 = UnetSkipUnit(ngf * 8, ngf * 4, unet2, use_c_at=True, use_resblock=True)
        unet4 = UnetSkipUnit(ngf * 4, ngf * 2, unet3, use_c_at=True, use_resblock=True)

        self.model = unet4

    def hybrid_forward(self, F, x):
        x1= self.model(x)
        return x1


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
    net = UnetGenerator()
    return net

class middlelayer(HybridBlock):
    def __init__(self,innerblock=None,outer_channels=32,inner_channels=64,use_bias=False):
        super(middlelayer,self).__init__()
        with self.name_scope():
            res_block_1 = Res_Block(outer_channels=outer_channels)
            res_block_2 = Res_Block(outer_channels=inner_channels)
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)

            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)
            de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
            self.p_at=CA_M5(in_channel=inner_channels)
            self.c_at=CA_M4()
            
            res_block_3 = Res_Block(outer_channels=inner_channels)
            res_block_4 = Res_Block(outer_channels=outer_channels)
            res1 = res_block_1
            encoder = [en_conv, en_norm,en_relu]
            res2 = res_block_2
            res3 = res_block_3
            decoder = [de_conv, de_norm,de_relu]
            res4 = res_block_4
            
            self.encoder = HybridSequential()
            with self.encoder.name_scope():
                for block in encoder:
                    self.encoder.add(block)

            self.inner_block = innerblock

            self.res1=res1
            self.res2=res2
            self.res3=res3
            self.res4=res4

            self.decoder = HybridSequential()
            
            with self.decoder.name_scope():
                for block in decoder:
                    self.decoder.add(block)
    def hybrid_forward(self, F, x):
        x1=self.res1(x)
        x2 = self.encoder(x1)
        x2,pat = self.p_at(x2)
        x3= self.res2(x2)
        x4= self.inner_block(x3)
        x5= self.res3(x4)
        x6 = self.decoder(x5)
        x7= self.res4(x6)
        out=x7+x
        out,cat=self.c_at(out)
        return out,pat,cat        
class outerpart(HybridBlock):
    def __init__(self,innerblock=None):
        super(outerpart,self).__init__()
        with self.name_scope():
            en_conv1 = Conv2D(channels=32, kernel_size=4, strides=2, padding=1,
                             in_channels=3)
            en_relu1 = LeakyReLU(alpha=0.2)
            en_norm1 = BatchNorm(momentum=0.1, in_channels=32,prefix='en_norm1')

            de_relu1 = Activation(activation='relu')
            de_norm1 = BatchNorm(momentum=0.1, in_channels=3,prefix='de_norm1')
            de_conv1 = Conv2DTranspose(channels=3, kernel_size=4, strides=2, padding=1,
                                          in_channels=32)
            channel_trans=Conv2D(channels=1,in_channels=3,kernel_size=1, prefix='')
            encoder1 = [en_conv1,en_norm1,en_relu1]
            decoder1 = [de_conv1,de_norm1,de_relu1,channel_trans]
 
            self.encoder1 = HybridSequential()
            with self.encoder1.name_scope():
                for block in encoder1:
                    self.encoder1.add(block)
            self.innerblock=innerblock
            self.decoder1 = HybridSequential() 
            with self.decoder1.name_scope():
                for block in decoder1:
                    self.decoder1.add(block)
                    
    def hybrid_forward(self, F, x):
        x1=self.encoder1(x)
        x2,p_att,c_att=self.innerblock(x1)
        x3=self.decoder1(x2)        
        
        return x3,p_att,c_att

def get_net():
    inner_block=set_network()
    midnet=middlelayer(innerblock=inner_block)
    net=outerpart(innerblock=midnet)
    net.initialize(init='Xavier')
    return net