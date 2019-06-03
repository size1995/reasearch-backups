from mxnet.gluon import nn
class upsample(nn.HybridBlock):
    def __init__(self,in_channel):
        super(upsample,self).__init__()
        self.in_channel=in_channel
        with self.name_scope():
            self.deconv=nn.Conv2DTranspose(channels=self.in_channel//2,kernel_size=4,strides=2,padding=1,
                                      in_channels=self.in_channel,use_bias=True,activation='relu')

    def hybrid_forward(self, F, x):
        return self.deconv(x)
class downsample(nn.HybridBlock):
    def __init__(self,in_channel):
        super(downsample,self).__init__()
        self.in_channel=in_channel
        with self.name_scope():
            self.down=nn.Conv2D(channels=self.in_channel,kernel_size=4,strides=2,padding=1,
                                in_channels=self.in_channel,use_bias=True,activation='relu')
    def hybrid_forward(self, F, x):
        return self.down(x)
class middel_conv(nn.HybridBlock):
    def __init__(self,in_channel,out_channel):
        super(middel_conv,self).__init__()
        self.out_channels=out_channel
        self.in_channels=in_channel
        self.encode=nn.HybridSequential()
        with self.encode.name_scope():
            self.encode.add(nn.Conv2D(channels=self.out_channels, kernel_size=3,strides=1,padding=1,
                                      in_channels=self.in_channels,use_bias=True,activation='relu'))
            self.encode.add(nn.Conv2D(channels=self.out_channels, kernel_size=3,strides=1,padding=1,
                                      in_channels=self.out_channels,use_bias=True,activation='relu'))
    def hybrid_forward(self, F, x):
        return self.encode(x)

class basic_block(nn.HybridBlock):
    def __init__(self,in_channels,out_channels,inner_most=False,inner_block=None,outer_most=False):
        super(basic_block,self).__init__()
        self.inner_most=inner_most
        self.inner_block=inner_block
        self.outer_most=outer_most
        self.in_channels=in_channels
        self.out_channels=out_channels
        
        en=middel_conv(self.in_channels,self.out_channels)
        down=downsample(self.in_channels)
        up=upsample(self.out_channels)
        inner=self.inner_block
        de=middel_conv(self.out_channels*2,self.out_channels)
        
        if self.inner_most:        
            model=[down,en,up]
        else:
            model=[down,en,inner,de,up]
            
        self.net=nn.HybridSequential()
        with self.net.name_scope():
            for n in model:
                self.net.add(n)
                
    def hybrid_forward(self, F, x): 
        y=self.net(x)
        return F.concat(x, y)


class unet_mid(nn.HybridBlock):
    def __init__(self):
        super(unet_mid,self).__init__()
        u1=basic_block(512,1024,inner_most=True)
        u2=basic_block(256,512,inner_block=u1)
        u3=basic_block(128,256,inner_block=u2)
        u4=basic_block(64,128,inner_block=u3)
        self.net=u4

    def hybrid_forward(self, F, x):
        return self.net(x)

class unet(nn.HybridBlock):
    def __init__(self):
        super(unet,self).__init__()
        with self.name_scope():
            self.inner=unet_mid()
            self.encon=middel_conv(3,64)
            self.decon=middel_conv(128,64)
            self.trans=nn.Conv2D(channels=1,kernel_size=3,strides=1,padding=1,in_channels=64,use_bias=True,activation='relu')
    def hybrid_forward(self,F,x):
        y1=self.encon(x)
        y2=self.inner(y1)
        y3=self.decon(y2)
        y4=self.trans(y3)
        return y4
def set_network():
    net = unet()
    net.initialize()
    return net