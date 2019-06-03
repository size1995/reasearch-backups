from mxnet import nd
from mxnet.gluon import nn

class multi_scale_block(nn.HybridBlock):
    def __init__(self, in_channels, inter_channels=None):
        super( multi_scale_block, self).__init__()

        self.in_channels=in_channels
        self.inter_channels=inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 3
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2D

        bn = nn.BatchNorm()
        d_relu=nn.Activation(activation='relu')
        c1=conv_nd(in_channels=self.in_channels,channels=self.in_channels,kernel_size=3, strides=1,dilation=2,padding=2)
        c2=conv_nd(in_channels=self.in_channels,channels=self.in_channels,kernel_size=3, strides=1,dilation=3,padding=3)
        c3=conv_nd(in_channels=self.in_channels,channels=self.in_channels,kernel_size=3, strides=1,dilation=5,padding=5)
        self.d1 = nn.HybridSequential()
        with self.d1.name_scope():
            self.d1.add(d_relu,c1,bn)

        self.d2=nn.HybridSequential()
        with self.d2.name_scope():
            self.d2.add(d_relu,c2,bn)

        self.d3=nn.HybridSequential()
        with self.d3.name_scope():
            self.d3.add(d_relu,c3,bn)

        W = conv_nd(channels=self.in_channels, kernel_size=1, strides=1, padding=0)
        self.W = nn.HybridSequential()
        with self.W.name_scope():
            self.W.add(W,bn)

    def hybrid_forward(self, F, x):
        x1=self.d1(x)
        x2=self.d2(x)
        x3=self.d3(x)
        f=F.concat(x1,x2,x3,dim=1)
        f=self.W(f)
        return f