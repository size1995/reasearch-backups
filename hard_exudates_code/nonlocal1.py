import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D
class None_local_block(nn.HybridBlock):
    def __init__(self , in_channels, inter_channels=None,sub_sample=True,bn_layer=True):
        super(None_local_block, self).__init__()
        self.sub_sample=sub_sample
        self.in_channels=in_channels
        self.inter_channels=inter_channels


        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1


        conv_nd=Conv2D
        max_pool_layer=nn.MaxPool2D(pool_size=(2,2))
        bn=nn.BatchNorm
        g = conv_nd(in_channels=self.in_channels, channels=self.inter_channels, kernel_size=1, strides=1,padding=0)
        if bn_layer:
            W=nn.HybridSequential()
            with W.name_scope():
                W.add(conv_nd(channels=self.in_channels,kernel_size=1, strides=1, padding=0),
                bn())
        else:
            W=conv_nd(channels=self.in_channels,kernel_size=1,strides=1,padding=0)

        theta=conv_nd(in_channels=self.in_channels,channels=self.inter_channels,kernel_size=1,strides=1,padding=0)
        phi=conv_nd(in_channels=self.in_channels,channels=self.inter_channels,kernel_size=1,strides=1,padding=0)

        if sub_sample:
            self.g=nn.HybridSequential()
            with self.g.name_scope():
                self.g.add(g, max_pool_layer)

            self.phi=nn.HybridSequential()
            with self.phi.name_scope():
                self.phi.add(phi, max_pool_layer)
        else:
            self.g=g
            self.phi=phi
        self.theta=theta
        self.W=W
    def hybrid_forward(self, F, x):
        indata3=self.g(x)
        g_x=F.reshape(indata3,shape=(0,0,-1))
        theta_x1=self.theta(x)
        theta_x=F.reshape(theta_x1,shape=(0,0,-1))
        phi_x=self.phi(x)
        phi_x=F.reshape(phi_x,shape=(0,0,-1))
        f = F.batch_dot(lhs=theta_x, rhs=phi_x, transpose_a=True, name='nonlocal_dot1')
        f=F.softmax(f,axis=2)
        y = F.batch_dot(lhs=f, rhs=g_x, transpose_b=True, name='nonlocal_dot2' )
        y=F.reshape_like(lhs=F.transpose(y, axes=(0, 2, 1)), rhs=theta_x1)     
        W_y=self.W(y)      
        z=W_y+x  
        return z
