import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D
class None_local_block(nn.HybridBlock):
    def __init__(self , in_channels, inter_channels=None,sub_sample=True,bn_layer=True):
        super(None_local_block, self).__init__()
        self.in_channels=in_channels
        conv_nd=Conv2D
        W=nn.HybridSequential()
        bn=nn.BatchNorm
        with W.name_scope():
            W.add(conv_nd(channels=self.in_channels,kernel_size=1, strides=1, padding=0),
            bn())
        self.W=W
        self.bn=bn
    def hybrid_forward(self, F, x):

        f = F.batch_dot(lhs=x.reshape(0,0,-1), rhs=x.reshape(0,0,-1), transpose_b=True, name='nonlocal_dot1')
        f = self.bn(f)
        f = F.batch_dot(lhs=f, rhs=x.reshape(0,0,-1), name='nonlocal_dot2' )
        f = self.bn(f)
        f=F.reshape_like(lhs=f, rhs=x)
        f=F.concat(f, x, dim=1)
        f=self.W(f)
        
        return f
