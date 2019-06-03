import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.ndarray import one_hot


class Net(gluon.Block):
    def __init__(self):
        super(Net, self).__init__()
        ####### encoding layers #######
        self.d0 = nn.Conv2D(in_channels=3, channels=64, kernel_size=3, strides=2, padding=1)
        self.d1 = nn.Conv2D(in_channels=64, channels=128, kernel_size=3, strides=2, padding=1)

        self.d2 = nn.Conv2D(in_channels=128, channels=256, kernel_size=3, strides=2, padding=1)

        self.d3 = nn.Conv2D(in_channels=256, channels=512, kernel_size=3, strides=2, padding=1)
        
        self.d4 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
       
        self.d5 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
        
        self.d6 = nn.Conv2D(in_channels=512, channels=512, kernel_size=3, strides=2, padding=1)
        
        self.d7 = nn.Conv2D(in_channels=512, channels=512, kernel_size=4, strides=2, padding=1)
        

        ####### decoding layers #######
        self.u0 = nn.Conv2DTranspose(in_channels=512, channels=512, kernel_size=4, strides=2, padding=1)
        
        self.u1 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        
        self.u2 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        
        self.u3 = nn.Conv2DTranspose(in_channels=1024, channels=512, kernel_size=4, strides=2, padding=1)
        
        self.u4 = nn.Conv2DTranspose(in_channels=1024, channels=256, kernel_size=4, strides=2, padding=1)
        
        self.u5 = nn.Conv2DTranspose(in_channels=512, channels=128, kernel_size=4, strides=2, padding=1)
        
        self.u6 = nn.Conv2DTranspose(in_channels=256, channels=64, kernel_size=4, strides=2, padding=1)
        
        self.u7 = nn.Conv2DTranspose(in_channels=128, channels=1, kernel_size=4, strides=2, padding=1)

    def forward(self, x):
        ####### encoding layers #######
        x_d0 = F.relu(self.d0(x))
        x_d1 = F.relu(self.d1(x_d0))
        x_d2 = F.relu(self.d2(x_d1))
        x_d3 = F.relu(self.d3(x_d2))
        x_d4 = F.relu(self.d4(x_d3))
        x_d5 = F.relu(self.d5(x_d4))
        x_d6 = F.relu(self.d6(x_d5))
        x_d7 = F.relu(self.d7(x_d6))

        ####### decoding layers #######
        x = F.relu(self.u0(x_d7))
        xcat = F.concat(x, x_d6, dim=1)
        x = F.relu(self.u1(xcat))
        xcat = F.concat(x, x_d5, dim=1)
        x = F.relu(self.u2(xcat))
        xcat = F.concat(x, x_d4, dim=1)
        x = F.relu(self.u3(xcat))
        xcat = F.concat(x, x_d3, dim=1)
        x = F.relu(self.u4(xcat))
        xcat = F.concat(x, x_d2, dim=1)
        x = F.relu(self.u5(xcat))
        xcat = F.concat(x, x_d1, dim=1)
        x = F.relu(self.u6(xcat))
        xcat = F.concat(x, x_d0, dim=1)
        x = self.u7(xcat)
        return x
def set_network():
    net=Net()
    net.initialize()
    return net