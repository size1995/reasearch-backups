from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=4,strides=2,padding=1, activation='relu'),
        nn.BatchNorm(),
        nn.Conv2D(channels=16, kernel_size=4,strides=2,padding=1, activation='relu'),
        nn.BatchNorm(),
        nn.Conv2D(channels=32, kernel_size=4,strides=2,padding=1, activation='relu'),
        nn.BatchNorm(),
        nn.GlobalAvgPool2D(),
        nn.Dense(7, activation='relu'),
        nn.Dense(1))