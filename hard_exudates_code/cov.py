from mxnet import nd
from mxnet.gluon import nn
gp = nn.GlobalAvgPool2D()

x=nd.array([1,2,3,4,3,7,5,2,1,9,6,5]).reshape((1,3,2,2))

x1=x.reshape(0,0,-1)
num=x1.shape[2]

x1_m=gp(x).reshape((0,0,0))
x1_m=x1_m.reshape((0,0,1))
cov1=nd.batch_dot(x1-x1_m,(x1-x1_m).transpose((0,2,1)))/num
##fangcha
x2=x1-x1_m
x2=x2*x2
x2=gp(x2)
dx=nd.batch_dot(x2,x2.transpose((0,2,1)))
