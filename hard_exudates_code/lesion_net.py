from mxnet.gluon import nn

from mxnet.gluon.model_zoo import vision as models
res_net152=models.resnet50_v2(prefix="a_")
res_net152.load_parameters("/home/sz/.mxnet/models/resnet50_v2-81a4e66a.params")
net=nn.HybridSequential(prefix="a_")
for layer in res_net152.features[0:12]:
    with net.name_scope(): 
        net.add(layer)
        
net.add(nn.Dense(7,activation="relu"))
net.add(nn.Dense(1))