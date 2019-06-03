from mxnet.gluon import nn

from mxnet.gluon.model_zoo import vision as models
dense_net=models.densenet201(pretrained=False,prefix="a_")

net=nn.HybridSequential(prefix="a_")
for layer in dense_net.features[0:13]:
   net.add(layer) 

class output_block(nn.HybridBlock):
    def __init__(self,**kwargs):
        super (output_block,self).__init__(**kwargs)
        with self.name_scope():
            self.dense1=nn.Dense(256,activation='relu')
            self.globalavg=nn.GlobalAvgPool2D()   
            self.dense2=nn.Dense(1)
    def hybrid_forward(self,F,x):
        out=self.globalavg(x)
        out=self.dense1(out)
        out=self.dense2(out)
        return out
    
output=output_block(prefix="a_")
with net.name_scope():
    net.add(output)
