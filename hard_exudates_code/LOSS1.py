from mxnet import nd
from mxnet import gluon
#A positive weight Bnegative Weight

# warning output and lable have to live on the same context!
class weight_imbalance_loss(gluon.loss.Loss):
    def __init__(self,A=1,B=1,batch_axis=0,**kwargs):
        super(weight_imbalance_loss,self).__init__(None,batch_axis,**kwargs)
        self._A=nd.array([A])
        self._B=nd.array([B])
        
    def hybrid_forward(self,F,output,label):
        a=F.sigmoid(output.reshape(output.shape[0]))#sigmoid transport the output to 0-1
        y=label
        lna=F.log(a+1e-10)
        lna1=F.log(1-a+1e-10)
        loss=-(self._A.as_in_context(label.context)*y*lna+self._B.as_in_context(label.context)*(1-y)*lna1)
        return loss
