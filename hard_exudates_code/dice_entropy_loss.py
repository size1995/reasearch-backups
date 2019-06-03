from mxnet import nd
from mxnet import gluon
class seg_loss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(seg_loss, self).__init__(weight, batch_axis, **kwargs)


    def hybrid_forward(self, F, pred, label, sample_weight=None):
        pred=F.sigmoid(pred)
        label = label.reshape(pred.shape)
        
        dice1=2*(F.sum(label*pred,axis=[1,2,3]))/(F.sum(pred**2,axis=[1,2,3])+F.sum(label**2,axis=[1,2,3])+1e-12)
        dice2=2*(F.sum((1-label)*(1-pred),axis=[1,2,3]))/(F.sum((1-pred)**2,axis=[1,2,3])+F.sum((1-label)**2,axis=[1,2,3])+1e-12)
        label_sign=F.sum(label,axis=[1,2,3])/(F.sum(label,axis=[1,2,3])+1e-12)
        
        dice_cross_entropy_loss=-label_sign*(F.log(dice1+1e-12)+F.log(dice2+1e-12))-F.log(dice2+1e-12)*(1-label_sign)*2
        
        cross_entropy_loss = -(F.log(pred+1e-12))*F.exp(1-pred)*label - (F.log(1.-pred+1e-12)*F.exp(pred)*(1.-label))
        
        cross_entropy_loss_mean=F.mean(cross_entropy_loss, axis=self._batch_axis, exclude=True)
            
        loss=dice_cross_entropy_loss + cross_entropy_loss_mean
        
        return loss
    
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
        loss=-(y*lna+(1-y)*lna1)
        return loss
