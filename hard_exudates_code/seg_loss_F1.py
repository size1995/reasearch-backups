from mxnet import nd
from mxnet import gluon
class seg_loss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(seg_loss, self).__init__(weight, batch_axis, **kwargs)


    def hybrid_forward(self, F, pred, label, sample_weight=None):
        pred=F.sigmoid(pred)
        label = label.reshape(pred.shape)
        tp=F.sum(pred*label)
        p=F.sum(label)
        fn=p-tp
        tn=F.sum((1-label)*(1-pred))
        n=F.sum(1-label)
        
        fp=n-tn
        tpr=tp/(tp+fn+1e-12)
        ppv=tp/(tp+fp+1e-12)
        
        A=F.exp(tpr)+F.exp(ppv)
        
        A1=A/F.exp(tpr)
        A2=A/F.exp(ppv)
        
        B1=F.log(p)
        B2=F.log(n)
        C1=B2/B1
        C2=B1/B2

        
        loss = -C1*A1*(F.log(pred+1e-12))*F.exp(1-pred)*label - C2*A2*(F.log(1.-pred+1e-12)*F.exp(pred)*(1.-label))    
        return F.mean(loss, axis=self._batch_axis, exclude=True)