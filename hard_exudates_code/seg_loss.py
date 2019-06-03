from mxnet import nd
from mxnet import gluon
class seg_loss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(seg_loss, self).__init__(weight, batch_axis, **kwargs)


    def hybrid_forward(self, F, pred, label, sample_weight=None):
        pred=F.sigmoid(pred)
        label = label.reshape(pred.shape)

        loss = -100*(F.log(pred+1e-12))*F.exp(1-pred)*label - F.log(1.-pred+1e-12)*(1.-label)*F.exp(pred)
        
        return F.mean(loss, axis=self._batch_axis, exclude=True)