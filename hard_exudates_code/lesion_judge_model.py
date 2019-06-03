import mxnet as mx
from mxnet import nd
import numpy as np
from mxnet import gluon
import LOSS1
from time import time
from mxnet import autograd
import matplotlib.pyplot as plt
net=[mx.gpu(7)]

def data_settel(data,label):
    datas=nd.array([])
    labels=nd.array([])
    for i in range(data.shape[0]):
        if i%24==0:
            if i==0:
                datas=data[i].reshape((1,-1))
                labels=label[i]
            else:
                datas=nd.concat(datas,data[i].reshape((1,-1)),dim=0)
                labels=nd.concat(labels,label[i],dim=0)
    return datas,labels


train_data=nd.array(np.load("/home/sz/hard_ex_segmentation/train_feature.npy"))
train_data1=train_data[:,7:10]

train_label=nd.array(np.load("/home/sz/hard_ex_segmentation/train_label.npy"))
train_datas,train_labels=data_settel(train_data1,train_label)

train_iter = mx.io.NDArrayIter(train_datas, train_labels, 32, True, last_batch_handle='pad')

test_data=nd.array(np.load("/home/sz/hard_ex_segmentation/test_feature.npy"))
test_data1=test_data[:,7:10]
test_label=nd.array(np.load("/home/sz/hard_ex_segmentation/test_label.npy"))
test_datas,test_labels=data_settel(test_data1,test_label)

test_iter = mx.io.NDArrayIter(test_datas, test_labels, 1, False, last_batch_handle='dicard')
ctx=[mx.gpu(7)]
net=gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(64,activation='relu'))
    net.add(gluon.nn.Dense(1))
net.initialize(ctx=ctx)    
loss = LOSS1.weight_imbalance_loss(A=1,B=3)

def get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.label[0]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])
def evaluate_accuracy(data_iterator, net, ctx=[mx.cpu()] ,best_th=0.5):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc = nd.array([0])
    k = 0.
    data_iterator.reset()
    for batch in data_iterator:
        data, label, batch_size = get_batch(batch, ctx)
        for X, y in zip(data, label):
            acc += nd.sum((nd.sigmoid(net(X).reshape(-1))>best_th)==y).copyto(mx.cpu())
            k += y.size
        acc.wait_to_read()
    return acc.asscalar() / k

import numpy as np
def score_plt(pred,label):
    from numpy import arange
    ths=arange(0,1,0.001)
    one=np.ones(pred.shape[0])
    scores=[]
    for th in ths:
        pred1=pred>th     
        pred1=pred1*one
        
        tp=pred1*label
        fp=pred1-tp
        fn=label-tp
        FP=np.sum(fp)
        FN=np.sum(fn)
        score=(np.sum(one)-FP-FN)/np.sum(one)
        scores.append(score)
    return scores,ths

def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0    
        train_iter.reset()
        start = time()
        
        for i,batch in enumerate(train_iter):        
            data, label, batch_size = get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X).reshape(-1) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            for l in losses:
                l.backward()             
            train_acc += sum([((nd.sigmoid(yhat)>0.5)==y).sum().asscalar()
                              for yhat, y in zip(outputs, label)])
            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size
            m += sum([y.size for y in label])
            trainer.step(batch_size)
        
        print("Epoch %d. Loss: %.5f, Train acc %.5f, Time %.1f sec" % (
            epoch, train_loss/n, train_acc/m,time() - start
        ))
        test_iter.reset()
        for i,batch in enumerate(test_iter):
            data=batch.data[0]
            label=batch.label[0]
            data=data.as_in_context(ctx[0])
            label=label.as_in_context(ctx[0])
            yhat=net(data).reshape(-1)
            pred=nd.sigmoid(yhat)
            nd.waitall()
            if (i==0):
                cpl_pred=pred
                cpl_lable=label
            else:
                cpl_pred=nd.concat(cpl_pred,pred,dim=0)
                cpl_lable=nd.concat(cpl_lable,label,dim=0)
        
        pred=cpl_pred.asnumpy()
        label=cpl_lable.asnumpy()
        scores,ths=score_plt(pred,label)
        print('score=',max(scores))     
        print('threshold=',scores.index(max(scores)))
        plt.plot(ths,scores,color='green')
        plt.show()
trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.001, 'wd':1e-3})
def test(cpl_pred,cpl_lable):
    pred_list=[]
    label_list=[]
    pred=0
    label=0
    counter=0
    for i in range(cpl_pred.shape[0]): 
        pred=pred+cpl_pred[i].asscalar()
        label=label+cpl_lable[i].asscalar()
        counter=counter+1
        if counter==24:
            pred_list.append(pred/24)
            label_list.append(label/24)
            pred=0
            label=0
            counter=0
    return np.array(pred_list),np.array(label_list)