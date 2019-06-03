import lesion_net
from mxnet import init
import mxnet as mx
from mxnet import nd
from time import time
from mxnet import gluon
from tqdm import tqdm
from mxnet import autograd
import matplotlib.pyplot as plt
import LOSS1

ctx=[mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7)]
net=lesion_net.net
net[-1].initialize(init=init.Xavier())
net[-2].initialize(init=init.Xavier())
net.collect_params().reset_ctx(ctx)

net.hybridize()
best_th=0.5

train_data = mx.io.ImageRecordIter(
            path_imgrec="/home/sz/hard_ex_segmentation/lesion_train.rec",
            data_shape= (3,320,320),                   
            batch_size=128,
            preprocess_threads=5,
            prefetch_buffer=16,
            scale=1./255,
            shuffle=True
        )

test_data = mx.io.ImageRecordIter(
            path_imgrec="/home/sz/hard_ex_segmentation/lesion_test.rec",
            data_shape= (3,320,320),                      
            batch_size=103,
            preprocess_threads=5,
            prefetch_buffer=16,
            scale=1./255,
            shuffle=False,
            round_batch=False
        )
####LOSS
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
###########################SCORE############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
        TP=np.sum(tp)
        FP=np.sum(fp)
        FN=np.sum(fn)
        p=TP/(TP+FP+1e-10)
        r=TP/(TP+FN+1e-10)
        score=(np.sum(one)-FP-FN)/np.sum(one)
        scores.append(score)
    return scores,ths
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

def train(train_data, test_data, net, loss, trainer, ctx, num_epochs,best_score=0.78):
    print("Start training on ", ctx)
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    best_ths=0.5
    for epoch in range(num_epochs):
        train_loss, train_acc, n, m = 0.0, 0.0, 0.0, 0.0    
        train_data.reset()
        start = time()
        
        for i,batch in enumerate(train_data):        
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
        test_data.reset()
        for i,batch in enumerate(test_data):
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
trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.00001, 'wd':1e-3})
"""
train(train_data, test_data, net, loss, trainer, ctx, num_epochs=10)
"""
net.collect_params().save('loss_beta30.params')