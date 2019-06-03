import unet_attention2
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from time import time
import numpy as np
import datatest
from mxboard import SummaryWriter
import seg_loss_F1
batch_size=200


img_root="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/train_data/"
label_root="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/train_label/"

dataset=datatest.get_dataset(img_root,label_root)
train_data=mx.gluon.data.DataLoader(dataset, batch_size,shuffle=True,
                                    last_batch='rollover',num_workers=8)

img_root1="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/val_data/"
label_root1="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/val_label/"
testset=datatest.get_dataset(img_root1,label_root1)
test_data=mx.gluon.data.DataLoader(testset, batch_size=100,shuffle=False,
                                   last_batch='keep',num_workers=8)
ctx=[mx.gpu(4),mx.gpu(5)]


net= unet_attention2.set_network()
net.collect_params().reset_ctx(ctx)
net.hybridize()

def get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.data[1]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])

loss=seg_loss_F1.seg_loss()

sw = SummaryWriter(logdir='./logs3', flush_secs=2)
    
global_step = 0
    
epoch_step=0   


def train(train_data, net, loss, ctx,global_step,epoch_step, num_epochs,best_F1=0.6):
    print("Start training on ", ctx)
      
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        if epoch<80:
            trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.001, 'wd':1e-3})
        elif epoch<110:
            trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.0001, 'wd':1e-3})
        elif epoch<140:
            trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': 0.00001, 'wd':1e-3})
        else:
            trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate': 0.000001,'momentum': 0.9,'wd':1e-3})
        train_loss, n,  = 0.0, 0.0
        TP,TN,FP,FN=0,0,0,0
        start = time()
        for i,batch in enumerate(train_data):
            data, label, batch_size = get_batch(batch, ctx)
            losses = []
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            
            for l in losses:
                l.backward()
            sw.add_scalar(tag='cross_entropy', value=l.mean().asscalar(), global_step=global_step)
            global_step += 1

            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size
            
            trainer.step(batch_size)
        for data,label in test_data:
            data=data.as_in_context(ctx[0])
            label=label.as_in_context(ctx[0])
            pred=net(data)
            nd.waitall()
            pred=nd.sigmoid(pred)
            pred=(pred>0.5).reshape(-1,256,256)
            
            TPt=nd.sum(pred*label).asscalar()
            FPt=nd.sum(pred-(pred*label)).asscalar()
            FNt=nd.sum(label-(pred*label)).asscalar()
            TNt=nd.sum((1-pred)*(1-label)).asscalar()
            
            TP=TP+TPt
            FP=FP+FPt
            FN=FN+FNt
            TN=TN+TNt

        ACC=(TP+TN)/(TP+TN+FP+FN+1e-15)
        TPR=TP/ (TP+ FN+1e-15)     
        TNR= TN/(FP+TN+1e-15)
        PPV=TP/(TP+FP+1e-15)
        F1=2*PPV*TPR/(PPV+TPR+1e-15)
        
        sw.add_scalar(tag='test_acc', value=ACC, global_step=epoch_step)
        sw.add_scalar(tag='test_TPR', value=TPR, global_step=epoch_step)
        sw.add_scalar(tag='test_TNR', value=TNR, global_step=epoch_step)
        sw.add_scalar(tag='test_PPV', value=PPV, global_step=epoch_step)
        sw.add_scalar(tag='F1', value=F1, global_step=epoch_step)
        epoch_step+=1
        print('EPOCH',epoch)
        print('test_acc=',ACC)
        print('test_TPR=',TPR)
        print('test_TNR=',TNR)
        print('test_PPV=',PPV) 
        print('F1=',F1)  

        if F1>best_F1:
            net.save_parameters('u_attention.params')
            best_F1=F1
        if epoch == 0:
            sw.add_graph(net)
            
        print('train_loss=',train_loss/n)
        print('time:',time() - start)
    sw.close()
    net.export("mynet", epoch)

train(train_data, net, loss, ctx,global_step=0,epoch_step=0, num_epochs=150)
