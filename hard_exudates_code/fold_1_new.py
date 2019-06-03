import test_net
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from time import time
import numpy as np
import datatest

from mxboard import SummaryWriter
import tools_for_ex_seg as tool
import matplotlib.pyplot as plt
import dice_entropy_loss
batch_size=40

img_path="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/test/img/"
label_path="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/test/label/"

img_root="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/train_data/"
label_root="/home/sz/hard_ex_segmentation/home/sz/e_optha_dataset/train_label/"

dataset=datatest.get_dataset(img_root,label_root)
train_data=mx.gluon.data.DataLoader(dataset, batch_size,shuffle=True,
                                    last_batch='rollover',num_workers=8)

ctx=[mx.gpu(0),mx.gpu(1),mx.gpu(2),mx.gpu(3),mx.gpu(4),mx.gpu(5),mx.gpu(6),mx.gpu(7)]


net=test_net.set_network1()
net.collect_params().reset_ctx(ctx)


def get_batch(batch, ctx):
    if isinstance(batch, mx.io.DataBatch):
        data = batch.data[0]
        label = batch.data[1]
    else:
        data, label = batch
    return (gluon.utils.split_and_load(data, ctx),
            gluon.utils.split_and_load(label, ctx),
            data.shape[0])
"""
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
"""
loss=dice_entropy_loss.seg_loss()

sw = SummaryWriter(logdir='./logs_fold1', flush_secs=2)
    
global_step = 0
    
epoch_step=0   


def train(train_data, net, loss, ctx ,global_step,epoch_step, num_epochs,best_F1=0):
    print("Start training on ", ctx)
      
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
        
    for epoch in range(num_epochs):
        if epoch==0:
            learning_rate=0.001
            print('learning_rate=',learning_rate)
            last_loss=None
            now_loss=None
            trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': learning_rate, 'wd':1e-3})
        else:
            different = now_loss-last_loss
            if different>0:
                learning_rate=learning_rate*0.3
                trainer = gluon.Trainer(net.collect_params(),'sgd', {'learning_rate': learning_rate, 'wd':1e-3})
            print('learning_rate=',learning_rate)
        train_loss, n,  = 0.0, 0.0
        
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
           
        acc_list,tpr_list,tnr_list,ppv_list,ACC,TPR,TNR,PPV,F1=tool.evaluate(img_path,label_path,net,crop_size=256,stride=128,ctx=ctx[0],threshold=0.5)

        sw.add_scalar(tag='test_acc', value=ACC, global_step=epoch_step)
        sw.add_scalar(tag='test_TPR', value=TPR, global_step=epoch_step)
        sw.add_scalar(tag='test_TNR', value=TNR, global_step=epoch_step)
        sw.add_scalar(tag='test_PPV', value=PPV, global_step=epoch_step)
        sw.add_scalar(tag='F1', value=F1, global_step=epoch_step)
        epoch_step+=1
        print('EPOCH',epoch)


        if F1>best_F1:
            net.save_parameters('fold_1_st.params')
            best_F1=F1

        print('train_loss=',train_loss/n)
        if epoch==0: 
            last_loss=train_loss/n
            now_loss=train_loss/n
        else:
            last_loss=now_loss
            now_loss=train_loss/n
            
        
        print('time:',time() - start)
    sw.close()

train(train_data, net, loss, ctx ,global_step,epoch_step, num_epochs=150,best_F1=0)
