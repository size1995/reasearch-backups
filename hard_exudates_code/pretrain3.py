import res_u_net_attention
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from time import time
import numpy as np
import datatest
from mxboard import SummaryWriter
import tools_for_ex_seg as tool
import data1
import dice_entropy_loss
batch_size=150

img_path="/home/sz/hard_ex_segmentation/IDRID/test/img/"
label_path="/home/sz/hard_ex_segmentation/IDRID/test/label/"

img_root="/home/sz/hard_ex_segmentation/IDRID/train_crop/img/"
label_root="/home/sz/hard_ex_segmentation/IDRID/train_crop/label/"

dataset=datatest.get_dataset(img_root,label_root)
train_data=mx.gluon.data.DataLoader(dataset, batch_size,shuffle=True,
                                    last_batch='rollover',num_workers=8)

whole_img_path="/home/sz/hard_ex_segmentation/IDRID/train_append/img/"
whole_label_path="/home/sz/hard_ex_segmentation/IDRID/train_append/label/"
dataset1=data1.get_dataset(whole_img_path,whole_label_path)
train_data1=mx.gluon.data.DataLoader(dataset1, batch_size=10,shuffle=True,
                                    last_batch='rollover',num_workers=8)

img_root1="/home/sz/hard_ex_segmentation/IDRID/test_crop/img/"
label_root1="/home/sz/hard_ex_segmentation/IDRID/test_crop/label/"
testset=datatest.get_dataset(img_root1,label_root1)
test_data=mx.gluon.data.DataLoader(testset, batch_size=100,shuffle=False,
                                   last_batch='keep',num_workers=8)


ctx=[mx.gpu(6),mx.gpu(7)]


net=res_u_net_attention.set_network()
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
"""
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
"""
loss=dice_entropy_loss.seg_loss()

sw = SummaryWriter(logdir='./logs_pretrain3', flush_secs=2)
    
global_step = 0
global_step1 = 0
epoch_step=0   


def train(train_data,train_data1, net, loss, ctx ,global_step = 0,global_step1 = 0,epoch_step=0, num_epochs=150,best_F1=0):
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
                trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': learning_rate, 'wd':1e-3})
            print('learning_rate=',learning_rate)
        train_loss,train_loss1, n, n1  = 0.0,0.0, 0.0, 0.0
        
        start = time()
        print('patch train')
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
        print('time=',time()-start)
        start1=time()
        print('big_patch')
        for i,batch in enumerate(train_data1):
            data,label,batch_size=get_batch(batch, ctx)
            losses=[]
            with autograd.record():
                outputs = [net(X) for X in data]
                losses = [loss(yhat, y) for yhat, y in zip(outputs, label)]
            
            for l in losses:
                l.backward()
            sw.add_scalar(tag='cross_entropy1', value=l.mean().asscalar(), global_step=global_step1)
            global_step1 += 1
            
            train_loss1 += sum([l.sum().asscalar() for l in losses])
            n1 += batch_size
            
            trainer.step(batch_size)
        print('time=',time()-start1)
        TP,TN,FP,FN=0,0,0,0
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
        print("acc=",ACC)
        print('TPR=',TPR)
        print('TNR=',TNR)
        print('PPV=',PPV)
        print('F1=',F1)
       
        sw.add_scalar(tag='test_acc', value=ACC, global_step=epoch_step)
        sw.add_scalar(tag='test_TPR', value=TPR, global_step=epoch_step)
        sw.add_scalar(tag='test_TNR', value=TNR, global_step=epoch_step)
        sw.add_scalar(tag='test_PPV', value=PPV, global_step=epoch_step)
        sw.add_scalar(tag='F1', value=F1, global_step=epoch_step)
        epoch_step+=1
        print('EPOCH',epoch)


        if F1>best_F1:
            net.save_parameters('pretriain3.params')
            best_F1=F1
        if epoch == 0:
            sw.add_graph(net)
        
        print('train_loss1=',train_loss/n)
        
        print('train_loss2=',train_loss1/n1)
        if epoch==0: 
            last_loss=train_loss/n
            now_loss=train_loss/n
        else:
            last_loss=now_loss
            now_loss=train_loss/n
         
        
        print('time:',time() - start)
    sw.close()
    net.export("mynet", epoch)
"""
train(train_data,train_data1, net, loss, ctx ,global_step,epoch_step, num_epochs=150,best_F1=0.52)
"""