import u_double_attention
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from time import time
import numpy as np
import datatest
from mxboard import SummaryWriter
import matplotlib.pyplot as plt
import tools_for_ex_seg as tool
import dice_entropy_loss

batch_size=100


img_root="/home/sz/hard_ex_segmentation/IDRID/half_train_crop/img/"
label_root="/home/sz/hard_ex_segmentation/IDRID/half_train_crop/label/"

dataset=datatest.get_dataset(img_root,label_root)
train_data=mx.gluon.data.DataLoader(dataset, batch_size,shuffle=True,
                                    last_batch='rollover',num_workers=8)

img_path="/home/sz/hard_ex_segmentation/IDRID/test/img/"
label_path="/home/sz/hard_ex_segmentation/IDRID/test/label/"
ctx=[mx.gpu(6),mx.gpu(7)]


net=u_double_attention.set_network()
net.collect_params().reset_ctx(ctx)

"""
net.hybridize()
"""
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
loss=seg_loss_F1.seg_loss()

loss=dice_entropy.seg_loss()
"""
loss=dice_entropy_loss.seg_loss()

    
global_step = 0
    
epoch_step=0   

def plot_scale(scale):
    point=scale.reshape(shape=([-1]))
    point=point.asnumpy()
    x=np.arange(0,point.shape[0])
    plt.plot(x,point)

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
            global_step += 1

            train_loss += sum([l.sum().asscalar() for l in losses])
            n += batch_size
            
            trainer.step(batch_size)
 
        acc_list,tpr_list,tnr_list,ppv_list,ACC,TPR,TNR,PPV,F1=tool.evaluate(img_path,label_path,net,crop_size=128,stride=64,ctx=ctx[0],threshold=0.5)
        if F1>best_F1:
            
            net.save_parameters('pretrain_half.params')
            
            best_F1=F1

        print('EPOCH',epoch)
        print('train_loss=',train_loss/n)
        if epoch==0: 
            last_loss=train_loss/n
            now_loss=train_loss/n
        else:
            last_loss=now_loss
            now_loss=train_loss/n
            
        
        print('time:',time() - start)


train(train_data, net, loss, ctx ,global_step,epoch_step, num_epochs=80,best_F1=0.65)
