import u_net_attention
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

img_root="/home/sz/hard_ex_segmentation/e_dataset/data2/train_crop/img/"
label_root="/home/sz/hard_ex_segmentation/e_dataset/data2/train_crop/label/"

dataset=datatest.get_dataset(img_root,label_root)
train_data=mx.gluon.data.DataLoader(dataset, batch_size,shuffle=True,
                                    last_batch='rollover',num_workers=8)


