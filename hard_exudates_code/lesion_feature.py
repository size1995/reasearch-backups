import lesion_net
from mxnet import init
import mxnet as mx
from mxnet import nd
from time import time
from mxnet import gluon
import matplotlib.pyplot as plt
from mxnet.gluon import nn
import pandas as pd
import os
import random


ctx=[mx.gpu(7)]
def csv_reader(csv_file):
    name_list=[]
    label_list=[]
    features=[]
    df = pd.read_csv(csv_file,header=0)

    for (name,series) in df.iterrows():
        feature=[]
        
        series=series
        name_list.append(series['name'])
        if series['grade']>0:
            label=1
        else:
            label=0
        label_list.append(label)
        feature.append(series['hard']/20000)
        feature.append(series['mic']/10000)
        feature.append(series['heam']/40000)
        
        features.append(feature)
    return name_list,label_list,features

def get_net():   
    net=lesion_net.net
    net[-1].initialize(init=init.Xavier())
    net[-2].initialize(init=init.Xavier())
    
    
    net.collect_params().reset_ctx(ctx)
    
    net.load_parameters('lesion.params',ctx=ctx)
    
    net1=nn.HybridSequential()
    for layer in net[0:13]:
        with net1.name_scope(): 
            net1.add(layer)
    return net, net1
        
train_data = mx.io.ImageRecordIter(
            path_imgrec="/home/sz/hard_ex_segmentation/lesion_train.rec",
            data_shape= (3,320,320),                   
            batch_size=24,
            preprocess_threads=5,
            prefetch_buffer=16,
            scale=1./255,
            shuffle=False,
            round_batch=False
        )

test_data = mx.io.ImageRecordIter(
            path_imgrec="/home/sz/hard_ex_segmentation/test_append.rec",
            data_shape= (3,320,320),                      
            batch_size=24,
            preprocess_threads=5,
            prefetch_buffer=16,
            scale=1./255,
            shuffle=False,
            round_batch=False
        )

def feature_extract(net,data):
    data.reset()
    for i,batch in enumerate(data):
        data=batch.data[0]
        label=batch.label[0]
        data=data.as_in_context(ctx[0])
        label=label.as_in_context(ctx[0])
        feature=net(data)
        nd.waitall()
        if (i==0):
            cpl_feature=feature
            cpl_lable=label
        else:
            cpl_feature=nd.concat(cpl_feature,feature,dim=0)
            cpl_lable=nd.concat(cpl_lable,label,dim=0)
    return cpl_feature,cpl_lable
    