#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:12:18 2019

@author: ljh
"""
from mxnet.gluon import nn
import mxnet as mx
from mxnet import nd, gluon, init
from mxboard import SummaryWriter
from time import time

class BasicBlockV2(nn.HybridBlock):
    def __init__(self,channels,stride,dilation=1,downsample=False,**kwargs):
        super(BasicBlockV2,self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv2D(channels,kernel_size=3,strides=stride,padding=dilation,\
                     use_bias=False,dilation=dilation)
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels,kernel_size=1,strides=1,use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels,kernel_size=1,strides=stride,\
                                        use_bias=False)
        else:
            self.downsample = None
    def hybrid_forward(self,F,x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type="relu")
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = F.Activation(x, act_type="relu")
        x = self.conv2(x)    
        
        return x + residual

class ASPP(nn.HybridBlock):
    def __init__(self,output_stride,depth=256,**kwargs):
        super(ASPP, self).__init__(**kwargs)
        with self.name_scope():
            self.GLBpooling = nn.GlobalAvgPool2D()
            self.image_level_features = nn.Conv2D(depth,kernel_size=1)
            self.at_pool1x1 = nn.Conv2D(depth,kernel_size=1)
            if output_stride==16:
                self.at_pool3x3_1=nn.Conv2D(depth,kernel_size=3,dilation=6,padding=6)
                self.at_pool3x3_2=nn.Conv2D(depth,kernel_size=3,dilation=12,padding=12)
                self.at_pool3x3_3=nn.Conv2D(depth,kernel_size=3,dilation=18,padding=18)
            elif output_stride==8:
                self.at_pool3x3_1 = nn.Conv2D(depth, kernel_size=3, dilation=1, padding=1)
                self.at_pool3x3_2 = nn.Conv2D(depth, kernel_size=3, dilation=2, padding=2)
                self.at_pool3x3_3 = nn.Conv2D(depth, kernel_size=3, dilation=3, padding=3)# dilation=4
                
            self.image_pool_deconv = nn.Conv2DTranspose(depth,kernel_size=16,strides=8,padding=4)
            self.last = nn.Conv2D(depth,kernel_size=1)                 
            self.bn = nn.BatchNorm()
            
    def hybrid_forward(self,F,x):
        image_level_features=self.image_level_features(self.GLBpooling(x))
        image_level_features = self.image_pool_deconv(image_level_features)
        image_level_features=self.bn(image_level_features)
        at_pool1x1=self.at_pool1x1(x)
        at_pool1x1=self.bn(at_pool1x1)
        at_pool3x3_1=self.at_pool3x3_1(x)
        at_pool3x3_1=self.bn(at_pool3x3_1)
        at_pool3x3_2=self.at_pool3x3_2(x)
        at_pool3x3_2=self.bn(at_pool3x3_2)
        at_pool3x3_3=self.at_pool3x3_3(x)
        at_pool3x3_3=self.bn(at_pool3x3_3)
        out = F.concat(image_level_features,at_pool1x1,at_pool3x3_1,at_pool3x3_2,at_pool3x3_3,dim=1)
        out = self.bn(self.last(out))
        return out  # out.shape = Nx256x8x8
        
class My_Deeplab(nn.HybridBlock):
    def __init__(self,classes=1,**kwargs):
        super(My_Deeplab,self).__init__(**kwargs)
        self.block1 = nn.HybridSequential(prefix="")  #block1.output = Nx64x64x64
        with self.block1.name_scope():
            self.block1.add(
                    BasicBlockV2(channels=64,stride=1,downsample=True),
                    BasicBlockV2(channels=64,stride=1,downsample=False)
            )
        self.block2 = nn.HybridSequential(prefix="")  #block2.output = Nx64x32x32
        with self.block2.name_scope():
            self.block2.add(
                    BasicBlockV2(channels=64,stride=1,downsample=False),
                    BasicBlockV2(channels=64,stride=2,downsample=True)                   
            )
        self.block3 = nn.HybridSequential(prefix="")  #block3.output = Nx128x16x16
        with self.block3.name_scope():
            self.block3.add(
                    BasicBlockV2(channels=128,stride=1,downsample=True),
                    BasicBlockV2(channels=128,stride=2,downsample=True)
            )
        self.block4 = nn.HybridSequential(prefix="")  #block4.output = Nx256x8x8
        with self.block4.name_scope():
            self.block4.add(
                    BasicBlockV2(channels=256,stride=1,downsample=True),
                    BasicBlockV2(channels=256,stride=1,downsample=False),
                    BasicBlockV2(channels=256,stride=2,downsample=True)
                    )
        # middle model
        self.aspp_model = ASPP(output_stride=8,depth=256) #out.shape=Nx256x8x8                    
        self.upsample_1 = nn.Conv2DTranspose(channels=128,kernel_size=4,strides=2,padding=1,use_bias=False)
        self.upsample_2 = nn.Conv2DTranspose(channels=64,kernel_size=4,strides=2,padding=1,use_bias=False)
        self.upsample_3 = nn.Conv2DTranspose(channels=64,kernel_size=4,strides=2,padding=1,use_bias=False)
        
        self.branch_16 = nn.HybridSequential(prefix="")
        with self.branch_16.name_scope():
            self.branch_16.add(
                    BasicBlockV2(channels=128,stride=1,dilation=3,downsample=False),
                    BasicBlockV2(channels=128,stride=1,dilation=5,downsample=False),
                    BasicBlockV2(channels=128,stride=1,dilation=7,downsample=False)
            )
        
        self.branch_32 = nn.HybridSequential(prefix="")
        with self.branch_32.name_scope():
            self.branch_32.add(
                    BasicBlockV2(channels=64,stride=1,dilation=7,downsample=False),
                    BasicBlockV2(channels=64,stride=1,dilation=9,downsample=False),
                    BasicBlockV2(channels=64,stride=1,dilation=11,downsample=False)
            )
        
        self.branch_64 = nn.HybridSequential(prefix="")
        with self.branch_64.name_scope():
            self.branch_64.add(
                    BasicBlockV2(channels=64,stride=1,dilation=9,downsample=False),
                    BasicBlockV2(channels=64,stride=1,dilation=13,downsample=False),
                    BasicBlockV2(channels=64,stride=1,dilation=17,downsample=False)
                    )
        #decoder model
        self.deconv1 = nn.HybridSequential(prefix="") #deconv1.shape=Nx128x16x16
        with self.deconv1.name_scope():
            self.deconv1.add(
                    BasicBlockV2(channels=256,stride=1,downsample=False),
                    BasicBlockV2(channels=256,stride=1,downsample=False),
                    BasicBlockV2(channels=256,stride=1,downsample=False)
                    )
        self.deconv2 = nn.HybridSequential(prefix="") #deconv2.shape=Nx64x32x32
        with self.deconv2.name_scope():
            self.deconv2.add(
                    BasicBlockV2(channels=128,stride=1,downsample=False),
                    BasicBlockV2(channels=128,stride=1,downsample=False)
                    ) 

        self.last_conv = nn.HybridSequential(prefix="")
        with self.last_conv.name_scope():
            self.last_conv.add(
                    BasicBlockV2(channels=128,stride=1,downsample=False),
                    BasicBlockV2(channels=classes,stride=1,downsample=True)
                    )
        
    def hybrid_forward(self,F,x):
        block1_out = self.block1(x)
        block2_out = self.block2(block1_out)
        block3_out = self.block3(block2_out)
        block4_out = self.block4(block3_out)
        
        aspp = self.aspp_model(block4_out) #aspp.shape=1x256x8x8
        
        deconv1_out = self.upsample_1(aspp)
        branch_1 = self.branch_16(block3_out)
        deconv1_out = F.concat(deconv1_out,branch_1,dim=1) #1x256x16x16
        deconv1_out = self.deconv1(deconv1_out)
        
        deconv2_out = self.upsample_2(deconv1_out)
        branch_2 = self.branch_32(block2_out)
        deconv2_out = F.concat(deconv2_out,branch_2,dim=1)#1x128x32x32
        deconv2_out = self.deconv2(deconv2_out)
        
        deconv3_out = self.upsample_3(deconv2_out)
        branch_3 = self.branch_64(block1_out)
        out = F.concat(deconv3_out,branch_3,dim=1) #1x128x64x64
        out = self.last_conv(out)
        return out

net = My_Deeplab(classes=1)

