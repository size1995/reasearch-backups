import multi_att_net
import tools_for_ex_seg as tool
import mxnet as mx
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from mxnet import nd
from mxnet import image
from skimage.measure import label as la

ctx1=mx.gpu(4)
ctx2=mx.gpu(5)
ctx3=mx.gpu(6)
ctx4=mx.gpu(7)



net_vessel=multi_att_net.set_network()
net_vessel.collect_params().reset_ctx(ctx1)
net_vessel.load_parameters("/home/sz/hard_ex_segmentation/vessel.params",ctx=ctx1)

net_hard=multi_att_net.set_network()
net_hard.collect_params().reset_ctx(ctx2)
net_hard.load_parameters("/home/sz/hard_ex_segmentation/IDRID_HE.params",ctx=ctx2)

net_mic=multi_att_net.set_network()
net_mic.collect_params().reset_ctx(ctx3)
net_mic.load_parameters("/home/sz/hard_ex_segmentation/micro.params",ctx=ctx3)

net_heam=multi_att_net.set_network()
net_heam.collect_params().reset_ctx(ctx4)
net_heam.load_parameters("/home/sz/hard_ex_segmentation/heam.params",ctx=ctx4)


def prob_map_get(img,net,ctx,crop_size=256,stride=128,threshold=0.5):
    pred,prob_map=tool.hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
    return prob_map

def one_zero2heatmap(prob):
    prob=(prob*255).astype('uint8')
    prob_color = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
    return prob_color

def hard_sum(img_path1,img_path2):
    whole_data=[]
    img_name_list1,label_name_list1=tool.generate_list(img_path1,img_path1)
    img_name_list2,label_name_list2=tool.generate_list(img_path2,img_path2)
    
    number=len(img_name_list1)
    name_list=[]
    sum_hard=[]
    sum_mic=[]
    sum_heam=[]
    
    for i in range(number):
        img=image.imread(img_name_list1[i])
        img1=image.imread(img_name_list2[i])

        x,y,z=img.shape
        imgr=tool.imresize(img1)  
        stamp,prob_map=tool.optic_disc_seg(imgr,net_vessel,crop_size=256,stride=128,ctx=ctx1,threshold=0.5)
        stamp=cv2.resize(stamp, (y, x), interpolation=cv2.INTER_NEAREST)
        prob_map=cv2.resize(prob_map.asnumpy(), (y, x), interpolation=cv2.INTER_NEAREST)
        
        hardpr=prob_map_get(img,net_hard,ctx=ctx2,crop_size=256,stride=128,threshold=0.5)
        heampr=prob_map_get(img,net_heam,ctx=ctx4,crop_size=256,stride=128,threshold=0.5)
        micpr=prob_map_get(img,net_mic,ctx=ctx3,crop_size=256,stride=128,threshold=0.5)

        hard_mask=(hardpr.asnumpy()*(1-stamp)*(1-prob_map))>0.5
        mic_mask=((1-stamp)*(1-heampr.asnumpy())*micpr.asnumpy())>0.5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        mic_mask = cv2.morphologyEx((mic_mask*np.array([1])).astype('uint8'), cv2.MORPH_OPEN, kernel)
        heam_mask=((1-stamp)*(1-prob_map)*heampr.asnumpy())>0.5
        
        hardpr1=hardpr.asnumpy()*hard_mask
        micpr1=micpr.asnumpy()*mic_mask
        heampr1=heampr.asnumpy()*heam_mask
        
        img_name=img_name_list1[i].split('/')[-1]
        print(img_name)
        name_list.append(img_name)
        sum_hard.append(int(np.sum(hardpr1)))    
        sum_mic.append(int(np.sum(micpr1)))
        sum_heam.append(int(np.sum(heampr1)))
        whole_data.append([img_name,int(np.sum(hardpr1)),int(np.sum(micpr1)),int(np.sum(heampr1))])
        
    return whole_data
        
        
        
        