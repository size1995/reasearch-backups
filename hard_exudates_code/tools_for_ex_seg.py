import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
from mxnet import nd
import mxnet as mx
from mxnet import image
from skimage.measure import label as la
import pandas as pd
import copy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

classes=['background','target']
colormap=[[0,0,0],[255,255,255]]
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i
    
def image2label(im):
    data=im.astype('int32').asnumpy()
    idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])
def label2image(pred):
    x=pred.astype('int32').asnumpy()
    cm=np.array(colormap).astype('uint8')
    return nd.array(cm[x,:])
#图片边缘像素黑色填充
def img_pad(img):
    BLACK=[0,0,0]
    h, w, c=img.shape
    append_length=max(h,w,)
    hh=False
    if append_length==h:
        hh=True
    if hh:
        constant = cv2.copyMakeBorder(img, 0, 0, int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT, value=BLACK)
    else:
        constant = cv2.copyMakeBorder(img, int((w-h)/2), int((w-h)/2),0,0 ,cv2.BORDER_CONSTANT, value=BLACK)
    return constant
#图片旋转
def img_rotate(img, angle, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated
#图片镜像翻转
def img_flip(img):
    h_flip = cv2.flip(img, 1)
    return h_flip
#获得img_path路径下所有图片路径列表，label_path路径下所有标签路径列表
def generate_list(img_path,label_path):
    for r_d_f in os.walk(img_path):
        file_dir=r_d_f
    img_root=file_dir[0]
    img_name_list = file_dir[2]
    img_name_list.sort()
    for i in range(len(img_name_list)):
        img_name_list[i]=img_root+img_name_list[i]
    for r_d_f in os.walk(label_path):
        file_dir1 = r_d_f
    img_root1 = file_dir1[0]
    label_name_list = file_dir1[2]
    label_name_list.sort()
    for i in range(len(label_name_list)):
        label_name_list[i]= img_root1+label_name_list[i]
    return img_name_list,label_name_list
#image_enhancing
def img_enhance(img):
    b, g, r = cv2.split(img)
    b=b.astype('float32')
    b_blur=cv2.GaussianBlur(b, (101, 101), 40)
    b1 = b - b_blur
    b1 = (b1 - b1.min()) / (b1.max() - b1.min())
    b1 = b1 * 255
    b1 = b1.astype('uint8')

    g=g.astype('float32')
    g_blur=cv2.GaussianBlur(g, (101, 101), 40)
    g1 = g - g_blur
    g1 = (g1 - g1.min()) / (g1.max() - g1.min())
    g1 = g1 * 255
    g1 = g1.astype('uint8')

    r=r.astype('float32')
    r_blur=cv2.GaussianBlur(r, (101, 101), 40)
    r1 = r - r_blur
    r1 = (r1 - r1.min()) / (r1.max() - r1.min())
    r1 = r1 * 255
    r1 = r1.astype('uint8')

    merged = cv2.merge([b1, g1, r1])
    return merged
def img_extend(img_path,label_path,img_extend_path,label_extend_path):
    img_name_list, label_name_list=generate_list(img_path,label_path)
    counter=0
    i1=len(img_name_list)
    i2=len(label_name_list)
    if i1!=i2:
        print('number of img and label not matched')
        return
    for i in range(i1):
        img_raw=cv2.imread(img_name_list[i])
        label_raw=cv2.imread(label_name_list[i])

        img_pad_ad=img_pad(img_raw)#base1
        label_pad_ad=img_pad(label_raw)#base1_label

        img_pad_ad_flip=img_flip(img_pad_ad)#base2
        label_pad_ad_flip=img_flip(label_pad_ad)#base2_label
        """
        angle_list=[0,60,120,180,240,300]
        """
        angle_list=[0,30,60,90,120,150,180,210,240,270,300,330]
        
        for angle in angle_list:
            img_r_tempt=img_rotate(img_pad_ad, angle, center=None, scale=1.0)
            label_r_tempt=img_rotate(label_pad_ad, angle, center=None,scale=1.0)

            label_r_gray = cv2.cvtColor(label_r_tempt, cv2.COLOR_BGR2GRAY)
            _, label_r_gray = cv2.threshold(label_r_gray, 20, 255, cv2.THRESH_BINARY)

            cv2.imwrite(img_extend_path+str(counter)+'.jpg', img_r_tempt)
            cv2.imwrite(label_extend_path+str(counter)+'.tif',label_r_gray)
            counter+=1
            print(counter)

            img_r_flip_tempt=img_rotate(img_pad_ad_flip, angle, center=None, scale=1.0)
            label_r_flip_tempt=img_rotate(label_pad_ad_flip, angle, center=None,scale=1.0)

            label_r_flip_gray = cv2.cvtColor(label_r_flip_tempt, cv2.COLOR_BGR2GRAY)
            _, label_r_flip_gray = cv2.threshold(label_r_flip_gray, 20, 255, cv2.THRESH_BINARY)

            cv2.imwrite(img_extend_path+str(counter)+'.jpg', img_r_flip_tempt)
            cv2.imwrite(label_extend_path+str(counter)+'.tif',label_r_flip_gray)
            counter+=1
            print(counter)

def img_and_label_crop(img,label,crop_size,stride):
    h, w, c = img.shape
    h1,w1,c1 = label.shape
    if (h, w, c)!=(h1,w1,c1):
        print('image and label not match')
        return
    h_number=(h-crop_size)//stride+1
    w_number=(w-crop_size)//stride+1

    img_crop_list=[]
    label_crop_list=[]
    for hi in range(h_number):
        for wi in range(w_number):
            img_crop_tempt=img[hi*stride:hi*stride+crop_size,wi*stride:wi*stride+crop_size,:]
            img_crop_list.append(img_crop_tempt)
            label_crop_tempt=label[hi*stride:hi*stride+crop_size,wi*stride:wi*stride+crop_size,:]
            label_crop_list.append(label_crop_tempt)
    return img_crop_list,label_crop_list

def crop_write(img_crop_list,label_crop_list):
    crop_number=len(img_crop_list)
    crop_number1=len(label_crop_list)
    img_crop_manage_list=[]
    label_crop_manage_list=[]
    label_bool_list=[]
    if crop_number!=crop_number1:
        print('img crop and label crop not match')
    positive_counter=0
    nagative_counter=0
    for i in range(crop_number): 
        if np.sum(label_crop_list[i])!=0:
            img_crop_manage_list.append(img_crop_list[i])
            label_crop_manage_list.append(label_crop_list[i])
            positive_counter=positive_counter+1
            label_bool_list.append(1)
        elif np.sum(np.sum(img_crop_list[i],axis=2)>100)>17000:

            if random.randint(0,3)==1:

                img_crop_manage_list.append(img_crop_list[i])
                label_crop_manage_list.append(label_crop_list[i])
                nagative_counter=nagative_counter+1
                label_bool_list.append(0)
            
    return img_crop_manage_list,label_crop_manage_list,positive_counter,nagative_counter,label_bool_list

def all_img_crop(img_path,label_path,img_save_path,label_save_path,crop_size=256,stride=128,fold_name='None'):
    counter=0
    img_name_list, label_name_list = generate_list(img_path,label_path)
    img_num=len(img_name_list)
    label_num=len(label_name_list)
    if img_num !=label_num:
        print('data & label not matched')
        return
    positive=0
    negative=0
    f=open(fold_name+'.lst','w')
    for i in range(img_num):
        print('croping '+str(i)+' img')
        img_t=cv2.imread(img_name_list[i])
        label_t=cv2.imread(label_name_list[i])
        img_crop_list, label_crop_list=img_and_label_crop(img_t,label_t,crop_size,stride)
        img_crop_manage_list,label_crop_manage_list,positive_counter,nagative_counter,label_bool_list=crop_write(img_crop_list,label_crop_list)
        positive=positive+positive_counter
        negative=negative+nagative_counter     
        for k in range(len(img_crop_manage_list)):
            img=img_crop_manage_list[k]
            cv2.imwrite(img_save_path+str(counter)+'.jpg',img)
            cv2.imwrite(label_save_path+str(counter)+'.tif', label_crop_manage_list[k])
            s=str(counter)+'\t'+str(label_bool_list[k])+'\t'+str(counter)+'.jpg'+ '\n'
            f.write(s)
            counter=counter+1
    f.close()
    print('positive_patch_num=',positive)
    print('negative_patch_num=',negative)

def show_img(img,label,pred):
    img=(img)*255
    img=img.asnumpy().transpose(1, 2, 0).astype('uint8')
    label=label.asnumpy().astype('uint8')
    pred=nd.sigmoid(pred)
    pred=(pred>0.5).asnumpy().astype('uint8')
    pred=pred.reshape(256,256)
    plt.figure(figsize=(16,9),dpi=98)
    p1=plt.subplot(311)
    p1.imshow(img)
    p1.set_title('original img')
    p2=plt.subplot(312)
    p2.imshow(label)
    p2.set_title('label')
    p3=plt.subplot(313)
    p3.imshow(pred)
    p3.set_title('pred')
    plt.show()
#patch predict
def patch_predict(img_patch,net,ctx):
    h, w, c=img_patch.shape
    img_patch=img_patch.astype(np.float32)/255
    data=nd.array(img_patch).transpose((2,0,1)).expand_dims(axis=0)
    yhat = net(data.as_in_context(ctx))###
    yhat=yhat.reshape((h,w))
    return yhat
    
#whole img predict,return predict mask
def one_img_predict(img,net,ctx,threshold=0.5):
    pred=patch_predict(img,net,ctx)
    pred=nd.sigmoid(pred)
    pred_out=pred>threshold
    return pred_out
def whole_img_evaluate(img_path,net,label_path):
    img_name_list,label_name_list=generate_list(img_path,label_path)
    acc_list=[]
    tpr_list=[]
    tnr_list=[]
    ppv_list=[]
    for i,img_name in enumerate(img_name_list):
        img=image.imread(img_name)
        mask=image.imread(label_name_list[i])
        label=image2label(mask)
        pred=one_img_predict(img,net,ctx=mx.gpu(0),threshold=0.5)
        print(i)
        TP=nd.sum(pred*label).asscalar()
        FP=nd.sum(pred-(pred*label)).asscalar()
        FN=nd.sum(label-(pred*label)).asscalar()
        TN=nd.sum((1-pred)*(1-label)).asscalar()
        
        ACC=(TP+TN)/(TP+TN+FP+FN)
        TPR=TP/ (TP+ FN)     
        TNR= TN/(FP+TN)
        PPV=TP/(TP+FP)
        acc_list.append(ACC)
        tpr_list.append(TPR)
        tnr_list.append(TNR)
        ppv_list.append(PPV)
    a=0
    for i in acc_list:
        a=a+i
    print('acc=',a/len(acc_list))
    a=0
    for i in tpr_list:
        a=a+i
    print('tpr=',a/len(tpr_list))
    a=0
    for i in tnr_list:
        a=a+i
    print('tnr=',a/len(tnr_list))    
    a=0
    for i in ppv_list:
        a=a+i
    print('ppv=',a/len(ppv_list))
        
        
    return acc_list,tpr_list,tnr_list,ppv_list    
def hard_exh_predict(img,net,crop_size,stride,ctx,threshold=0.5):
    h, w, c=img.shape
    h_number=(h-crop_size)//stride+1
    w_number=(w-crop_size)//stride+1
    crop_position=[]
    for hi in range(h_number):
        for wi in range(w_number):
            position_tempt=(hi*stride,wi*stride)
            crop_position.append(position_tempt)
            
    platform=nd.zeros(shape=(h,w))
    vote_num=nd.ones(shape=(h,w))*1e-5
    vote_patch=nd.ones(shape=(crop_size,crop_size))
    
    for position in crop_position:
        patch_tempt=img[position[0]:position[0]+crop_size,position[1]:position[1]+crop_size,:]

        pred=patch_predict(patch_tempt,net,ctx)
        pred=nd.sigmoid(pred)
        pred=pred.as_in_context(mx.cpu(0))
        x,y=position
        platform[x:x+crop_size,y:y+crop_size]+=pred
        vote_num[x:x+crop_size,y:y+crop_size]+=vote_patch
    pred_out=platform/vote_num
    prob_map=pred_out
    pred_out=pred_out>threshold
    return pred_out,prob_map
#save the predict mask
def predict_save(img_path,save_path,net,crop_size=256,stride=128,ctx=mx.gpu(0),threshold=0.5,mask_use=None):
    img_list=[]
    for r_d_f in os.walk(img_path):
        file_dir=r_d_f
    img_root=file_dir[0]
    img_name_list = file_dir[2]
    img_name_list.sort()
    for i in range(len(img_name_list)):
        img_list.append(img_root+img_name_list[i])
    for i,img in enumerate(img_list):
        img=image.imread(img)

        mask,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
        mask_v=1
        if mask_use:
            mask_v,d_mask,red,green=get_mask(img)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
            mask_v=cv2.erode(mask_v,kernel)
        prob_map=prob_map*255
        prob_map=prob_map.asnumpy().astype('uint8')*mask_v
        mask_img=label2image(mask*nd.array(mask_v))
        total_pred_im=mask_img.asnumpy().astype('uint8')
        R2,G2,B2=cv2.split(total_pred_im)
        total_pred_im=cv2.merge([B2, G2, R2])
        cv2.imwrite(save_path+img_name_list[i][0:len(img_name_list[i])-4]+'.png',total_pred_im)
        cv2.imwrite(save_path+img_name_list[i][0:len(img_name_list[i])-4]+'.tif',prob_map)
#evaluate
def evaluate(img_path,label_path,net,crop_size=256,stride=128,ctx=mx.gpu(0),threshold=0.5,disk_remove=False,net_vessel=None):
    img_name_list,label_name_list=generate_list(img_path,label_path)
    acc_list=[]
    tpr_list=[]
    tnr_list=[]
    ppv_list=[]
    F1_list= []
    
    for i,img_name in enumerate(img_name_list):
        img=image.imread(img_name)
        mask=image.imread(label_name_list[i])
        label=image2label(mask)
        pred,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
        if disk_remove:
            stamp,vessel=optic_disc_seg(img,net_vessel,crop_size=256,stride=128,ctx=ctx,threshold=0.5)
            pred=pred*(1-nd.array(stamp))*(1-vessel>0.5)
        TP=nd.sum(pred*label).asscalar()
        FP=nd.sum(pred-(pred*label)).asscalar()
        FN=nd.sum(label-(pred*label)).asscalar()
        TN=nd.sum((1-pred)*(1-label)).asscalar()
        
        ACC=(TP+TN)/(TP+TN+FP+FN+1e-12)
        TPR=TP/ (TP+ FN+1e-12)     
        TNR= TN/(FP+TN+1e-12)
        PPV=TP/(TP+FP+1e-12)
        F1=2*TPR*PPV/(TPR+PPV+1e-12)
        
        acc_list.append(ACC)
        tpr_list.append(TPR)
        tnr_list.append(TNR)
        ppv_list.append(PPV)
        F1_list.append(F1)
    a=0
    for i in acc_list:
        a=a+i
    ACC=a/len(acc_list)
    print('acc=',a/len(acc_list))
    a=0
    for i in tpr_list:
        a=a+i
    TPR=a/len(tpr_list)
    print('tpr=',a/len(tpr_list))
    a=0
    for i in tnr_list:
        a=a+i
    TNR=a/len(tnr_list)
    print('tnr=',a/len(tnr_list))    
    a=0
    for i in ppv_list:
        a=a+i
    PPV=a/len(ppv_list)
    print('ppv=',a/len(ppv_list))
    a=0
    for i in F1_list:
        a=a+i
    F1=a/len(F1_list)
    print('F1=',a/len(F1_list))
        
    return acc_list,tpr_list,tnr_list,ppv_list,ACC,TPR,TNR,PPV,F1
 
def evaluate_region(img_path,label_path,net,crop_size=256,stride=128,ctx=mx.gpu(0),threshold=0.5,disk_remove=False, net_vessel=None):
    img_name_list,label_name_list=generate_list(img_path,label_path)
    tpr_list=[]
    tnr_list=[]
    ppv_list=[]
    F1_list=[]
    for i,img_name in enumerate(img_name_list):
        img=image.imread(img_name)
        mask=image.imread(label_name_list[i])
        label=image2label(mask)
        pred,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
        if disk_remove:
            stamp,vessel=optic_disc_seg(img,net_vessel,crop_size=256,stride=128,ctx=ctx,threshold=0.5)
            pred=pred*(1-nd.array(stamp))*(1-vessel>0.5)
        
        TP,FP,FN=region_based_evaluation(label,pred)
        
        tp=np.sum(TP)
        fp=np.sum(FP)
        fn=np.sum(FN)
        tn=np.sum(np.ones(shape=(label.shape)))-tp-fp-fn
        
        
        TPR=tp/ (tp+ fn)     
        TNR= tn/(fp+tn)
        PPV=tp/(tp+fp)
        F1=2*TPR*PPV/(TPR+PPV+1e-12)
        F1_list.append(F1)
        tpr_list.append(TPR)
        tnr_list.append(TNR)
        ppv_list.append(PPV)

    a=0
    for i in tpr_list:
        a=a+i
    print('tpr=',a/len(tpr_list))
    a=0
    for i in tnr_list:
        a=a+i
    print('tnr=',a/len(tnr_list))    
    a=0
    for i in ppv_list:
        a=a+i
    print('ppv=',a/len(ppv_list))
    a=0
    for i in F1_list:
        a=a+i
    print('F1=',a/len(F1_list))
        
    return tpr_list,tnr_list,ppv_list,F1_list  
def imresize(img):
    pic = cv2.resize(img.asnumpy(), (img.shape[1]//2, img.shape[0]//2), interpolation=cv2.INTER_NEAREST)
    img=nd.array(pic)
    return img
def imresize_size(img,y,x):
    pic = cv2.resize(img.asnumpy(), (y, x), interpolation=cv2.INTER_NEAREST)
    img=nd.array(pic)
    return img    
def region_based_evaluation(label,pred):
    print("begin")
    label=label.asnumpy()
    pred=pred.asnumpy()
    label_area = la(label)
    pred_area = la(pred)

    M = np.max(label_area)
    N = np.max(pred_area)

    DUG = label * pred
    TP1 = DUG
    FP = np.zeros(shape=(label_area.shape))
    TP2 = np.zeros(shape=(label_area.shape))
    TP3 = np.zeros(shape=(label_area.shape))
    FN = np.zeros(shape=(label_area.shape))
    thta = 0.2
    for i in range(N):

        num = i + 1
        current_area_D = (pred_area == num)
        scale_Di = np.sum(current_area_D)
        DiUG = current_area_D * label
        scale_DiUG = np.sum(DiUG)
        if (scale_DiUG / scale_Di) > thta:
            TP2 = TP2 + current_area_D
        if (scale_DiUG / scale_Di) <= thta:
            tempt = current_area_D * (1 - label)
            FP = FP + tempt

    for i in range(M):
        num = i + 1
        current_area_G = (label_area == num)
        scale_Gi = np.sum(current_area_G)
        GiUD = current_area_G * pred
        scale_GiUD = np.sum(GiUD)
        if (scale_GiUD / scale_Gi) > thta:
            TP3 = TP3 + current_area_G
        if (scale_GiUD / scale_Gi) <= thta:
            tempt = current_area_G * (1 - pred)
            FN = FN + tempt
    TP = TP1 + TP2 + TP3
    TP = (TP > 0).astype('uint8')
    FP = (FP > 0).astype('uint8')
    FN = (FN > 0).astype('uint8')
    return TP,FP,FN    
    
def attention_map_po(amap,size):
    im=amap.reshape((size,size))
    im=im.asnumpy()*255
    im=im.astype('uint8')
    cv2.imwrite('kk.png',im)
    plt.imshow(im)
def attention_map_ch(cmap,size):
    one=nd.ones(shape=(size,size))
    im=cmap*one
    im=im.asnumpy()*255
    im=im.astype('uint8')
    im=cv2.applyColorMap(im, cv2.COLORMAP_JET)
    cv2.imwrite('kk1.png',im)
    plt.imshow(im)

def get_mask(img):
    red_channel=img[:,:,0]
    green_channel=img[:,:,1]
    mask=(red_channel>30).asnumpy()
    d_mask=2*np.sqrt(np.sum(mask)/3.14)  
    return mask,d_mask,red_channel.asnumpy(),green_channel.asnumpy()

def vessel_density(mask,d_mask,vessel):
    i=vessel.asnumpy()
    l=d_mask/4
    i=cv2.blur(i,(int(l),int(l)))
    return i

def intrest_region(red,green,mask,d_mask,vessel):
    l=int(d_mask/7)
    l1=int(d_mask/4)
    v_dens=vessel_density(mask,d_mask,vessel)
    v_dens=(v_dens-v_dens.min())/(v_dens.max()-v_dens.min())
    rg=green*0.5+red

    rg=(rg-rg.min())/(rg.max()-rg.min())*255
    m1=rg*(v_dens)
    m2=cv2.blur(m1,(int(l),int(l)))
    m3=(m2-m2.min())/(m2.max()-m2.min())
    m4=m1*(1+m3)
    m4_blur=cv2.blur(m4,(int(l),int(l)))
    m4_blur=(m4_blur-m4_blur.min())/(m4_blur.max()-m4_blur.min())
    region_map=rg*(1+m4_blur)
    region_map=255*(region_map-region_map.min())/(region_map.max()-region_map.min())
    region_blur=cv2.blur(region_map,(int(l),int(l)))
    x1,x2=np.where(region_blur>int(np.max(region_blur))-0.1)
    x1=int(x1.mean())
    x2=int(x2.mean())
    pad=np.zeros(red.shape)
    a,b=red.shape
    pad[max(0,x1-int(l1/2)):min(a,x1+int(l1/2)),max(0,x2-int(l1/2)):min(b,x2+int(l1/2))]=1
    region=region_map[max(0,x1-int(l1/2)):min(a,x1+int(l1/2)),max(0,x2-int(l1/2)):min(b,x2+int(l1/2))]
    vessel_region=vessel.asnumpy()[max(0,x1-int(l1/2)):min(a,x1+int(l1/2)),max(0,x2-int(l1/2)):min(b,x2+int(l1/2))]
    f_mask=mask*pad
    pad_region=f_mask[max(0,x1-int(l1/2)):min(a,x1+int(l1/2)),max(0,x2-int(l1/2)):min(b,x2+int(l1/2))]
    return pad_region,region,max(0,x1-int(l1/2)),min(a,x1+int(l1/2)),max(0,x2-int(l1/2)),min(b,x2+int(l1/2)),vessel_region

def mask_otsu(mask_pad,img):
    area=np.sum(mask_pad)
    g_max=0
    best_th=0
    u=np.sum(img*mask_pad)/np.sum(mask_pad)
    for th in range(int(u),256,1):
        bin_img = (img>th)*mask_pad
        bin_img_inv = (img<=th)*mask_pad
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if fore_pix==0:
            break
        if back_pix==0:
            continue
        W0=np.sum((img>th)*mask_pad)/area
        u0=np.sum(img*(img>th)*mask_pad)/np.sum((img>th)*mask_pad)
        W1=np.sum((img<=th)*mask_pad)/area
        u1=np.sum(img*(img<=th)*mask_pad)/np.sum((img<=th)*mask_pad)
        g=W0*W1*((u0-u1)**2)
        
        if g>g_max:
            g_max=g
            best_th=th
    
    im_result=(img>best_th)*mask_pad
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(50,50))
    opened = cv2.morphologyEx(im_result, cv2.MORPH_OPEN, kernel)
    return opened

def optic_disc_seg(img,net,crop_size=256,stride=128,ctx=mx.gpu(0),threshold=0.5):
    pred,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
    mask,d_mask,red,green=get_mask(img)
    stamp=np.zeros(red.shape)
    
    mask_pad,region,a,b,c,d,vessel_region=intrest_region(red,green,mask,d_mask,prob_map)
    v_mask=(vessel_region>0.1)
    mask1=(1-v_mask)*mask_pad
    region1=mask1*region
    r_blur=cv2.blur(region1,(50,50))
    m_blur=cv2.blur(mask1,(50,50))
    region_blur=r_blur/(m_blur+1e-10)
    region=region*(1-v_mask)+region_blur*v_mask
    im=mask_otsu(mask_pad,region)
    
    stamp[a:b,c:d]=im
    return stamp,prob_map

def img_extend_plus(img_path,csv_file,img_extend_path):
    df = pd.read_csv(csv_file,header=0)
    whole_data=[]
    for (name,series) in df.iterrows():
        name=name
        series=series
        img_name=series[0]
        hard=series[1]
        mic=series[2]
        heam=series[3]
        grade=series[4]       
        whole_data.append([img_name,hard,mic,heam,grade])
    img_name_list, label_name_list=generate_list(img_path,img_path)
    new_data=[]
    
    counter=0
    i1=len(img_name_list)
    i2=len(whole_data)
    if i1!=i2:
        print('something wrong with the data number')
        return
    for i in range(i1):
        img_name=img_name=img_name_list[i].split('/')[-1]
        if img_name.find(whole_data[i][0])==-1:
            print('warning! Data Matching Error')
            return 
        current_label=whole_data[i]
        
        img=cv2.imread(img_name_list[i])
        img_smaller = cv2.resize(img, (536, 356))
        img_pad_ad = img_pad(img_smaller)  # base1
        img_smaller = cv2.resize(img_pad_ad, (320, 320))
        img_pad_ad_flip = img_flip(img_smaller)  # base2
        angle_list=[0,30,60,90,120,150,180,210,240,270,300,330]        
        for angle in angle_list:
            img_r_tempt=img_rotate(img_pad_ad, angle, center=None, scale=1.0)
            a=current_label.copy()
            a[0]=str(counter)+'.jpg'
            new_data.append(a)
            cv2.imwrite(img_extend_path+str(counter)+'.jpg', img_r_tempt)
            counter+=1
            print(counter)

            img_r_flip_tempt=img_rotate(img_pad_ad_flip, angle, center=None, scale=1.0)
            b=current_label.copy()
            b[0]=str(counter)+'.jpg'
            new_data.append(b)
            cv2.imwrite(img_extend_path+str(counter)+'.jpg', img_r_flip_tempt)
            counter+=1
            print(counter)
    return new_data

def img_list(img_path):
    for r_d_f in os.walk(img_path):
        file_dir=r_d_f
    img_root=file_dir[0]
    img_name_list = file_dir[2]
    img_name_list.sort()
    for i in range(len(img_name_list)):
        img_name_list[i]=img_root+img_name_list[i]
    return img_name_list

def twotime(img_name,time=3):
    img=cv2.imread(img_name)
    x,y,z=img.shape
    img_resize=cv2.resize(img,(y*time,x*time),interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(img_name,img_resize)

def mask_evaluate(img_path,label_path,mask_path,net,crop_size=256,stride=128,over_TN=0,ctx=mx.gpu(0),threshold=0.5,disk_remove=False,net_vessel=None):
    img_name_list,label_name_list=generate_list(img_path,label_path)
    mask_list,_=generate_list(mask_path,mask_path)
    acc_list=[]
    tpr_list=[]
    tnr_list=[]
    ppv_list=[]
    F1_list= []
    AUC_list=[]
    for i,img_name in enumerate(img_name_list):
        print(i)
        img=image.imread(img_name)
        mask=image.imread(label_name_list[i])
        mask1=image.imread(mask_list[i])
        label=image2label(mask)
        mask1=image2label(mask1)
        
        
        
        pred,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
        pred=pred*mask1
        prob_map=prob_map*mask1
        
        score2 = prob_map.asnumpy().ravel().tolist()
        label2 = label.asnumpy().ravel().tolist()
        mask2 = mask1.asnumpy().ravel().tolist()

        pred_FOV = [i for i,k in zip(score2,mask2) if k==1]
        label_FOV = [j for j,k in zip(label2,mask2) if k==1]  
    
        pred_FOV = np.array(pred_FOV)         
        label_FOV = np.array(label_FOV)
        
        auc = roc_auc_score(label_FOV,pred_FOV)
        
        AUC_list.append(auc)
        if disk_remove:
            stamp,vessel=optic_disc_seg(img,net_vessel,crop_size=256,stride=128,ctx=ctx,threshold=0.5)
            pred=pred*(1-nd.array(stamp))*(1-vessel>0.5)
        TP=nd.sum(pred*label).asscalar()
        FP=nd.sum(pred-(pred*label)).asscalar()
        FN=nd.sum(label-(pred*label)).asscalar()
        TN=nd.sum((1-pred)*(1-label)).asscalar()
        TN=TN-over_TN
        ACC=(TP+TN)/(TP+TN+FP+FN+1e-12)
        TPR=TP/ (TP+ FN+1e-12)     
        TNR= TN/(FP+TN+1e-12)
        PPV=TP/(TP+FP+1e-12)
        F1=2*TPR*PPV/(TPR+PPV+1e-12)
        
        acc_list.append(ACC)
        tpr_list.append(TPR)
        tnr_list.append(TNR)
        ppv_list.append(PPV)
        F1_list.append(F1)
    a=0
    for i in acc_list:
        a=a+i
    ACC=a/len(acc_list)
    print('acc=',a/len(acc_list))
    a=0
    for i in tpr_list:
        a=a+i
    TPR=a/len(tpr_list)
    print('tpr=',a/len(tpr_list))
    a=0
    for i in tnr_list:
        a=a+i
    TNR=a/len(tnr_list)
    print('tnr=',a/len(tnr_list))    
    a=0
    for i in ppv_list:
        a=a+i
    PPV=a/len(ppv_list)
    print('ppv=',a/len(ppv_list))
    a=0
    for i in F1_list:
        a=a+i
    F1=a/len(F1_list)
    print('F1=',a/len(F1_list))
    
    a=0
    for i in AUC_list:
        a=a+i
    AUC=a/len(AUC_list)
    print('AUC=',a/len(AUC_list))
    return acc_list,tpr_list,tnr_list,ppv_list,AUC_list,ACC,TPR,TNR,PPV,F1,AUC
    
    
def prob_curv(pred,label,mask):
    lab1=label
    lab2=1-label
    list1=[]
    list2=[]
    pred=(pred*100).astype('int')
    for i in range(100):
        list1.append(np.sum(mask*lab1*(pred==i)))
        list2.append(np.sum(mask*lab2*(pred==i)))
    return list1,list2
def pred_curv(img_path,label_path,net,crop_size=256,stride=128,ctx=mx.gpu(0),threshold=0.5,disk_remove=False, net_vessel=None):
    img_name_list,label_name_list=generate_list(img_path,label_path)
    array1=np.zeros(100)
    array2=np.zeros(100)
    for i,img_name in enumerate(img_name_list):
        print(img_name)
        img=image.imread(img_name)
        mask=image.imread(label_name_list[i])
        label=image2label(mask)
        pred,prob_map=hard_exh_predict(img,net,crop_size,stride,ctx,threshold)
        mask,d_mask,red,green=get_mask(img)
        prob=prob_map.asnumpy()
        label=label.asnumpy()
        list1,list2=prob_curv(prob,label,mask)
        array1=array1+np.array(list1)
        array2=array2+np.array(list2)
    return array1,array2