import cv2
import h5py
import numpy as np
from mxnet import nd

from mxnet import image
import tools_for_ex_seg as tools
classes=['background','target']
colormap=[[0,0,0],[255,255,255]]
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i

def image2label(im):
    data=im.astype('int32').asnumpy()
    idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])

img_path="/data/dataset/SiZe/hard_e_datasets/train_patch/"
label_path="/data/dataset/SiZe/hard_e_datasets/train_label_patch/"
img_name_list,label_name_list=tools.generate_list(img_path,label_path)

def creat_h5_file(img_name_list,label_name_list):
    for times in range(101):
        print(times)
        if times==0:
            h5f=h5py.File("train_data.h5",'w')
            x=h5f.create_dataset("data",(1000,3,256,256),maxshape=(None,3,256,256),dtype=np.uint8)
            y=h5f.create_dataset("label",(1000,256,256),maxshape=(None,256,256),dtype=np.float32)
        h5f=h5py.File("train_data.h5","a") #add mode
        x=h5f["data"]
        y=h5f["label"]
        if times!=100:
            x.resize([times*1000+1000,3,256,256])
            y.resize([times*1000+1000,256,256])

            index=times*1000
            data_zeros=np.zeros(shape=(1000,3,256,256))
            label_zeros=np.zeros(shape=(1000,256,256))
            for k, i in enumerate(range(index,index+1000)):
                img_t=image.imread(img_name_list[i])
                img_t=img_t.asnumpy().transpose((2,0,1))
                data_zeros[k, :, :, :] = img_t

                label_t=image.imread(label_name_list[i])
                label_t=image2label(label_t)
                label_zeros[k,:,:]=label_t.asnumpy()
     
            x[times * 1000:times * 1000 + 1000,:,:,:] = data_zeros
            y[times * 1000:times * 1000 + 1000,:,:] = label_zeros
          
        else:
            print('the last one')
            x.resize([times*1000+566,3,256,256])
            y.resize([times*1000+566,256,256])
            index=times*1000

            data_zeros=np.zeros(shape=(566,3,256,256))
            label_zeros=np.zeros(shape=(566,256,256))
            for k, i in enumerate(range(index,index+566)):
                img_t= image.imread(img_name_list[i])
                img_t=img_t.asnumpy().transpose((2,0,1))
                data_zeros[k,:,:,:]=img_t

                label_t=image.imread(label_name_list[i])
                label_t=image2label(label_t)
                label_zeros[k,:,:]=label_t.asnumpy()
            x[times * 1000:times * 1000 + 566,:,:,:] = data_zeros
            y[times * 1000:times * 1000 + 566,:,:] = label_zeros
    h5f.close()
    print("the end")       