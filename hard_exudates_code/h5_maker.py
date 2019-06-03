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

def h5_maker(img_path,label_path):
    img_name_list, label_name_list = tools.generate_list(img_path,label_path)
    num1=len(img_name_list)
    num2=len(label_name_list)
    if num1!=num2:
        print('data and label is not match')
        return
    img_list=[]
    label_list=[]

    for i in range(num1):
        img_t=image.imread(img_name_list[i])
        img_t=img_t.transpose((2,0,1))
        img_list.append(img_t)
        label_img_t=image.imread(label_name_list[i])
        label_t=image2label(label_img_t)
        label_list.append(label_t)

    with h5py.File("exudates.h5", "w") as f:
        f.create_dataset("img", data=img_list)
        f.create_dataset("label", data=label_list)