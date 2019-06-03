import numpy as np
from mxnet import nd
from mxnet import gluon
from mxnet import image
import tools_for_ex_seg as tools
import mxnet as mx
classes=['background','target']
colormap=[[0,0,0],[255,255,255]]
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i

def image2label(im):
    data=im.astype('int32').asnumpy()
    idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])

def read_images(img_path="/data/dataset/SiZe/hard_e_datasets/train_patch/",
                label_path="/data/dataset/SiZe/hard_e_datasets/train_label_patch/"):
    img_name_list,label_name_list=tools.generate_list(img_path,label_path)
    n=len(img_name_list)
    n1=len(label_path)
    data,label=[None]*(n//4),[None]*(n//4)  
    for i in range(n//4):
        print(i)
        data[i]=image.imread(img_name_list[i])
        label[i]=image.imread(label_name_list[i])
    return data,label

def normalize_image(data):
    return(data.astype('float32')/255)
    
class IDRIDDATASET(gluon.data.Dataset):
    def __init__(self,train):
        data,label=read_images()
        self.data=[normalize_image(im) for im in data]
        self.label=label
        print('read'+str(len(self.data))+'examples')
    def __getitem__(self,idx):
        data,label=self.data[idx],self.label[idx]
        data=data.transpose((2,0,1))
        label=image2label(label)
        return data,label
    def __len__(self):
        return len(self.data)

def load_data(batch_size, is_reversed=False,
              img_path="/data/dataset/SiZe/hard_e_datasets/train_patch/",
                label_path="/data/dataset/SiZe/hard_e_datasets/train_label_patch/"):
    img_in_list = []
    img_out_list = []
    
    img_name_list,label_name_list=tools.generate_list(img_path,label_path)
    n=len(img_name_list)
      
    for i in range(n//20):
        print(i)
        img_in=image.imread(img_name_list[i]).astype(np.float32)/127.5 - 1
        label_out=image.imread(label_name_list[i])
        label_out=image2label(label_out)
        
        img_in=nd.transpose(img_in, (2,0,1))
        img_in=img_in.reshape((1,)+img_in.shape)
        label_out=label_out.reshape((1,)+label_out.shape)
        
        img_in_list.append(img_in)
        img_out_list.append(label_out)
    return mx.io.NDArrayIter(data=[nd.concat(*img_in_list, dim=0), nd.concat(*img_out_list, dim=0)],
                             batch_size=batch_size)