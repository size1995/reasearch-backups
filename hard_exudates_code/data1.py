import mxnet as mx # used version '1.0.0' at time of writing
from mxnet.gluon.data import dataset
import os
import numpy as np
from mxnet import nd
mx.random.seed(42) # set seed for repeatability
import random
 

class ImageWithMaskDataset(dataset.Dataset):
    def __init__(self, img_root,label_root,transform=None):
        self._img_root = img_root
        self._label_root= label_root

        self._transform = transform
        self._list_images(self._img_root)
        self._list_labels(self._label_root)

    def _list_images(self, img_root):
        _image_list=[]
        for filename in sorted(os.listdir(img_root)):
            _image_list.append(filename)
        self._image_list =_image_list

    def _list_labels(self,label_root):
        _label_list=[]
        for filename in sorted(os.listdir(label_root)):
            _label_list.append(filename)
        self._label_list=_label_list

    def __getitem__(self, idx):
        image_filepath = os.path.join(self._img_root, self._image_list[idx])
        image = mx.image.imread(image_filepath)
        label_filepath = os.path.join(self._label_root, self._label_list[idx])
        label = mx.image.imread(label_filepath)

        if self._transform is not None:
            return self._transform(image, label)
        else:
            return image, label

    def __len__(self):
        return len(self._image_list)

classes=['background','target']
colormap=[[0,0,0],[255,255,255]]
cm2lbl=np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]]=i

def image2label(im):
    data=im.astype('int32').asnumpy()
    idx=(data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return nd.array(cm2lbl[idx])


def img_transform(image, label):
    image = image.astype(np.float32)/127.5 - 1
    image = nd.transpose(image, (2, 0, 1))
    label =image2label(label)
    c,w,h=image.shape
    x=random.randint(0,w-1024)
    y=random.randint(0,w-1024)
    image=image[:,x:x+1024,y:y+1024]
    label=label[x:x+1024,y:y+1024]
    return image, label

    

def get_dataset(img_root,label_root):
    dataset = ImageWithMaskDataset(img_root, label_root, transform=img_transform)
    return dataset
