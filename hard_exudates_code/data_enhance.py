import mxnet as mx # used version '1.0.0' at time of writing
from mxnet.gluon.data import dataset
import os
import numpy as np
from mxnet import nd
mx.random.seed(42) # set seed for repeatability
import cv2

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

def img_transform(image, label):
    image = image.asnumpy()
    image=img_enhance(image)
    image=nd.array(image)
    image = image.astype(np.float32)/127.5 - 1
    image = nd.transpose(image, (2, 0, 1))
    label =image2label(label)
    return image, label

def get_dataset(img_root,label_root):
    dataset = ImageWithMaskDataset(img_root, label_root, transform=img_transform)
    return dataset
