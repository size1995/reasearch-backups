import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('134.jpg')
mask = cv2.imread('134.tif')
def affine_transform(img):
    h, w, c = img.shape
    red = (255, 0, 0)
    for i in range(10):
        cv2.line(img, (i*80, 0), (i*80, h), red, 2)
        cv2.line(img, (0, i*80), (w, i*80),red, 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = img.shape
    shape_size = shape[:2]
    # Random affine
    alpha_affine = 50
    random_state = np.random.RandomState()
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                                size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    plt.figure(figsize=(8, 7), dpi=98)
    p1 = plt.subplot(211)
    p1.imshow(img)
    p1.set_title('Input')

    p2 = plt.subplot(212)
    p2.imshow(image)
    p2.set_title('Output')

    plt.show()


def test_affine_transform():
    img = cv2.imread('134.jpg')
    rows, cols, ch = img.shape
    img = cv2.resize(img,(rows,rows))

    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[0, 0], [200, 0], [28, 200], [300, 300]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (rows, rows))

    plt.figure(figsize=(8, 7), dpi=98)
    p1 = plt.subplot(211)
    p1.imshow(img)
    p1.set_title('Input')

    p2 = plt.subplot(212)
    p2.imshow(dst)
    p2.set_title('Output')

    plt.show()


def test_random():

    for i in range(1):
        state = np.random.RandomState(1)
        arr = state.uniform(0, 1, 5)
        print(arr,'\n')

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



def Elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState()
    shape = image.shape
    shape_size = shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape),\
           map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

# image = Elastic_transform(img,120,6)
def plot(img,image):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(16, 9), dpi=98)
    p1 = plt.subplot(211)
    p1.imshow(img)
    p1.set_title('Input')

    p2 = plt.subplot(212)
    p2.imshow(image)
    p2.set_title('Output')

    plt.show()

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    h, w, c = image.shape
    red = (255, 0, 0)
    for i in range(10):
        cv2.line(image, (i*80, 0), (i*80, h), red, 2)
        cv2.line(image, (0, i*80), (w, i*80),red, 2)

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

#image = elastic_transform(img,500,15,120)
#plot(img,image)

def elastic_demo(image,label_mask,alpha, sigma):

    h, w, c = image.shape
    white = (255, 255, 255)
    for i in range(10):
        cv2.line(image, (i*80, 0), (i*80, h), white, 2)
        cv2.line(image, (0, i*80), (w, i*80),white, 2)
    for i in range(10):
        cv2.line(label_mask, (i*80, 0), (i*80, h), white, 2)
        cv2.line(label_mask, (0, i*80), (w, i*80),white, 2)

    random_state = np.random.RandomState()
    shape = image.shape
    shape_size = shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    image_elastic= map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    label_mask_elastic= map_coordinates(label_mask, indices, order=1, mode='reflect').reshape(shape)

    label_mask_elastic_gray = cv2.cvtColor(label_mask_elastic,cv2.COLOR_BGR2GRAY)
    _,label_mask_elastic_threshold = cv2.threshold(label_mask_elastic_gray,20,255,cv2.THRESH_BINARY)

    img_show = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#
    img_e_show = cv2.cvtColor(image_elastic, cv2.COLOR_BGR2RGB)#

    label_mask_show=cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
    _,label_mask_show_threshold = cv2.threshold(label_mask_show ,20,255,cv2.THRESH_BINARY)
    label_mask_show_RGB= cv2.cvtColor(label_mask_show_threshold, cv2.COLOR_GRAY2RGB)#
    label_mask_elastic_RGB = cv2.cvtColor(label_mask_elastic_threshold, cv2.COLOR_GRAY2RGB)#

    plt.figure(figsize=(16, 9), dpi=98)
    p1 = plt.subplot(221)
    p1.imshow(img_show)
    p1.set_title('original img')

    p2 = plt.subplot(222)
    p2.imshow(label_mask_show_RGB)
    p2.set_title('original mask')

    p3 = plt.subplot(223)
    p3.imshow(img_e_show)
    p3.set_title('elastic img')

    p4 = plt.subplot(224)
    p4.imshow(label_mask_elastic_RGB)
    p4.set_title('elastic mask')

    plt.show()
