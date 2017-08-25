from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import matplotlib.pyplot as plt

sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug = iaa.SomeOf(1,[
        sometimes(iaa.OneOf([
            iaa.Dropout((0.01, 0.1)),
            iaa.CoarseDropout((0.03, 0.07),
                              size_percent=(0.03, 0.15)),
        ])),
        sometimes(iaa.DirectedEdgeDetect(0.4)),
        sometimes(iaa.AdditiveGaussianNoise(loc=0,
                                            scale=(0.3, 0.7))),
        sometimes(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
        sometimes(iaa.Add((-7, 7)))])

def show(image, cmap='gray'):
    plt.figure()
    plt.imshow(image[:,:,::-1].astype('uint8'), cmap=cmap)

top_left = (100, 190)
bottom_right = (520, 350)
h, w = 66, 200
for i, name in enumerate(['openhighway.png', 'nolane.png', 'highway.png', 'street.png']):
    # original shape (480, 640, 3)
    img = cv2.imread('./samples/' + name)
    print('original image shape', img.shape)
    # 120 320
    crop = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    resize = cv2.resize(crop.copy(), (w, h))
    cv2.rectangle(img,(100,190),(520,350),(0,0,255),2)
    print('cropped image shape: ', crop.shape)
    # print('resized image shape: ', resize.shape)
    # cv2.imshow('image',resize)
    # cv2.imshow('cropped', crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('%s_rect.png' % i,img)
    cv2.imwrite('%s_resize.png' % i,resize)


# show(image)
# show(crop_img)
# show(resize)
# show(img)
# plt.show()



# cv2.normalize(resize,resize,0,1,cv2.NORM_MINMAX)

# show an image with 8*8 augmented versions of image 0
# ia.show_grid(img, cols=8, rows=8)
