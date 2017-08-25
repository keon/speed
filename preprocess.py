from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np

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


def optical_flow(one, two):
    """
    method taken from (https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4)
    """
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((120, 320, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2,
                                        poly_n=5, poly_sigma=1.1, flags=0)
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # convert HSV to int32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return rgb_flow
