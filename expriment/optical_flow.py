from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import matplotlib.pyplot as plt

def optical_flow(one, two):
    """
    method taken from https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    """
    one_g = cv2.cvtColor(one, cv2.COLOR_RGB2GRAY)
    two_g = cv2.cvtColor(two, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((120, 320, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(two, cv2.COLOR_RGB2HSV)[:,:,1]
    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(one_g, two_g, flow=None,
                                        pyr_scale=0.5,
                                        levels=1,
                                        winsize=10,
                                        iterations=2,
                                        poly_n=5,
                                        poly_sigma=1.1,
                                        flags=0)
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

h, w = 80, 240
for i in range(1, 9):
    c = cv2.imread('./data/00000%s.png' % i)
    n = cv2.imread('./data/00000%s.png' % str(i+1))

    c = c[180:300, 160:480]
    n = n[180:300, 160:480]


    o = optical_flow(c, n)

    # c = cv2.resize(c.copy(), (w, h))
    # n = cv2.resize(n.copy(), (w, h))
    o = cv2.resize(o.copy(), (w, h))
    cv2.imwrite('op%s.png' % i, o)
