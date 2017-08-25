from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import argparse
from collections import defaultdict
import torch as th
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable as V
import pickle as pkl
import numpy as np
from dataset import Dataset
from models import MiniC3D, AlexLSTM, DenseLSTM
import cv2


def init_datasets(arg, resize, n):
    """ Initialize N number of datasets for ensemble training """
    datasets = []
    for i in range(n):
        dset = Dataset(arg.train_folder,
                       resize=resize,
                       batch_size=arg.batch_size,
                       timesteps=arg.timesteps,
                       windowsteps=arg.timesteps // 2, shift=i*2, train=True)
        print('[!] train dataset len: %d - shift: %d' % (len(dset.data), i*2))
        datasets.append(dset)
    # Validation Dataset
    v_dataset = Dataset(arg.valid_folder,
                        resize=resize,
                        batch_size=arg.batch_size//2,
                        timesteps=arg.timesteps,
                        windowsteps=arg.timesteps //2, shift=0, train=True)
    print('[!] validation dataset samples: %d' % len(v_dataset.data))
    return datasets, v_dataset

def init_models(model_name, n, lr, restore=False, cuda=False):
    """ Initialize N number of models and optimizers for ensemble """
    models = []
    for i in range(1, n+1):
        if model_name == 'minic3d':
            m = MiniC3D()
        elif model_name == 'alexlstm':
            m = AlexLSTM()
        elif model_name == 'denselstm':
            m = DenseLSTM()
        if restore:
            print('[!] Using pretrained model...')
            m.load_state_dict(th.load('./save/%s1.p' % model_name))
        if cuda:
            print('[!] Using cuda...')
            m = m.cuda()
        # initialize optimizer and scheduler. lr decay scheduler is optional.
        optimizer = optim.Adam(m.parameters(), lr=lr)
        models.append([m, optimizer])
    print('[!] Model Summary: ', models[0])
    return models
