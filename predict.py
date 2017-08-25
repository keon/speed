from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import argparse
import torch as th
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd import Variable as V
import pickle as pkl
import numpy as np
from dataset import Dataset
from models import MiniC3D, AlexLSTM, DenseLSTM
from utils import init_models
import cv2


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', type=str, default='alexlstm')
    p.add_argument('-t', '--timesteps', type=int, default=10,
                   help='number of frames fed to the model')
    p.add_argument('-v', '--verbose', type=bool, default=True,
                   help='verbose mode: prints out the loss for every epoch')
    p.add_argument('--test_folder', type=str, default='data/test/')
    return p.parse_args()

def predict(models, dataset, arg, cuda=False):
    prediction_file = open('save/predictions.txt', 'w')
    batcher = dataset.get_batcher(shuffle=False, augment=False)
    for b, (x, _) in enumerate(batcher, 1):
        x = V(th.from_numpy(x).float()).cuda()
        # Ensemble average
        logit = None
        for model, _ in models:
            model.eval()
            logit = model(x) if logit is None else logit + model(x)
        logit = th.div(logit, len(models))
        prediction = logit.cpu().data[0][0]
        prediction_file.write('%s\n' % prediction)
        if arg.verbose and b % 100 == 0:
            print('[predict] [b]:%s - prediction: %s' % (b, prediction))
    # prediction_file.close()

def main(arg):
    resize = (200, 66)

    # initialize dataset
    dataset = Dataset(arg.test_folder,
                      resize=resize,
                      batch_size=1,
                      timesteps=arg.timesteps,
                      windowsteps=1,
                      shift=0,
                      train=False)
    print('[!] testing dataset samples: %d' % len(dataset.data))

    # initialize model
    cuda = th.cuda.is_available()
    models = init_models(arg.model, n=3, lr=0, restore=True, cuda=cuda)

    # Initiate Prediction
    t0 = datetime.datetime.now()
    try:
        predict(models, dataset, arg, cuda=cuda)
    except KeyboardInterrupt:
        print('[!] KeyboardInterrupt: Stopped Training...')
    t1 = datetime.datetime.now()

    print('[!] Finished Training, Time Taken4 %s' % (t1-t0))

if __name__ == '__main__':
    arg = parse_arguments()
    main(arg)
