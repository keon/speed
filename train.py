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
from utils import init_datasets, init_models


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model', type=str, default='alexlstm')
    p.add_argument('--restore', type=bool, default=False)
    p.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    p.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    p.add_argument('-e', '--epochs', type=int, default=10, help='epoch size')
    p.add_argument('-t', '--timesteps', type=int, default=16,
                   help='number of frames fed to the model')
    p.add_argument('--train_folder', type=str, default='data/train/')
    p.add_argument('--valid_folder', type=str, default='data/valid/')
    p.add_argument('-v', '--verbose', type=bool, default=True,
                   help='verbose mode: prints out the loss for every epoch')
    return p.parse_args()


def train(e, model, opt, dataset, arg, cuda=False):
    model.train()
    criterion = nn.MSELoss()
    losses = []

    batcher = dataset.get_batcher(shuffle=True, augment=True)
    for b, (x, y) in enumerate(batcher, 1):
        x = V(th.from_numpy(x).float()).cuda()
        y = V(th.from_numpy(y).float()).cuda()
        opt.zero_grad()
        logit = model(x)
        loss = criterion(logit, y)
        loss.backward()
        opt.step()

        losses.append(loss.data[0])
        if arg.verbose and b % 50 == 0:
            loss_t = np.mean(losses[:-49])
            print('[train] [e]:%s [b]:%s - [loss]:%s' % (e, b, loss_t))
    return losses

def validate(models, dataset, arg, cuda=False):
    criterion = nn.MSELoss()
    losses = []
    batcher = dataset.get_batcher(shuffle=True, augment=False)
    for b, (x, y) in enumerate(batcher, 1):
        x = V(th.from_numpy(x).float()).cuda()
        y = V(th.from_numpy(y).float()).cuda()
        # Ensemble average
        logit = None
        for model, _ in models:
            model.eval()
            logit = model(x) if logit is None else logit + model(x)
        logit = th.div(logit, len(models))
        loss = criterion(logit, y)
        losses.append(loss.data[0])
    return np.mean(losses)


def main(arg):
    # initialize datasets for training and v_datasets for validation
    resize = (200, 66)
    datasets, v_dataset = init_datasets(arg, resize, 3)

    # initialize models
    cuda = th.cuda.is_available()
    models = init_models(arg.model, 3, arg.lr, arg.restore, cuda)

    # Initiate Training
    t0 = datetime.datetime.now()
    try:
        t_losses = [[],[],[]]
        v_losses = []
        for e in range(arg.epochs):
            i = 0
            for (model, opt), dataset in zip(models, datasets):
                print('training model %d' % i)
                losses = train(e, model, opt, dataset, arg, cuda=cuda)
                th.save(model.state_dict(), './save/%s%d_%s.p' % (arg.model, i, e))
                t_losses[i] += losses
                v = validate([[model, None]], v_dataset, arg, cuda=cuda)
                print('[!] model %d - validation loss: %s' % (i, v))
                i += 1
            v_loss = validate(models, v_dataset, arg, cuda=cuda)
            v_losses.append(v_loss)
            print('[valid] [e]:%s - [loss]:%s' % (e, v_loss))
    except KeyboardInterrupt:
        print('[!] KeyboardInterrupt: Stopped Training...')

    # save stuffs
    pkl.dump(t_losses, open('./save/%s_t_loss.p' % arg.model, 'wb'))
    pkl.dump(v_losses, open('./save/%s_v_loss.p' % arg.model, 'wb'))
    for i, (m, _) in enumerate(iter(models)):
        th.save(m.state_dict(), './save/%s%d.p' % (arg.model, i))
    t1 = datetime.datetime.now()
    print('[!] Finished Training, Time Taken4 %s' % (t1-t0))

if __name__ == '__main__':
    arg = parse_arguments()
    main(arg)
