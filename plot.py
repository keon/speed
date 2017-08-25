import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import plotly.plotly as py


def plot_labels():
    with open('train.txt') as f:
        data = f.read()
    y = [float(row) for row in data.split('\n')]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Speed per Frame')
    ax1.set_xlabel('Frames')
    ax1.plot(y, c='r', label='training loss')
    leg = ax1.legend()
    plt.show()

def plot_val_loss():
    valid_loss = []
    for e in range(19):
        with open('./save/conv3d_val_loss_%s.p' % e, 'rb') as f:
            data = pkl.load(f)
            valid_loss.append(np.mean(data))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Validation Loss')
    ax1.set_xlabel('epochs')
    ax1.plot(valid_loss[10:], c='r', label='validation loss')
    plt.show()

def plot_train_loss(smooth=False):
    train_loss = []
    for e in range(19):
        with open('./save/conv3d_train_loss_%s.p' % e, 'rb') as f:
            data = pkl.load(f)
            train_loss += data
    # if smooth:
    #     train_loss = np.linspace(min(train_loss), max(train_loss), 100000)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('batchs')
    ax1.plot(train_loss, c='r', label='speed')
    plt.show()
# plot_loss('conv3d_val_loss_0')
plot_val_loss()
# plot_train_loss(smooth=True)
