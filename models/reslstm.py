import torch.nn as nn
from torch.autograd import Variable as V
import torch as th
from torchvision import models


class ResLSTM(nn.Module):
    def __init__(self, n_layers=2, h_size=512):
        super(ResLSTM, self).__init__()
        print('Building AlexNet + LSTM model...')
        self.h_size = h_size
        self.n_layers = n_layers

        resnet = models.resnet50(pretrained=True)
        self.conv = nn.Sequential(*list(resnet.children())[:-1])

        self.lstm = nn.LSTM(1280, h_size, dropout=0.2, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(h_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, timesteps = x.size()[0], x.size()[2]
        state = self._init_state(b_size=batch_size)

        convs = []
        for t in range(timesteps):
            conv = self.conv(x[:, :, t, :, :])
            conv = conv.view(batch_size, -1)
            convs.append(conv)
        convs = th.stack(convs, 0)
        lstm, _ = self.lstm(convs, state)
        logit = self.fc(lstm[-1])

        return logit

    def _init_state(self, b_size=1):
        weight = next(self.parameters()).data
        return (
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01)),
            V(weight.new(self.n_layers, b_size, self.h_size).normal_(0.0, 0.01))
        )
