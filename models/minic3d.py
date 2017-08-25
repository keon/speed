import torch.nn as nn


class MiniC3D(nn.Module):
    """
    Mini version of the model described in https://arxiv.org/abs/1412.0767
    """
    def __init__(self):
        super(MiniC3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(3, 64, 7, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Conv3d(64, 128, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(128, 256, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(256, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, (3, 3, 3), padding=2),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(512, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, (3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(6144, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1),
        )


    def forward(self, x):
        conv = self.conv(x)
        conv = conv.view(conv.size(0), -1)
        logit = self.fc(conv)
        return logit
