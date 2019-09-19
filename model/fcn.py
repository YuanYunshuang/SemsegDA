# FCN Small

import functools
import torch
import torch.nn as nn
from loss import cross_entropy2d
import matplotlib.pylab as plt


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)


class FCN(nn.Module):
    def __init__(self, n_classes=2):
        super(FCN, self).__init__()
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d)
        self.encoderDSM = nn.Sequential(
            self.conv_block1(1, 32),
            self.conv_block2(32, 32),
            self.conv_block3(32, 64),
            self.conv_block2(64, 64),
            self.conv_block3(64, 128),
            self.conv_block2(128, 128),
            self.conv_block3(128, 256),
            self.conv_block2(256, 256),
            self.conv_block3(256, 384),
        )

        self.decoder = nn.Sequential(
            self.tr_conv_block1(384, 256),
            self.tr_conv_block2(256, 256),
            self.tr_conv_block1(256, 256),
            self.tr_conv_block2(256, 128),
            self.tr_conv_block1(128, 128),
            self.tr_conv_block2(128, 128),
            self.tr_conv_block1(128, 64),
            self.tr_conv_block2(64, 64),
            self.tr_conv_block3(64, 64),
            nn.Conv2d(64, self.n_classes, 1, padding=0),
        )

        self.softmax = nn.Softmax2d()


    def conv_block1(self, channel_in, channels_out):

        return nn.Sequential(
                nn.Conv2d(channel_in, channels_out, 5, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def conv_block2(self, channel_in, channels_out):

        return nn.Sequential(
                nn.Conv2d(channel_in, channels_out, 2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def conv_block3(self, channel_in, channels_out):

        return nn.Sequential(
                nn.Conv2d(channel_in, channels_out, 3, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def tr_conv_block1(self, channel_in, channels_out):

        return nn.Sequential(
                nn.ConvTranspose2d(channel_in, channels_out, 3, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def tr_conv_block2(self, channel_in, channels_out):

        return nn.Sequential(
                nn.ConvTranspose2d(channel_in, channels_out, 2, stride=2, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def tr_conv_block3(self, channel_in, channels_out):

        return nn.Sequential(
                nn.ConvTranspose2d(channel_in, channels_out, 5, padding=0),
                nn.Dropout2d(0.1),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        dsm_encoded = self.encoderDSM(x)
        out = self.decoder(dsm_encoded)
        out = self.softmax(out)

        return out
