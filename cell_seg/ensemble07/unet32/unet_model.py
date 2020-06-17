""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x) #[b, 64, h, w]
        x2 = self.down1(x1) #[b, 128, h/2, w/2]
        x3 = self.down2(x2) #[b, 256, h/4, w/4]
        x4 = self.down3(x3) #[b, 512, h/8, w/8]
        x5 = self.down4(x4) #[b, 512, h/16, w/16]
        x = self.up1(x5, x4) #[b, 256, h/8, w/8]
        x = self.up2(x, x3) #[b, 128, h/4, w/4]
        x = self.up3(x, x2) #[b, 64, h/2, w/2]
        x = self.up4(x, x1) #[b, 64, h, w]
        logits = self.outc(x) #[b, 1, h, w]
        return logits
