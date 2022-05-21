from __future__ import absolute_import

import math
import torch.nn as nn
from torch.nn import functional as F


class IntegrationDistributionModule(nn.Module):
    def __init__(self,
                 in_dim,
                 factor=16,
                 t=4):
        super(IntegrationDistributionModule, self).__init__()
        if in_dim == 256:
            in_thw = t * 64 * 32
        elif in_dim == 512:
            in_thw = t * 32 * 16
        else:
            in_thw = t * 16 * 8

        inter_dim = in_dim // factor
        inter_thw = in_thw // factor

        self.thw_reduction = nn.Sequential(
            nn.Conv2d(in_thw, inter_thw, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(inter_thw))

        self.thw_expansion = nn.Sequential(
            nn.Conv2d(inter_thw, in_thw, kernel_size=(1, 1), stride=(1, 1)))

        self.chl_reduction = nn.Sequential(
            nn.Conv3d(in_dim, inter_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(inter_dim))

        self.chl_expansion = nn.Sequential(
            nn.Conv3d(inter_dim, in_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)))

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.zero_init(self.chl_expansion)

    def zero_init(self, W):
        nn.init.constant_(W[-1].weight.data, 0.0)
        nn.init.constant_(W[-1].bias.data, 0.0)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        x1 = self.chl_reduction(x)
        b, c, t, h, w = x1.size()

        x1 = x1.reshape(b, c, -1, 1).transpose(1, 2)    # (b, t*h*w, c, 1)
        x2 = self.thw_reduction(x1)
        x3 = self.thw_expansion(x2)
        x3 = x3 + x1
        x3 = x3.transpose(1, 2).reshape(b, c, t, h, w)
        x4 = self.chl_expansion(x3)

        z = F.relu(x + x4)
        return z



