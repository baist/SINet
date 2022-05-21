from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.nn import functional as F


class StatisticAttentionBlock(nn.Module):
    """
        SA block, statistic attention block
        include:
            1. down-channel:
                channel reduction for speed and channel shuffle、
            2. get_moments:
                calculate moments
            3. sta_distribute：
                distribute statistics based on the similarity
            4. up-channel:
                channel recover + residual connection
    """
    def __init__(
            self,
            in_dim,
            inter_dim=None,
            moments=None,
            moment_norm=True):
        super(StatisticAttentionBlock, self).__init__()
        self.in_dim = in_dim
        self.inter_dim = in_dim // 4 if inter_dim is None else inter_dim

        self.moments = [1, 2, 4, 5, 6] if moments is None else moments
        self.moment_norm = moment_norm

        self.down_channel = nn.Conv3d(self.in_dim, self.inter_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.up_channel = nn.Sequential(
            nn.Conv3d(self.inter_dim, self.in_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.in_dim))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.up_channel[-1].weight.data, 0.0)
        nn.init.constant_(self.up_channel[-1].bias.data, 0.0)

    def forward(self, x):
        z = self.down_channel(x)
        y = get_moments(z, moments=self.moments, moment_norm=self.moment_norm)
        z = sta_distribute(z, y)
        z = self.up_channel(z)
        z = z + x
        z = F.relu(z)
        return z

def get_moments(z, moments, moment_norm=True):
    """
    :param z:               (b, c, t, h, w)
    :param moments:         e.g. [1, 2, 3, 4]
    :param moment_norm:     True or False
    :return:
        (b, c, m), m=|moments|
    """
    b, c, t, h, w = z.size()

    mean = F.adaptive_avg_pool3d(z, output_size=1)  # (b, c, 1, 1, 1)
    mean = mean.reshape(b, c, 1)
    moments_set = [mean, ]

    z = z.reshape(b, c, t * h * w)  # (b, c, t*h*w)

    if 2 in moments:
        variance = torch.mean((z - torch.mean(z, dim=-1, keepdim=True)) ** 2, dim=-1, keepdim=True)
        if moment_norm: variance = torch.sqrt(variance)
        moments_set.append(variance)

        for i in moments:
            if i <= 2: continue

            c_moment = torch.mean((z - torch.mean(z, dim=-1, keepdim=True)) ** i, dim=-1, keepdim=True)
            if moment_norm:
                c_moment = c_moment / (variance ** i)
            moments_set.append(c_moment)

    y = torch.cat(moments_set, dim=2)   # (b, c, m)
    return y

def sta_distribute(x, mv, norm=True):
    """
    :param x:   feature map, size: (b, c, t, h, w)
    :param mv:  moment vectors, size: (b, c, m)
    :param norm:    True or False
    :return:
        (b, c, t, h, w)
    """
    b, c, t, h, w = x.size()

    if norm:
        mv = F.normalize(mv, dim=1)  # (b, c, m)

    x = x.reshape(b, c, t * h * w).permute(0, 2, 1)  # (b, t*h*w, c)
    mv = mv.reshape(b, c, -1)

    f = torch.matmul(x, mv)     # (b, t*h*w, m)
    f = F.softmax(f, dim=-1)    # (b, t*h*w, m)

    # (b, t*h*w, c) <- (b, t*h*w, m) * (b, m, c)
    x = torch.matmul(f, mv.permute(0, 2, 1))
    x = x.permute(0, 2, 1).reshape(b, c, t, h, w)

    return x


