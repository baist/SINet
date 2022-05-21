from __future__ import absolute_import


import math
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Salient2BroadModule(nn.Module):
    def __init__(self,
                 in_dim,
                 inter_dim=None,
                 split_pos=0,
                 k=3,
                 exp_beta=5.0,
                 cpm_alpha=0.1):
        super().__init__()
        self.in_channels = in_dim
        self.inter_channels = inter_dim or in_dim // 4
        self.pos = split_pos
        self.k = k
        self.exp_beta = exp_beta
        self.cpm_alpha = cpm_alpha

        self.kernel = nn.Sequential(
            nn.Conv3d(self.in_channels, self.k * self.k, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(self.k * self.k),
            nn.ReLU())

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Conv3d(self.in_channels, self.inter_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.inter_channels, self.in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _suppress(self, a, exp_beta=5.0):
        """
        :param a: (b, 1, t, h, w)
        :return:
        """
        a_sup = (a < 1).float().detach()
        a_exp = torch.exp((a-1)*a_sup*exp_beta)
        a = a_exp * a_sup + (1 - a_sup)
        return a

    def _channel_center(self, x):
        """
        :param x:  (b, c, t, h, w)
        :return:   (b, c, 1, 1, 1)
        """
        center_w_pad = torch.mean(x, dim=(2,3,4), keepdim=True)
        center_wo_pad = torch.mean(x[:,:,:,1:-1, 1:-1], dim=(2,3,4), keepdim=True)
        center = center_wo_pad/(center_w_pad + 1e-8)
        return center

    def channel_attention_layer(self, x):
        se = self.se(x)                     # (b, c, 1, 1, 1)
        center = self._channel_center(x)     # (b, c, 1, 1, 1)
        center = (center > 1).float().detach()
        return se * center

    def _forward(self, x, pos=None):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        pos = self.pos if pos is None else pos

        b, c, t, h, w = x.shape
        xf = x[:, :, :pos + 1]
        xl = x[:, :, pos + 1:]

        cal = self.channel_attention_layer(x)
        xf_se = F.relu(xf * cal)

        # k*k spatial attention
        spatial_att = self.kernel(xf_se)    # (b, k*k, tf, h, w)
        # (b, tf*hw, k*k)
        spatial_att = spatial_att.reshape(b, self.k*self.k, -1).transpose(-2, -1)
        if self.k != 1:
            spatial_att = F.normalize(spatial_att, dim=-1, p=1)
        spatial_att = F.normalize(spatial_att, dim=1, p=1)

        # obtain k*k conv kernel
        xf_reshape = xf_se.reshape(b, c, -1)
        # (b, c, 1, k, k)
        kernel = torch.matmul(xf_reshape, spatial_att)
        kernel = kernel.reshape(b, c, 1, self.k, self.k)

        # perform convolution with calculated kernel
        xl_se = F.relu(xl * cal)    # (1, b*c, tl, h, w)
        xl_reshape = xl_se.reshape(b*c, -1, h, w)

        pad = (self.k-1)//2
        xl_reshape = F.pad(xl_reshape, pad=[pad,pad,pad,pad], mode='replicate')
        xl_reshape = xl_reshape.unsqueeze(0)
        f = F.conv3d(xl_reshape, weight=kernel, bias=None, stride=1, groups=b)
        f = f / (self.k * self.k)

        # suppress operation
        f = f.reshape(b, -1, h*w)
        f = F.softmax(f, dim=-1)
        f = f.reshape(b, 1, -1, h, w).clamp_min(1e-4)

        f = 1.0 / (f * h * w)
        f = self._suppress(f, exp_beta=self.exp_beta)

        # cross propagation
        xl_res = xl * f + self.cpm_alpha * F.adaptive_avg_pool3d(xf, 1)
        xf_res = xf + self.cpm_alpha * F.adaptive_avg_pool3d((1-f)* xl, 1)/F.adaptive_avg_pool3d((1-f), 1)
        res = torch.cat([xf_res, xl_res], dim=2)

        return res

    def forward(self, x, pos=None):
        b, c, t, h, w = x.shape
        if t == 4:
            return self._forward(x, pos)
        else:
            assert t % 4 == 0
            x = x.reshape(b, c, 2, 4, h, w)
            x = x.transpose(1, 2).reshape(b*2, c, 4, h, w)
            x = self._forward(x, pos)
            x = x.reshape(b, 2, c, 4, h, w).transpose(1, 2)
            x = x.reshape(b, c, t, h, w)

        return x





