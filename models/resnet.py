from __future__ import absolute_import


from functools import partial
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from models import inflate
from models.statistic_attention_block import StatisticAttentionBlock
from models.salient_to_broad_module import Salient2BroadModule
from models.integration_distribution_module import IntegrationDistributionModule


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d, inflate_time=False):
        super(Bottleneck3d, self).__init__()

        if inflate_time == True:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=3, time_padding=1, center=True)
        else:
            self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class _ResNet50(nn.Module):

    def __init__(self, num_classes, losses, plugin_dict):
        super(_ResNet50, self).__init__()
        self.losses = losses

        resnet2d = torchvision.models.resnet50(pretrained=True)
        resnet2d.layer4[0].conv2.stride = (1, 1)
        resnet2d.layer4[0].downsample[0].stride = (1, 1)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.layer1 = self._inflate_reslayer(resnet2d.layer1, plugin_dict[1])
        self.layer2 = self._inflate_reslayer(resnet2d.layer2, plugin_dict[2])
        self.layer3 = self._inflate_reslayer(resnet2d.layer3, plugin_dict[3])
        self.layer4 = self._inflate_reslayer(resnet2d.layer4, plugin_dict[4])

        self.bn = nn.BatchNorm1d(2048)
        self.bn.apply(weights_init_kaiming)

        self.classifier = nn.Linear(2048, num_classes)
        self.classifier.apply(weights_init_classifier)

        if 'infonce' in self.losses:
            self.projection = nn.Conv1d(2048, 2048, (1,), (1,))
            self.projection.apply(weights_init_kaiming)

    def _inflate_reslayer(self, reslayer2d, plugin_dict):
        reslayers3d = []
        for i, layer2d in enumerate(reslayer2d):
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

            if i in plugin_dict:
                reslayers3d.append(plugin_dict[i](in_dim=layer2d.bn3.num_features))

        return nn.Sequential(*reslayers3d)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        x = x.transpose(1, 2)  # (b, c, t)

        if not self.training:
            x = self.bn(x)
            return x

        v = x.mean(-1)
        f = self.bn(v)
        y = self.classifier(f)

        if 'infonce' in self.losses:
            x = self.bn(x)
            x = self.projection(x)
            return y, f, x

        return y, f


def ResNet50(num_classes, losses):
    plugin_dict = {
        1: {},
        2: {},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SANet(num_classes, losses, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: StatisticAttentionBlock,
            3: StatisticAttentionBlock},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def IDNet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: partial(IntegrationDistributionModule, t=seq_len),
            3: partial(IntegrationDistributionModule, t=seq_len)},
        3: {},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SBNet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {},
        3: {1: partial(Salient2BroadModule, split_pos=0),
            3: partial(Salient2BroadModule, split_pos=1),
            5: partial(Salient2BroadModule, split_pos=2)},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)


def SINet(num_classes, losses, seq_len, **kwargs):
    plugin_dict = {
        1: {},
        2: {1: partial(IntegrationDistributionModule, t=seq_len),
            3: partial(IntegrationDistributionModule, t=seq_len)},
        3: {1: partial(Salient2BroadModule, split_pos=0),
            3: partial(Salient2BroadModule, split_pos=1),
            5: partial(Salient2BroadModule, split_pos=2)},
        4: {}
    }
    return _ResNet50(num_classes, losses, plugin_dict)
