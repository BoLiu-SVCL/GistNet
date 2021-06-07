import math
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, *args):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x


class GIST(nn.Module):

    def __init__(self, n_classes=1000, n_struct=0):
        super(GIST, self).__init__()
        self.n_feat = 512
        self.n_classes = n_classes
        self.n_struct = n_struct

        self.feat_model = ResNet(BasicBlock, [1, 1, 1, 1])
        self.weights_c = nn.Parameter(torch.randn(self.n_classes, self.n_feat), requires_grad=True)
        self.weights_r = nn.Parameter(torch.randn(self.n_classes, self.n_feat), requires_grad=True)
        self.tau_c = nn.Parameter(torch.randn(1), requires_grad=True)
        self.tau_r = nn.Parameter(torch.randn(1), requires_grad=True)
        if self.n_struct > 0:
            self.struct = nn.Parameter(torch.zeros(self.n_struct, self.n_feat), requires_grad=True)
            self.pooling = nn.MaxPool1d(kernel_size=self.n_struct+1, stride=self.n_struct+1)
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, x, train=True):
        feat_maps = self.feat_model(x)
        N = feat_maps.size(0)

        # Train
        if train:
            if self.n_struct > 0:
                weights_aug_c = self.weights_c.unsqueeze(1) + self.struct.detach().unsqueeze(0)
                weights_all_c = torch.cat((self.weights_c.unsqueeze(1), weights_aug_c), dim=1).view(-1, 512)
                M_c = weights_all_c.size(0)
                weights_aug_r = self.weights_r.unsqueeze(1) + self.struct.unsqueeze(0)
                weights_all_r = torch.cat((self.weights_r.unsqueeze(1), weights_aug_r), dim=1).view(-1, 512)
                M_r = weights_all_r.size(0)

                feat_maps_c = feat_maps.unsqueeze(1).expand(-1, M_c, -1)
                feat_maps_r = feat_maps.unsqueeze(1).expand(-1, M_r, -1)
                weights_all_c = weights_all_c.unsqueeze(0).expand(N, -1, -1)
                weights_all_r = weights_all_r.unsqueeze(0).expand(N, -1, -1)
                similarity_c = self.tau_c * self.cos(feat_maps_c, weights_all_c)
                similarity_c = self.pooling(similarity_c.unsqueeze(1)).squeeze()
                similarity_r = self.tau_r * self.cos(feat_maps_r, weights_all_r)
                similarity_r = self.pooling(similarity_r.unsqueeze(1)).squeeze()
                return similarity_c, similarity_r

            else:
                M_c = self.weights_c.size(0)
                M_r = self.weights_r.size(0)

                feat_maps_c = feat_maps.unsqueeze(1).expand(-1, M_c, -1)
                feat_maps_r = feat_maps.unsqueeze(1).expand(-1, M_r, -1)
                weights_all_c = self.weights_c.unsqueeze(0).expand(N, -1, -1)
                weights_all_r = self.weights_r.unsqueeze(0).expand(N, -1, -1)
                similarity_c = self.tau_c * self.cos(feat_maps_c, weights_all_c)
                similarity_r = self.tau_r * self.cos(feat_maps_r, weights_all_r)
                return similarity_c, similarity_r

        # Test
        else:
            if self.n_struct > 0:
                weights_aug_c = self.weights_c.unsqueeze(1) + self.struct.detach().unsqueeze(0)
                weights_all_c = torch.cat((self.weights_c.unsqueeze(1), weights_aug_c), dim=1).view(-1, 512)
                M_c = weights_all_c.size(0)

                feat_maps_c = feat_maps.unsqueeze(1).expand(-1, M_c, -1)
                weights_all_c = weights_all_c.unsqueeze(0).expand(N, -1, -1)
                similarity_c = self.tau_c * self.cos(feat_maps_c, weights_all_c)
                output_c = self.pooling(similarity_c.unsqueeze(1)).squeeze()
                return output_c

            else:
                M_c = self.weights_c.size(0)

                feat_maps_c = feat_maps.unsqueeze(1).expand(-1, M_c, -1)
                weights_all_c = self.weights_c.unsqueeze(0).expand(N, -1, -1)
                similarity_c = self.tau_c * self.cos(feat_maps_c, weights_all_c)
                return similarity_c
