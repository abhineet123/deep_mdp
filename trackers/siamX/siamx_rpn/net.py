# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..siamx_models import builder


class SiamRPN(nn.Module):

    def __init__(self, tracker_name):
        super(SiamRPN, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def xcorr(self, z, x, channels):
        out = []
        kernel_size = z.data.size()[-1]
        for i in range(x.size(0)):
            out.append(F.conv2d(x[i, :, :, :].unsqueeze(0),
                                z[i, :, :, :].unsqueeze(0).view(channels, self.model.features.feature_channel, kernel_size, kernel_size)))
        return torch.cat(out, dim=0)

    def forward(self,  z, detection):
        detection_feature = self.model.features(detection)

        conv_score = self.model.conv_cls2(detection_feature)
        conv_regression = self.model.conv_r2(detection_feature)

        r1_kernel, cls1_kernel = z

        pred_score = self.xcorr(cls1_kernel.cuda(), conv_score, 10)
        pred_regression = self.model.regress_adjust(self.xcorr(r1_kernel.cuda(), conv_regression, 20))

        return pred_regression, pred_score

    def temple(self, z):
        z_f = self.model.features(z)
        r1_kernel = self.model.conv_r1(z_f).cpu()
        cls1_kernel = self.model.conv_cls1(z_f).cpu()

        return r1_kernel, cls1_kernel


class SiamRPNVGG(SiamRPN):
    def __init__(self, tracker_name):
        super(SiamRPNVGG, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355


class SiamRPNResNeXt22(SiamRPN):
    def __init__(self, tracker_name='SiamRPNResNeXt22'):
        super(SiamRPNResNeXt22, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355


class SiamRPNPP(nn.Module):
    def __init__(self, tracker_name):
        super(SiamRPNPP, self).__init__()
        self.tracker_name = tracker_name
        self.model = getattr(builder, tracker_name)()

    def temple(self, z):
        zf = self.model.features(z)
        zf = self.model.neck(zf)

        return zf

    def forward(self, zf, x):
        xf = self.model.features(x)
        xf = self.model.neck(xf)

        zf = [k.cuda() for k in zf]
        cls, loc = self.model.head(zf, xf)
        return loc, cls


class SiamRPNPPRes50(SiamRPNPP):
    def __init__(self, tracker_name='SiamRPNPP'):
        super(SiamRPNPPRes50, self).__init__(tracker_name)
        self.cfg = {'lr': 0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 255, 'adaptive': False} # 0.355

