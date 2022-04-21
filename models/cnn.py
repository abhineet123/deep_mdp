import functools

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

import paramparse

from models.neural_net import NeuralNet
from models.nets.convnet import ConvNet
from models.nets.resnet import resnet18, resnet50, resnet101, resnet152, \
    resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2
from models.nets.inception import inception_v3
from models.nets.mobilenet import mobilenet_v2
from models.nets.alexnet import AlexNet
from models.nets.vgg import vgg11, vgg13, vgg16

from utilities import annotate_and_show, CustomLogger


class CNN(NeuralNet):
    """
    :type _params: CNN.Params
    :type _net: nn.Module
    """

    class Params(NeuralNet.Params):
        """
        :ivar net: {
            'ConvNet': ('-1', 'cn', 'convnet'),
            'AlexNet': ('0', 'alx', 'alexnet'),
            'VGG11': ('1', 'v11', 'vgg11'),
            'VGG13': ('2', 'v13', 'vgg13'),
            'VGG16': ('3', 'v16', 'vgg16'),
            'Mobilenet_v2': ('4', 'mb2', 'mbn2', 'mobilenetv2'),
            'Inception_v3': ('5', 'incp3', 'ic3', 'inceptionv3'),
            'ResNet18': ('6', 'r18', 'resnet18'),
            'ResNet50': ('7', 'r50', 'resnet50'),
            'ResNet101': ('8', 'r101', 'resnet101'),
            'ResNet152': ('9', 'r152', 'resnet152'),
            'ResNext50': ('10', 'rx50', 'resnext50'),
            'ResNext101': ('11', 'rx101', 'resnext101'),
            'WideResNet50': ('12', 'wr50', 'wresnet50'),
            'WideResNet101': ('13', 'wr101', 'wresnet101'),
        }
        """

        def __init__(self):
            NeuralNet.Params.__init__(self)

            self.net = '0'
            self.siamese = 0
            self.convnet = ConvNet.Params()

    def __init__(self, params, logger, feature_shape, name='', n_classes=2):
        """
        :type params: CNN.Params
        :type logger: CustomLogger
        :rtype: None
        """

        assert 3 >= len(feature_shape) >= 2, "CNN can only be used with 2D or 3D inputs"
        if len(feature_shape) == 2:
            feature_shape = [1, ] + list(feature_shape)

        NeuralNet.__init__(self, params, logger, feature_shape, name, 'CNN', n_classes)

        """cannot move to Model for intellisense to work"""
        self._params = params

        # self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                        std=[0.229, 0.224, 0.225])

        common_params = {
            'logger': self._logger,
            'pretrained': self._params.pretrained,
        }

        self._nets = {
            'ConvNet': functools.partial(ConvNet, logger=self._logger, params=self._params.convnet, name=self.name),
            'AlexNet': AlexNet,
            'VGG11': vgg11,
            'VGG13': vgg13,
            'VGG16': vgg16,
            'ResNet18': functools.partial(resnet18, siamese=self._params.siamese, **common_params),
            'ResNet50': functools.partial(resnet50, **common_params),
            'ResNet101': functools.partial(resnet101, **common_params),
            'ResNet152': functools.partial(resnet152, **common_params),
            'ResNext50': functools.partial(resnext50_32x4d, **common_params),
            'ResNext101': functools.partial(resnext101_32x8d, **common_params),
            'WideResNet50': functools.partial(wide_resnet50_2, **common_params),
            'WideResNet101': functools.partial(wide_resnet101_2, **common_params),
            'Inception_v3': functools.partial(inception_v3, **common_params),
            'Mobilenet_v2': functools.partial(mobilenet_v2, **common_params)
        }
        _net_type = paramparse.match_opt(self._params, 'net')
        _net_func = self._nets[_net_type]
        self._logger.info(f'Using {_net_type} with input_shape {feature_shape}')

        if self._params.siamese:
            self._logger.info(f'Using siamese mode')

        self._net = _net_func(self.feature_shape, self._n_classes)
        self._net.to(self._device)

        self.__prev_saved_sanmples = []

        self._pause = 1

    @staticmethod
    def vis_samples(features, labels, gt_labels, _input_shape, batch_size):
        start_id = 0
        batch_id = 0
        _pause = 1
        # nms_dist = 8
        # n_conf = 5

        _h, _w = _input_shape[-2:]
        # cy, cx = int(_h / 2), int(_w / 2)

        _n_samples = features.shape[0]

        # max_offset_x = float(max(cx, _w - cx - 1))
        # max_offset_y = float(max(cy, _h - cy - 1))

        # max_dist_sqr = max_offset_x ** 2 + max_offset_y ** 2

        while True:
            concat_imgs = []
            # concat_features = []
            for i in range(batch_size):
                if start_id + i >= _n_samples:
                    break
                feature_map = features[start_id + i, ...].reshape(_input_shape)

                feature_map = [feature_map[i, ...].squeeze() for i in range(_input_shape[0])]
                feature_map = np.concatenate(feature_map, axis=1)
                feature_map = cv2.resize(feature_map, (300 * _input_shape[0], 300), interpolation=cv2.INTER_AREA)
                feature_map = np.stack([feature_map, ] * 3, axis=2)

                # annotate_and_show('feature_map', feature_map, self._logger, pause=self._pause)

                # if len(self._input_shape) == 2:
                #     feature_map = cv2.cvtColor(feature_map, cv2.COLOR_GRAY2BGR)
                concat_img = [feature_map, ]

                for _labels in (labels, gt_labels):

                    if _labels is None:
                        continue

                    label = _labels[start_id + i]

                    if label < 0:
                        label = 0

                    if label == 1:
                        label_col = (0, 1, 0)
                    else:
                        label_col = (0, 0, 1)
                    label_img = np.full((feature_map.shape[0], 25, 3), label_col[0])
                    label_img[..., 1].fill(label_col[1])
                    label_img[..., 2].fill(label_col[2])

                    # label_img[0, ...] = (1, 1, 1)
                    # label_img[-1, ...] = (1, 1, 1)
                    # label_img[:, 0, ...] = (1, 1, 1)
                    # label_img[:, -1, ...] = (1, 1, 1)

                    """5 pixel vertical border"""
                    border_img = np.zeros((label_img.shape[0], 5, 3), dtype=label_img.dtype)

                    concat_img += [border_img, label_img]

                if len(concat_img) > 1:
                    concat_img = np.concatenate(concat_img, axis=1)
                else:
                    concat_img = concat_img[0]

                concat_imgs.append(concat_img)
                # concat_features.append(features)

            concat_imgs = np.concatenate(concat_imgs, axis=0)
            # concat_features = np.asarray(concat_features)

            start_id += batch_size
            batch_id += 1

            yield batch_id, concat_imgs

            if start_id >= _n_samples:
                break

    def _vis_samples(self, features=None, labels=None, gt_labels=None, batch_size=None, return_only=0):

        if features is None:
            features = self._all_features
        if labels is None:
            labels = self._all_labels
        if batch_size is None:
            batch_size = self._params.batch_size

        vis_gen = self.vis_samples(features, labels, gt_labels, self.feature_shape, batch_size)
        for batch_id, concat_imgs in vis_gen:
            if return_only:
                return concat_imgs

            self._logger.info(f'target {self._target_id} batch {batch_id}')
            self._pause = annotate_and_show('features and labels', concat_imgs, self._logger, pause=self._pause)





