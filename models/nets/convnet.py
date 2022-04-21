import functools
import ast
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

from utilities import CustomLogger


class ConvNet(nn.Module):
    """
        :type _input_shape: list
        :type _params: ConvNet.Params
        :type _logger: logging.RootLogger
    """

    class Params:
        """
        :type act_type: int
        :type out_act_type: int
        """

        def __init__(self):
            self.n_hidden_layers = -1

            self.cfg_files = 'cfg/convnet.cfg'
            self.cfg_id = '0'
            self.act_type = 0
            self.dropout = [0, ]
            self.batch_norm = [0, ]
            self.out_act_type = -2

            self.configs = []

            self.help = {
                'n_hidden_layers': '>1 means that it can be used to fill in hidden_sizes, '
                                   'activation_types, dropout and batch_norm if each of these have unit length '
                                   'otherwise the lengths must match n_hidden_layers; '
                                   '<0 means that it is determined from hidden_sizes',
                'hidden_sizes': 'number of units n in each layer;',
                'dropout': '0 < dropout < 1; zeroing probability for each layer; 0 means no dropout',

            }

        def _process(self, var, name, type):
            if isinstance(var, type):
                var = [var, ] * self.n_hidden_layers
            elif isinstance(var, (tuple, list)):
                if len(var) == 1:
                    var = [var[0], ] * self.n_hidden_layers
                else:
                    assert len(var) == self.n_hidden_layers, \
                        f"Mismatch between no. of hidden_layers: {self.n_hidden_layers} and " \
                            f"{name}: {len(var)}"
                return var

        def read_cfg(self):

            cfg_files = self.cfg_files.split(',')
            self.configs = []
            for cfg_file in cfg_files:
                lines = open(cfg_file).readlines()
                start_idxs = [i for i, line in enumerate(lines) if line.startswith('[')]
                end_idxs = [i for i, line in enumerate(lines) if line.startswith(']')]
                n_cfg = len(start_idxs)

                assert n_cfg == len(end_idxs), \
                    f"Mismatch between no. of start_idxs {n_cfg} and end_idxs {len(end_idxs)}"

                for _cfg_id in range(n_cfg):
                    name_id = lines[start_idxs[_cfg_id]].find('##')
                    if name_id > 0:
                        _cfg_name = lines[start_idxs[_cfg_id]][name_id + 2:].strip()
                    else:
                        _cfg_name = f'cnv{_cfg_id}'

                    _cfg = lines[start_idxs[_cfg_id]:end_idxs[_cfg_id] + 1]
                    # _cfg = [k.strip() for k in _cfg]
                    _cfg_str = ''.join(_cfg)
                    _cfg_list = ast.literal_eval(_cfg_str)

                    self.configs.append((_cfg_name, _cfg_list))

        def process(self):
            self.read_cfg()

    def __init__(self, input_shape, n_classes, params, logger, name):
        """

        :param tuple(int) input_shape:
        :param ConvNet.Params params:
        :param logging.RootLogger logger:
        """
        assert len(input_shape) == 3, "only 3D inputs are supported"

        super(ConvNet, self).__init__()
        self._act_fn_list = [
            ('relu', nn.ReLU),
            ('lrelu', nn.LeakyReLU),
            ('prelu', nn.PReLU),
            ('rrelu', nn.RReLU),
            ('selu', nn.SELU),
            ('elu', nn.ELU),
            ('celu', nn.CELU),
            ('glu', nn.GLU),
            ('htanh', nn.Hardtanh),
            ('tanh', nn.Tanh),
            ('sigmoid', nn.Sigmoid),
            ('log_sigmoid', nn.LogSigmoid),
            ('hardshrink', nn.Hardshrink),
            ('softshrink', nn.Softshrink),
            ('softplus', nn.Softplus),
            ('softmax', functools.partial(nn.Softmax, dim=1)),
            ('log_softmax', nn.LogSoftmax)]

        self._act_types = [k[0] for k in self._act_fn_list]

        self._params = params
        self._name = name
        self._input_shape = input_shape
        self._n_classes = n_classes
        self._logger = CustomLogger(logger, names=('convnet',))

        self._params.process()

        self._net_config = []

        self._params.process()

        self._net_names = [k[0] for k in self._params.configs]

        try:
            cfg_id = int(self._params.cfg_id)
        except ValueError:
            try:
                cfg_id = self._net_names.index(self._params.cfg_id)
            except IndexError:
                raise IOError(f'Invalid cfg ID: {self._params.cfg_id}')

        self._net_name, self._net_config = list(self._params.configs[cfg_id]).copy()

        _act_fn = self._act_fn_list[self._params.act_type]

        self._layers = []

        _prev_size = self._input_shape[-2:]
        in_channels = self._input_shape[0]
        dummy_tensor = torch.zeros((10, *self._input_shape))
        n_layers = len(self._net_config)
        for i, _layer_cfg in enumerate(self._net_config):
            flatten = 0
            act_fn = 1
            _layer_cfg_meta = ''

            if '::' in _layer_cfg:
                _layer_cfg, _layer_cfg_meta = _layer_cfg.split('::')
                # n_ext_params = len(_layer_cfg_meta)

            _layer_cfg_params = _layer_cfg.split(':')
            n_params = len(_layer_cfg_params)
            if _layer_cfg_params[0] == 'c':
                assert n_params >= 3, \
                    f"Insufficient parameters specified for conv layer: {_layer_cfg_params}"
                conv_params = {
                    'in_channels': in_channels,
                    'out_channels': int(_layer_cfg_params[1]),
                    'kernel_size': int(_layer_cfg_params[2])
                }
                if n_params > 3:
                    conv_params['stride'] = int(_layer_cfg_params[3])
                if n_params > 4:
                    conv_params['padding'] = int(_layer_cfg_params[4])
                if n_params > 5:
                    conv_params['dilation'] = int(_layer_cfg_params[5])

                layer_name = 'conv_{}'.format(i)
                layer = nn.Conv2d(**conv_params)

                in_channels = conv_params['out_channels']

            elif _layer_cfg_params[0] == 'm':
                assert n_params >= 2, \
                    f"Insufficient parameters specified for max_pool layer: {_layer_cfg_params}"

                max_pool_params = {
                    'kernel_size': int(_layer_cfg_params[1]),
                }
                if n_params > 3:
                    max_pool_params['stride'] = int(_layer_cfg_params[3])
                if n_params > 4:
                    max_pool_params['padding'] = int(_layer_cfg_params[4])
                if n_params > 5:
                    max_pool_params['dilation'] = int(_layer_cfg_params[5])

                layer_name = 'max_pool_{}'.format(i)
                layer = nn.MaxPool2d(**max_pool_params)
                act_fn = 0

            elif _layer_cfg_params[0] == 'f':
                assert n_params >= 2, \
                    f"Insufficient parameters specified for fc layer: {_layer_cfg_params}"

                in_features = tuple(dummy_tensor.size())
                in_features = np.prod(in_features[1:])
                fc_params = {
                    'in_features': in_features,
                    'out_features': int(_layer_cfg_params[1]),
                }
                # in_features = fc_params['out_features']
                layer_name = 'fc_{}'.format(i)
                layer = nn.Linear(**fc_params)

                # if len(dummy_tensor.shape) > 2:
                #     layer.insert(0, nn.Flatten(start_dim=1))
                flatten = 1
            else:
                raise AssertionError(f'Invalid layer type: {_layer_cfg_params[0]}')

            in_shape = tuple(dummy_tensor.shape)[1:]
            if flatten and len(dummy_tensor.shape) > 2:
                flatten = nn.Flatten(start_dim=1)
                self._layers.append((layer_name + '_flatten', flatten))
                dummy_tensor = flatten(dummy_tensor)
            dummy_tensor = layer(dummy_tensor)
            self._layers.append((layer_name, layer))
            out_shape = tuple(dummy_tensor.shape)[1:]

            print(f'{i}: {layer_name} :: {in_shape} --> {out_shape}')

            # layer meta structure: batch norm, dropout, activation
            bn_layer = dropout_layer = None
            if _layer_cfg_meta:
                _layer_cfg_ext_params = _layer_cfg_meta.split(':')
                _id = 0
                while _id < len(_layer_cfg_ext_params):
                    if _layer_cfg_ext_params[_id] == 'b':
                        if _layer_cfg_params[0] == 'f':
                            bn_layer = ('bn_{}'.format(i), nn.BatchNorm1d(dummy_tensor.shape[1]))
                        else:
                            bn_layer = ('bn_{}'.format(i), nn.BatchNorm2d(dummy_tensor.shape[1]))
                        _id += 1
                    elif _layer_cfg_ext_params[_id] == 'd':
                        _dropout = float(_layer_cfg_ext_params[_id + 1])
                        dropout_layer = ('dropout_{}'.format(i), nn.Dropout(p=_dropout))
                        _id += 2

            if bn_layer:
                self._layers.append(bn_layer)

            if act_fn:
                self._layers.append(('{}_{}'.format(_act_fn[0], i), _act_fn[1]()))

            if dropout_layer:
                self._layers.append(dropout_layer)

        # self._layers.append(('fc_{}'.format(self._params.n_layers - 1),
        #                      nn.Linear(self._params.hidden_sizes[-1], 2)))
        # self._layers.append(
        #     ('{}_{}'.format(self._params.activation_type, self._params.n_layers - 1), self.activation_fn()))

        n_features = np.prod(tuple(dummy_tensor.shape)[1:])
        if len(dummy_tensor.shape) > 2 or n_features != n_classes:

            in_shape = tuple(dummy_tensor.shape)[1:]
            layer_name = ''

            if len(dummy_tensor.shape) > 2:
                layer_name = f'flatten_{n_layers}'
                flatten = nn.Flatten(start_dim=1)
                self._layers.append((layer_name, flatten))
                dummy_tensor = flatten(dummy_tensor)

            if n_features != n_classes:
                layer_name = f'fc_{n_layers}'
                layer = nn.Linear(n_features, n_classes)
                dummy_tensor = layer(dummy_tensor)
                self._layers.append((layer_name, layer))

            out_shape = tuple(dummy_tensor.shape)[1:]
            print(f'{n_layers}: {layer_name} :: {in_shape} --> {out_shape}')

            out_act_fn = self._act_fn_list[self._params.out_act_type]
            self._layers.append(('{}_{}'.format(out_act_fn[0], n_layers), out_act_fn[1]()))

        self.net = nn.Sequential(OrderedDict(self._layers))

        self._logger.debug(f'Using config {cfg_id}: {self._net_name} with'
                           f' structure:\n{self.net}\n')
        self.n_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._logger.debug(f'n_train_params: {self.n_train_params}')

    def forward(self, x):
        out = self.net(x)
        return out
