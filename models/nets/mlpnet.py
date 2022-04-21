import functools
from collections import OrderedDict
import torch
import torch.nn as nn

from utilities import CustomLogger


class MLPNet(nn.Module):
    """
        :type _input_size: int
        :type _params: MLPNet.Params
        :type _logger: CustomLogger
    """

    class Params:
        """
        :type hidden_sizes: list
        :type activation_types: list


        :ivar n_hidden_layers: '>1 means that it can be used to fill in hidden_sizes, '
                   'activation_types, dropout and batch_norm if each of these have unit length '
                   'otherwise the lengths must match n_hidden_layers; '
                   '<0 means that it is determined from hidden_sizes',
        :ivar hidden_sizes: 'number of units n in each layer;',
        :ivar dropout: '0 < dropout < 1; zeroing probability for each layer; 0 means no dropout',


        """

        def __init__(self):
            self.n_hidden_layers = -1
            self.hidden_sizes = [24, 48, 24]
            self.activation_types = ['relu', ]
            self.dropout = [0, ]
            self.batch_norm = [0, ]
            self.out_activation_type = 'softmax'

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

        def process(self):

            if self.n_hidden_layers < 0:
                self.n_hidden_layers = len(self.hidden_sizes)

            self.activation_types = self._process(self.activation_types, 'activation_types', str)
            self.hidden_sizes = self._process(self.hidden_sizes, 'hidden_sizes', int)
            self.dropout = self._process(self.dropout, 'dropout', float)
            self.batch_norm = self._process(self.batch_norm, 'batch_norm', int)

    def __init__(self, input_size, params, logger, name, n_classes):
        """

        :param int input_size:
        :param MLPNet.Params params:
        :param CustomLogger logger:
        """
        super(MLPNet, self).__init__()
        self.__activation_fn_dict = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'rrelu': nn.RReLU,
            'selu': nn.SELU,
            'elu': nn.ELU,
            'celu': nn.CELU,
            'glu': nn.GLU,
            'htanh': nn.Hardtanh,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'log_sigmoid': nn.LogSigmoid,
            'hardshrink': nn.Hardshrink,
            'softshrink': nn.Softshrink,
            'softplus': nn.Softplus,
            'softmax': functools.partial(nn.Softmax, dim=1),
            'log_softmax': nn.LogSoftmax,
        }

        self._params = params
        self._logger = logger
        self._name = name
        self._input_size = input_size
        self._n_classes = n_classes

        self._layer_sizes = []

        custom_log_header = {'custom_module': '{}:MLPNet'.format(self._name)}
        try:
            custom_log_header.update(self._logger.info.keywords['extra'])
        except:
            pass
        self._log_info = functools.partial(self._logger.info, extra=custom_log_header)

        self._params.process()

        self._activation_types = list(self._params.activation_types).copy()
        self._layer_sizes = list(self._params.hidden_sizes).copy()
        self._batch_norm = list(self._params.batch_norm).copy()
        self._dropout = list(self._params.dropout).copy()

        # Output layer
        self._activation_types.append(self._params.out_activation_type)
        self._layer_sizes.append(self._n_classes)
        self._batch_norm.append(0)
        self._dropout.append(0)

        try:
            self._activation_fn = [self.__activation_fn_dict[_activation_type]
                                   for _activation_type in self._activation_types]
        except KeyError:
            raise AssertionError('Invalid activation_type: {}'.format(self.activation_types))
        self._layers = []

        _prev_size = input_size
        for i, _layer_size in enumerate(self._layer_sizes):
            self._layers.append(
                ('fc_{}'.format(i), nn.Linear(_prev_size, _layer_size)))
            if self._batch_norm[i]:
                """Insert before the activation function
                """
                self._layers.append(
                    ('bn_{}'.format(i), nn.BatchNorm1d(_layer_size)))
            self._layers.append(('{}_{}'.format(self._activation_types[i], i), self._activation_fn[i]()))
            if self._dropout[i] > 0:
                self._layers.append(
                    ('dropout_{}'.format(i), nn.Dropout(p=self._dropout[i])))

            _prev_size = _layer_size

        # self._layers.append(('fc_{}'.format(self._params.n_layers - 1),
        #                      nn.Linear(self._params.hidden_sizes[-1], 2)))
        # self._layers.append(
        #     ('{}_{}'.format(self._params.activation_type, self._params.n_layers - 1), self.activation_fn()))

        self.net = nn.Sequential(OrderedDict(self._layers))

        self._log_info(f'Created net with'
                       # f' params:\n{pformat(self._params.__dict__)}\n'
                       f' structure:\n{self.net}\n')
        self.n_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._log_info(f'n_train_params: {self.n_train_params}')

    def forward(self, x):
        out = self.net(x)
        return out
