from collections import OrderedDict

import numpy as np
import torch.nn as nn

from utilities import CustomLogger

from models.neural_net import NeuralNet
from models.nets.mlpnet import MLPNet


class MLP(NeuralNet):
    """
    :type _params: MLP.Params
    :type _train_features: np.ndarray | None
    :type _train_labels: np.ndarray | None
    :type probabilities: np.ndarray | None
    :type _times: OrderedDict
    :type _net: nn.Module
    """

    class Params(NeuralNet.Params):
        """
        """

        def __init__(self):
            NeuralNet.Params.__init__(self)

            self.net = MLPNet.Params()

    def __init__(self, params, logger, input_size, name='', n_classes=2):
        """
        :type params: MLP.Params
        :type logger: CustomLogger
        :rtype: None
        """
        NeuralNet.__init__(self, params, logger, [input_size, ], name, 'MLP', n_classes)

        self._params = params

        self._net = MLPNet(input_size, self._params.net, self._logger, self._parent_name, self._n_classes)
        self._net.to(self._device)

