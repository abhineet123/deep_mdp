import torchvision
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data.sampler import Sampler

import math
import functools

import paramparse


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.labels[idx].item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class NLLLossOHEM(nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """

    def __init__(self, ratio, device):
        super(NLLLossOHEM, self).__init__()
        self.ratio = ratio
        self.device = device

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        inst_losses = torch.autograd.Variable(torch.zeros(num_inst)).to(self.device)
        for idx, label in enumerate(y.data):
            inst_losses[idx] = -x_.data[idx, label]
            # loss_incs = -x_.sum(1)
        _, idxs = inst_losses.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return nn.functional.nll_loss(x_hn, y_hn)


class CrossEntropyLossOHEM(nn.NLLLoss):
    """ Online hard example mining with CrossEntropyLoss"""

    def __init__(self, ratio, device):
        super(CrossEntropyLossOHEM, self).__init__()
        self.ratio = ratio
        self.device = device

    def forward(self, x, y, ratio=None):

        try:
            num_inst = x.size(0)
        except AttributeError:
            x = x[0]
            num_inst = x.size(0)

        losses = nn.functional.cross_entropy(x, y, reduction='none')
        if not self.training:
            ohem_loss = torch.sum(losses) / num_inst
        else:
            if ratio is not None:
                self.ratio = ratio

            num_hns = max(1, int(self.ratio * num_inst))

            _, idxs = losses.topk(num_hns)

            top_losses = losses.index_select(0, idxs)

            ohem_loss = torch.sum(top_losses) / num_hns

        if torch.isnan(ohem_loss).any():
            print('nan loss found')

        return ohem_loss


class PyTorchOptParams:
    """

    :ivar type: {
        'ADAM': (0, 'a', 'adam'),
        'AdamW': (1, 'aw', 'adamw'),
        'Adamax': (2, 'ax', 'adamax'),
        'SGD': (3, 'sg', 'sgd'),
        'ASGD': (4, 'as', 'asgd'),
        'Adagrad': (5, 'ag', 'adagrad'),
        'LBFGS': (6, 'lb', 'lbfgs'),
        'RMSprop': (7, 'rm', 'rms', 'rmsprop'),
        'Rprop': (8, 'rp', 'rprop'),
        'Adadelta': (9, 'add', 'adadelta'),
    }

    :ivar scheduler: {
        'None': (-1, 'none', 'fixed'),
        'StepLR': (0, 's', 'step'),
        'ReduceLROnPlateau': (1, 'p', 'plat'),
        'CosineAnnealingLR': (2, 'c', 'cos'),
        'ExponentialLR': (3, 'e', 'exp'),
        'MultiStepLR': (4, 'ms', 'mstep'),
        'MultiplicativeLR': (5, 'mu', 'mult'),
        'LambdaLR': (6, 'l', 'lamda'),
        'CyclicLR': (7, 'cy', 'cyclic'),
        'OneCycleLR': (8, '1cy', '1cyclic'),
        'CosineAnnealingWarmRestarts': (9, 'cr', 'cosr', 'cos_restart'),
    }

    """

    def __init__(self):
        self.type = 'adam'
        self.lr = 1e-3
        self.scheduler = 'none'

        # optimizer params
        self.sgd = self.SGD()
        self.adam = self.Adam()
        self.rms_prop = self.RMSprop()
        self.rprop = self.Rprop()
        self.asgd = self.ASGD()
        self.adadelta = self.Adadelta()
        self.adagrad = self.Adagrad()
        self.adamw = self.AdamW()
        self.adamax = self.Adamax()

        # scheduler params
        self.step = self.StepLR()
        self.plateau = self.ReduceLROnPlateau()
        self.cos = self.CosineAnnealingLR()
        self.exp = self.ExponentialLR()

    def get_optimizer(self):
        _opt_type = paramparse.match_opt(self, 'type', 'optim_type')
        # self._optim_types = self.help['type']
        # self._opt_type = [k for k in self._optim_types if self.type in self._optim_types[k]]

        _optim_type_dict = {
            'ADAM': functools.partial(torch.optim.Adam, lr=self.lr,
                                      **self.adam.__dict__),
            'SGD': functools.partial(torch.optim.SGD, lr=self.lr,
                                     **self.sgd.__dict__),
            'Rprop': functools.partial(torch.optim.Rprop, lr=self.lr,
                                       **self.rprop.__dict__),
            'RMSprop ': functools.partial(torch.optim.RMSprop, lr=self.lr,
                                          **self.rms_prop.__dict__),
            'ASGD': functools.partial(torch.optim.ASGD, lr=self.lr,
                                      **self.asgd.__dict__),
            'Adagrad': functools.partial(torch.optim.Adagrad, lr=self.lr,
                                         **self.adagrad.__dict__),
            # 'AdamW': functools.partial(torch.optim.AdamW, lr=self.lr,
            #                             **self.adamw.__dict__),
            'Adamax': functools.partial(torch.optim.Adamax, lr=self.lr,
                                        **self.adamax.__dict__),
            'Adadelta': functools.partial(torch.optim.Adadelta, lr=self.lr,
                                          **self.adadelta.__dict__),
        }

        if _opt_type == 'AdamW':
            try:
                _opt_fn = functools.partial(torch.optim.AdamW, lr=self.lr,
                                            **self.adamw.__dict__)
            except AttributeError:
                raise SystemError('AdamW optimizer is not available')
        else:
            _opt_fn = _optim_type_dict[_opt_type]

        return _opt_type, _opt_fn

    def get_scheduler(self):

        _schd_type = paramparse.match_opt(self, 'scheduler')
        # self._optim_types = self.help['type']
        # self._opt_type = [k for k in self._optim_types if self.type in self._optim_types[k]]

        _schd_type_dict = {
            'None': None,
            'StepLR': functools.partial(lr_scheduler.StepLR, **self.step.__dict__),
            'ReduceLROnPlateau': functools.partial(lr_scheduler.ReduceLROnPlateau, **self.plateau.__dict__),
            'CosineAnnealingLR': functools.partial(lr_scheduler.CosineAnnealingLR, **self.cos.__dict__),
            'ExponentialLR': functools.partial(lr_scheduler.ExponentialLR, **self.exp.__dict__),
        }

        _schd_fn = _schd_type_dict[_schd_type]

        return _schd_type, _schd_fn

    class CosineAnnealingLR:
        """
        :param NoneType T_max:
        :param int eta_min:
        :param int last_epoch:
        """

        def __init__(self):
            self.T_max = 10
            self.eta_min = 0
            self.last_epoch = -1

    class ExponentialLR:
        """
        :param NoneType gamma:
        :param int last_epoch:
        """

        def __init__(self):
            self.gamma = 0.1
            self.last_epoch = -1

    class ReduceLROnPlateau:
        """
        :param int cooldown:
        :param float eps:
        :param float factor:
        :param int min_lr:
        :param str mode:
        :param int patience:
        :param float threshold:
        :param str threshold_mode:
        :param bool verbose:
        """

        def __init__(self):
            self.cooldown = 0
            self.eps = 1e-08
            self.factor = 0.1
            self.min_lr = 0
            self.mode = 'max'
            self.patience = 10
            self.threshold = 0.0001
            self.threshold_mode = 'rel'
            self.verbose = False

    class StepLR:
        """
        :param float gamma:
        :param int last_epoch:
        :param NoneType step_size:
        """

        def __init__(self):
            self.gamma = 0.1
            self.last_epoch = -1
            self.step_size = 10

    class Adam:
        def __init__(self):
            self.amsgrad = False
            self.betas = (0.9, 0.999)
            self.eps = 1e-08
            self.weight_decay = 0

    class SGD:
        def __init__(self):
            self.dampening = 0
            self.momentum = 0
            self.nesterov = False
            self.weight_decay = 0

    class Adadelta:
        def __init__(self):
            self.eps = 1e-06
            self.rho = 0.9
            self.weight_decay = 0

    class Adagrad:
        def __init__(self):
            self.initial_accumulator_value = 0
            self.lr_decay = 0
            self.weight_decay = 0

    class AdamW:
        def __init__(self):
            self.amsgrad = False
            self.betas = (0.9, 0.999)
            self.eps = 1e-08
            self.weight_decay = 0.01

    class Adamax:
        def __init__(self):
            self.betas = (0.9, 0.999)
            self.eps = 1e-08
            self.weight_decay = 0

    class ASGD:
        def __init__(self):
            self.alpha = 0.75
            self.lambd = 0.0001
            self.t0 = 1000000.0
            self.weight_decay = 0

    class Rprop:
        def __init__(self):
            self.etas = (0.5, 1.2)
            self.step_sizes = (1e-06, 50)

    class RMSprop:
        def __init__(self):
            self.alpha = 0.99
            self.centered = False
            self.eps = 1e-08
            self.momentum = 0
            self.weight_decay = 0

    class LBFGS:
        def __init__(self):
            self.history_size = 100
            self.line_search_fn = None
            self.max_eval = None
            self.max_iter = 20
            self.tolerance_change = 1e-09
            self.tolerance_grad = 1e-05



def init_kaiming(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(m.bias, -bound, bound)


def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def process_class_metrics(metrics, metrics_vals):
    for _metric in metrics:
        tensor_val = metrics_vals[_metric]
        _metric_0 = f'{_metric}_0'
        _metric_1 = f'{_metric}_1'
        metrics_vals[_metric_0] = tensor_val[0].item()
        metrics_vals[_metric_1] = tensor_val[1].item()
        metrics_vals[_metric] = torch.mean(tensor_val).item()