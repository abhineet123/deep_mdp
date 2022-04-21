import os
import sys
import re
import time
from contextlib import contextmanager
from collections import OrderedDict
from pprint import pformat
from datetime import datetime
import numpy as np
import pandas as pd
import shutil
import logging
import cv2

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator, State
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.utils import setup_logger
from ignite.contrib.metrics import ROC_AUC
from ignite.contrib.handlers import ProgressBar

import paramparse

from models.utilities import PyTorchOptParams, NLLLossOHEM, CrossEntropyLossOHEM, ImbalancedDatasetSampler, \
    init_kaiming, init_xavier, process_class_metrics

from models.model_base import ModelBase
from utilities import linux_path, CustomLogger, print_df, annotate_and_show


class NeuralNet(ModelBase):
    """
    :type _params: CNN.Params
    :type _train_features: np.ndarray | None
    :type _train_labels: np.ndarray | None
    :type _times: OrderedDict
    :type _net: nn.Module
    """

    class Params(ModelBase.Params):
        """
        :ivar profile: Enable code profiling

        :ivar ohem_ratio: ratio of hard examples if >0 otherwise OHEM is disabled

        :ivar init_type: {
        'Xavier': ('0', 'x', 'xavier'),
        'Kaiming': ('1', 'k', 'kaiming'),
        }

        :ivar loss_type: {
        'CrossEntropy': ('0', 'ce', 'cross_entropy'),
        'NLL': ('1', 'nll'),
        }

        :ivar progress_bar: show progress bar when batch training
        :ivar pretrained: Load pretrained weights, where supported

        """

        def __init__(self):
            ModelBase.Params.__init__(self)

            self.epochs = 100
            self.loss_type = 0
            self.ohem_ratio = 0.
            self.init_type = 0

            self.shuffle = 1
            self.num_workers = 0
            self.device = 'gpu'

            self.weighted_loss = 0
            self.resize_features = 4
            self.pretrained = 0
            self.progress_bar = 1

            self.opt = PyTorchOptParams()
            self.profile = 0

    class Data(data.Dataset):
        def __init__(self, features, labels, idx, input_shape, features_idx, name):
            self.labels = labels.astype(np.int64)
            self.features = features
            self.idx = idx
            self.input_shape = input_shape
            self.features_idx = features_idx
            self.name = name
            self._extracted_idx = []
            self._extracted_labels = []
            if self.idx is not None:
                self._len = len(self.idx)
            else:
                self._len = len(self.features)

        def set_idx(self, idx):
            self.idx = idx
            self._len = len(self.idx)

        def __len__(self):
            return self._len

        def __getitem__(self, index):
            if self.idx is not None:
                _index = self.idx[index]
            else:
                _index = index

            name = self.name
            idx1, idx2 = self.features_idx[_index]
            x = np.array(self.features[idx1][idx2].reshape(self.input_shape))
            y = self.labels[_index]

            self._extracted_idx.append((index, _index))
            self._extracted_labels.append(y)

            # if name == 'valid':
            #     print()

            return x, y

    def __init__(self, params, logger, feature_shape, parent_name, name, n_classes=2):
        """
        :type params: NeuralNet.Params
        :type logger: CustomLogger
        :rtype: None
        """

        assert 3 >= len(feature_shape) >= 2, "CNN can only be used with 2D or 3D inputs"
        if len(feature_shape) == 2:
            feature_shape = [1, ] + list(feature_shape)

        ModelBase.__init__(self, params, logger, feature_shape, parent_name, name, n_classes)

        """cannot move to Model for intellisense to work"""
        self._params = params

        self._batch_shape = [self._params.batch_size, ] + self.feature_shape

        self._weights = None
        self._train_labels = None
        self._train_features = None

        if self._params.device != 'cpu' and torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._logger.info('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
        else:
            self._device = torch.device("cpu")
            self._logger.info('Running on CPU')

        self._features = None
        self._labels = None
        self._probabilities = None
        self._net_out = None
        self._net = None
        self._loss_fn = self._optimizer = self._scheduler = None

        self._losses = []
        self._times = OrderedDict()

        self.__prev_saved_sanmples = []

        # def metric_to_str(x):
        #     if isinstance(x, torch.Tensor):
        #         return '{:.3f}, {:.3f}'.format(x[0].item(), x[1].item())
        #     return '{:.3f}'.format(x)

        self._metrics_dict = OrderedDict({
            "acc": (0, np.greater, Accuracy(
                output_transform=lambda out: (out[2], out[1])
            )),
            "prec": (0, np.greater, Precision(
                output_transform=lambda out: (out[2], out[1])
            )),
            "prec_0": (0, np.greater, None),
            "prec_1": (0, np.greater, None),

            "rec": (0, np.greater, Recall(
                output_transform=lambda out: (out[2], out[1])
            )),
            "rec_0": (0, np.greater, None),
            "rec_1": (0, np.greater, None),

            "auroc": (0, np.greater, ROC_AUC(
                output_transform=lambda y: (torch.argmax(y[2], dim=1), y[1])
            )),
        })
        """class specific metrics"""
        self._class_metrics = ('rec', 'prec')

        self._pause = 0

    def train(self, labels, features):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :rtype: None
        """

        # if self._params.pause_for_debug:
        #     print()

        _labels = np.copy(labels).astype(np.float32)
        _labels[_labels == -1] = 0
        _features = np.reshape(features, [-1, ] + self.feature_shape).astype(np.float32)

        if not self._check_train(_labels, _features):
            return

        raise AssertionError('neural net should only be trained in batch mode')

        n_samples = self._train_labels.size

        self._logger.info('Training on {} samples with feature size {} for {} epochs'.format(
            n_samples, self._train_features.shape[1], self._params.epochs))

        self._losses = []

        for epoch in range(self._params.epochs):
            self._times = OrderedDict()
            train_losses = []
            with self.profile('total'):
                # start_t = time.time()

                with self.profile('train'):
                    self._net.train()

                # train_t = time.time()
                # time_taken['train_t'] = train_t - start_t

                with self.profile('FloatTensor'):
                    features = torch.FloatTensor(self._train_features).to(self._device)
                    labels = torch.FloatTensor(self._train_labels).to(self._device)

                with self.profile('zero_grad'):
                    self._optimizer.zero_grad()

                with self.profile('_net'):
                    outputs = self._net(features)

                if self._params.pause_for_debug and (epoch + 1) % 100 == 0:
                    self._outputs = outputs.detach().cpu().numpy()
                    self._logger.info('paused')

                with self.profile('_loss_fn'):
                    loss = self._loss_fn(outputs, labels.long())

                with self.profile('backward'):
                    loss.backward()

                with self.profile('step'):
                    self._optimizer.step()

            train_losses.append(loss.item())

            if self._params.profile:
                _total = self._times['total']
                for k in self._times:
                    self._times[k] = (self._times[k], (self._times[k] / _total) * 100)
                print('times:\n{}'.format(pformat(self._times)))

            self._losses.append(np.mean(train_losses))

            if self._params.verbose:
                sys.stdout.write('\repoch : {}, train loss : {}'.format(
                    epoch + 1, np.mean(train_losses)))
                sys.stdout.flush()

        if not self._params.accumulative:
            self._train_labels = None
            self._train_features = None

        self.is_trained = True
        self.is_loaded = False

        # self.logger.debug('Done')

    def predict(self, features, gt_labels=None, vis=None):
        """
        :type features: np.ndarray
        :type gt_labels np.ndarray
        :type vis bool | int | None
        :rtype: np.ndarray, np.ndarray
        """
        assert self.is_trained, "Untrained model should not be used for prediction"

        self._features = features.reshape((-1, *self.feature_shape))

        if vis is None:
            vis = self._params.vis

        self._net.eval()
        with torch.no_grad():
            net_out = self._net.forward(torch.FloatTensor(self._features).to(self._device))
            net_out = net_out.cpu().numpy()
            self._net_out = net_out

            """
            Swapping columns so that class 0 or negative association is the second column to conform with the annoying
            format used in the original code with libsvm that places class +1 in the first column in -1 in the second 
            for some asinine reason 
            """
            self._probabilities = net_out[:, [1, 0]]

            self._labels = np.argmax(net_out, axis=1)

            """
            convert class 0 to -1 to conform with asinine libsvm some more
            """
            self._labels[self._labels == 0] = -1

            if vis:
                self._vis_samples(self._features, self._labels, gt_labels=gt_labels,
                                  batch_size=self._features.shape[0])

        return self._labels, self._probabilities

    def _resample(self, resample, train_dataset, train_neg_idx, train_pos_idx):
        if resample == 1:
            self._logger.info(f'Using class resampling with ImbalancedDatasetSampler')
            train_sampler = ImbalancedDatasetSampler(train_dataset, self._train_idx)
        elif resample == 2:
            self._logger.info(f'Using class resampling with majority class random subset selection')
            n_train_neg_samples = train_neg_idx.size
            n_train_pos_samples = self._n_train - n_train_neg_samples
            if n_train_neg_samples > n_train_pos_samples:
                _train_neg_idx = np.random.permutation(train_neg_idx)[:n_train_pos_samples]
                _train_idx = np.random.permutation(
                    np.concatenate((_train_neg_idx, train_pos_idx), axis=0)).astype(np.int32)
                _train_idx = self._train_idx[_train_idx]
            elif n_train_neg_samples < n_train_pos_samples:
                _train_pos_idx = np.random.permutation(train_pos_idx)[:n_train_neg_samples]
                _train_idx = np.random.permutation(
                    np.concatenate((_train_pos_idx, train_neg_idx), axis=0)).astype(np.int32)
                _train_idx = self._train_idx[_train_idx]
            else:
                _train_idx = self._train_idx
            train_sampler = SubsetRandomSampler(_train_idx)

            resampled_labels = self._all_labels[_train_idx]
            resampled_neg_samples = np.flatnonzero(resampled_labels == 0)
            resampled_pos_samples = np.flatnonzero(resampled_labels == 1)
            n_resampled = len(resampled_labels)
            n_resampled_neg = len(resampled_neg_samples)
            n_resampled_pos = len(resampled_pos_samples)

            self._logger.info(f'n_resampled: {n_resampled}')
            self._logger.info(f'n_resampled_neg: {n_resampled_neg}')
            self._logger.info(f'n_resampled_pos: {n_resampled_pos}')

        else:
            self._logger.info(f'Class resampling is disabled')
            if self._params.shuffle:
                train_sampler = SubsetRandomSampler(self._train_idx)
            else:
                train_dataset.set_idx(self._train_idx)
                train_sampler = None
                # train_sampler = SequentialSampler(self._train_idx)

        return train_sampler

    def batch_train(self):
        """

        :return: str
        """

        assert self._params.batch_size > 1, "batch_size must be > 1 to avoid batch normalization issues"

        _batch_params = self._params.batch  # type: ModelBase.Params.BatchTrain

        self._load_samples_recursive(_batch_params.db_path, non_negative_labels=True)

        if self._params.tee_log:
            self._params.progress_bar = 0

        if self._n_samples == 0:
            self._logger.warning('No samples found so skipping batch training')
            return None

        if 0 < _batch_params.max_samples < self._n_samples:
            random_idx = np.random.permutation(range(self._n_samples))[:_batch_params.max_samples]
            self._all_features = self._all_features[random_idx, ...]
            self._all_labels = self._all_labels[random_idx, ...]
            self._n_samples = _batch_params.max_samples

        self._split_samples(_batch_params)

        if self._params.vis:
            self._vis_samples()

        self._init_opt()

        """disable annoyingly verbose ignite logging
        """
        logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.ERROR)

        train_neg_samples = (self._train_labels == 0)
        train_neg_idx = np.flatnonzero(train_neg_samples)
        train_pos_idx = np.flatnonzero(np.logical_not(train_neg_samples))

        n_train_neg_samples = train_neg_idx.size
        n_train_pos_samples = self._n_train - n_train_neg_samples

        assert n_train_pos_samples == train_pos_idx.size, f"mismatch in number of pos samples: " \
            f"{n_train_pos_samples}, {train_pos_idx.size}"

        self._logger.info(f'\nn_train_neg_samples: {n_train_neg_samples}\n'
                          f'n_train_pos_samples: {n_train_pos_samples}\n'
                          )
        if self._params.weighted_loss:
            self._logger.info('Using weighted loss')
            train_class_weights = [float(n_train_neg_samples) / self._n_train,
                                   float(n_train_pos_samples) / self._n_train]

            self._logger.info(f'train_class_weights: {train_class_weights}\n')
            self._loss_fn = self._get_loss_fn(train_class_weights)

        self._metrics_dict["loss"] = (
            np.inf, np.less, Loss(self._loss_fn, output_transform=lambda out: (out[2], out[1])))

        save_dir = _batch_params.save_dir
        load_dir = _batch_params.load_dir
        weights_name = _batch_params.weights_name

        if not save_dir:
            save_root_dir = os.path.dirname(self._db_path[0])
            save_dir = linux_path(save_root_dir, f'mlp_batch_{datetime.now().strftime("%y%m%d_%H%M%S")}')

        if not _batch_params.load_weights and os.path.isdir(save_dir):
            self._logger.info(f'deleting existing weights folder: {save_dir}\n')
            shutil.rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)

        if not load_dir:
            load_dir = save_dir

        train_dataset = NeuralNet.Data(self._all_features, self._all_labels, None,
                                       self.feature_shape, self._features_idx, 'train')

        valid_dataset = NeuralNet.Data(self._all_features, self._all_labels, self._val_idx,
                                       self.feature_shape, self._features_idx, 'valid')

        test_dataset = NeuralNet.Data(self._all_features, self._all_labels, self._test_idx,
                                      self.feature_shape, self._features_idx, 'test')

        # loader_params = {'shuffle': self._params.shuffle,
        #                  'num_workers': self._params.num_workers}

        train_labels = self._all_labels[self._train_idx]
        val_labels = self._all_labels[self._val_idx]

        val_metrics = {
            k: v[2] for k, v in self._metrics_dict.items() if v[2] is not None
        }
        train_evaluator = create_supervised_evaluator(self._net, metrics=val_metrics, device=self._device,
                                                      output_transform=lambda x, y, y_pred: (x, y, y_pred))
        train_evaluator.state.type = 'train'
        train_evaluator.logger = setup_logger("train_evaluator", level=logging.ERROR)

        train_evaluator.state.opt_metrics = {
            k: v[0] for k, v in self._metrics_dict.items()
        }
        train_evaluator.state.opt_epochs = {
            k: 0 for k, v in self._metrics_dict.items()
        }
        """separate validation evaluator for convenience to avoid copying the metrics every time
        """
        val_evaluator = create_supervised_evaluator(self._net, metrics=val_metrics, device=self._device,
                                                    output_transform=lambda x, y, y_pred: (x, y, y_pred))
        val_evaluator.state.type = 'val'

        train_evaluator.logger = setup_logger("val_evaluator", level=logging.ERROR)

        val_evaluator.state.opt_metrics = {
            k: v[0] for k, v in self._metrics_dict.items()
        }
        val_evaluator.state.opt_epochs = {
            k: 0 for k, v in self._metrics_dict.items()
        }

        metrics_functions = {
            k: v[1] for k, v in self._metrics_dict.items()
        }
        # metrics_fmt = {
        #     k: v[3] for k, v in metrics_dict.items()
        # }

        opt_metrics = {
            'train': train_evaluator.state.opt_metrics,
            'val': val_evaluator.state.opt_metrics
        }
        opt_epochs = {
            'train': train_evaluator.state.opt_epochs,
            'val': val_evaluator.state.opt_epochs,
        }

        # test_sampler = SequentialSampler(self._test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=self._params.batch.test_batch_size,
                                                  # sampler=test_sampler,
                                                  num_workers=self._params.num_workers,
                                                  )
        chkpt = None

        if _batch_params.load_weights:
            chkpt = self._load_weights(load_dir, weights_name)
            if _batch_params.load_weights == 1:
                assert self._n_test > 0, "No samples to test on"

                val_evaluator.run(test_loader)
                return save_dir

        trainer = create_supervised_trainer(self._net, self._optimizer, self._loss_fn, device=self._device,
                                            output_transform=lambda x, y, y_pred, loss: (x, y, y_pred, loss.item()))
        trainer.logger = setup_logger("trainer", level=logging.ERROR)
        start_epoch = 0

        if chkpt is not None and _batch_params.load_stats:
            start_epoch = chkpt['epoch']
            for data_type in opt_metrics:
                data_opt_metrics = opt_metrics[data_type]
                data_opt_epochs = opt_epochs[data_type]

                for metric in data_opt_metrics:
                    metric_id = f'{data_type}_{metric}'

                    if 'opt' in chkpt:
                        data_opt_metrics[metric], data_opt_epochs[metric] = chkpt['opt'][metric_id]
                    else:
                        """opt values unavailable in this checkpoint so use its own metric values instead"""
                        data_opt_metrics[metric] = chkpt[metric_id]
                        data_opt_epochs[metric] = chkpt['epoch']

        else:
            self._logger.info('Skipping stats loading')
        trainer.state.prev_save_epoch = start_epoch
        trainer.state.weights_paths = {k: None for k in _batch_params.save_criteria + ['regular', ]}

        trainer.state.get_last_lr = trainer.state.get_lr = trainer.state.opt_lr = self._params.opt.lr

        # n_train_batches = int(math.ceil(self._n_train / _batch_size))
        save_gap = _batch_params.save_gap

        max_stasis_ratio = _batch_params.max_stasis_ratio
        min_epochs = _batch_params.min_epochs
        acc_thresh = _batch_params.acc_thresh * 100.

        train_sampler = self._resample(_batch_params.resample, train_dataset, train_neg_idx, train_pos_idx)

        self._logger.info(f'batch_size: {self._params.batch_size}')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   sampler=train_sampler,
                                                   batch_size=self._params.batch_size,
                                                   num_workers=self._params.num_workers,
                                                   )

        val_batch_size = self._params.batch.val_batch_size
        if val_batch_size <= 0:
            val_batch_size = self._params.batch_size

        # val_sampler = SequentialSampler(self._val_idx)
        val_loader = torch.utils.data.DataLoader(valid_dataset,
                                                 batch_size=val_batch_size,
                                                 num_workers=self._params.num_workers,
                                                 )

        tb_path = linux_path(save_dir, 'tb')
        writer = SummaryWriter(log_dir=tb_path)

        self._logger.info(f'Saving weights to {save_dir}')
        self._logger.info(f'Saving tensorboard summary to: {tb_path}')

        if _batch_params.tb_vis:
            self._logger.warning(f'tensorboard visualization is enabled')

        tb_vis_path = None

        if _batch_params.tb_vis == 2:
            tb_vis_path = linux_path(tb_path, 'vis')
            os.makedirs(tb_vis_path, exist_ok=True)
            self._logger.warning(f'Saving vis to: {tb_vis_path}')

        if self._params.progress_bar:
            pbar = ProgressBar(persist=True)
            pbar.attach(trainer, metric_names="all")

        self._pause = 1

        def on_iteration_eval(engine):
            _iter = engine.state.iteration
            epoch = engine.state.epoch

            x, y, y_pred = engine.state.output

            features = x.cpu().numpy()
            gt_labels = y.cpu().numpy()
            labels_raw = y_pred.detach().cpu().numpy()
            labels = np.argmax(labels_raw, axis=1)
            batch_size = gt_labels.size

            title = engine.state.type

            if _batch_params.tb_vis:
                self.add_tb_image(writer, features, labels, gt_labels,
                                  iteration=_iter,
                                  tb_vis_path=tb_vis_path,
                                  title=title,
                                  batch_size=batch_size,
                                  epoch=epoch,
                                  stats=None)

        @trainer.on(Events.STARTED)
        def setup_state(engine):
            engine.state.epoch = start_epoch

        @trainer.on(Events.ITERATION_COMPLETED)
        def on_iteration_train(engine):
            iter = engine.state.iteration
            # epoch = engine.state.epoch
            x, y, y_pred, loss = engine.state.output

            writer.add_scalar('train/batch_loss', loss, iter)

        @trainer.on(Events.EPOCH_COMPLETED)
        def on_epoch(engine):
            save_weights = []
            save_dict = {}
            epoch = engine.state.epoch
            train_evaluator.run(train_loader)
            eval_metrics = {'train': train_evaluator.state.metrics}

            if self._n_valid and epoch % _batch_params.valid_gap == 0:
                val_evaluator.run(val_loader)
                eval_metrics.update({'val': val_evaluator.state.metrics})

            status_df = pd.DataFrame(
                np.zeros((len(eval_metrics.keys()), len(self._metrics_dict.keys())), dtype=np.float32),
                index=eval_metrics.keys(),
                columns=self._metrics_dict.keys(),
            )
            for _metric in self._metrics_dict:
                status_df[_metric] = status_df[_metric].astype(object)

            save_dict['opt'] = {}

            for data_type in eval_metrics:
                data_metrics = eval_metrics[data_type]
                process_class_metrics(self._class_metrics, data_metrics)

                data_opt_metrics = opt_metrics[data_type]
                data_opt_epochs = opt_epochs[data_type]

                for _metric, _value in data_metrics.items():

                    _metric_id = f'{data_type}_{_metric}'
                    _cmp_fn = metrics_functions[_metric]

                    _opt_value = data_opt_metrics[_metric]

                    writer.add_scalar(f'{data_type}/{_metric}', data_metrics[_metric], epoch)

                    if _cmp_fn(_value, _opt_value):
                        data_opt_metrics[_metric] = _value
                        data_opt_epochs[_metric] = epoch

                        if _metric_id in _batch_params.save_criteria:
                            save_weights.append(_metric_id)

                    save_dict[_metric_id] = _value
                    save_dict['opt'][_metric_id] = (data_opt_metrics[_metric], data_opt_epochs[_metric])

                    status_df[_metric][data_type] = '{:.2f}, {:.2f}, {:d}'.format(
                        _value * 100, data_opt_metrics[_metric] * 100, data_opt_epochs[_metric])

            self._logger.info("Training Results - Epoch: {}:\n".format(epoch))
            print_df(status_df)
            print()

            if not save_weights and ((save_gap > 0 and epoch % save_gap == 0) or epoch == self._params.epochs):
                save_weights = ['regular', ]

            """Save checkpoint"""
            if save_weights:
                save_dict.update({
                    'net': self._net.state_dict(),
                    'optimizer': self._optimizer.state_dict(),
                    'n_train_pos': n_train_pos_samples,
                    'n_train_neg': n_train_neg_samples,
                    'n_train': self._n_train,
                    'n_valid': self._n_valid,
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                })
                weights_paths = engine.state.weights_paths
                for _id, criterion in enumerate(save_weights):
                    if criterion == 'regular':
                        _weights_name = 'regular_{}'.format(weights_name)
                    else:
                        _weights_name = weights_name
                        engine.state.prev_save_epoch = epoch

                    weights_path = linux_path(save_dir, '{}.{:d}'.format(_weights_name, epoch))

                    if _id == 0 or not os.path.exists(weights_path):
                        print(f'saving weights for {criterion} to {os.path.basename(weights_path)}')
                        torch.save(save_dict, weights_path)
                    # else:
                    #     print(f'skipping saving weights for {criterion}')

                    prev_weights_path = weights_paths[criterion]
                    weights_paths[criterion] = weights_path

                    if prev_weights_path is not None and os.path.exists(prev_weights_path):
                        other_criteria = [k for k, v in weights_paths.items() if v == prev_weights_path]
                        if not other_criteria:
                            print(f'deleting weights for {criterion}: {os.path.basename(prev_weights_path)}')
                            os.remove(prev_weights_path)
                        # else:
                        #     print(f'skipping deleting weights {os.path.basename(prev_weights_path)} '
                        #           f'needed for {other_criteria}')

            if self._scheduler is not None:
                # engine.state.current_lr = self._scheduler.get_lr()
                try:
                    engine.state.get_last_lr = self._scheduler.get_last_lr()
                except AttributeError:
                    engine.state.get_last_lr = self._scheduler.get_lr()

                # engine.state.get_lr = self._scheduler.get_lr()
                engine.state.opt_lr = self._optimizer.param_groups[0]['lr']

                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(eval_metrics['val']['auroc'])
                else:
                    self._scheduler.step()

            writer.add_scalar('learning_rate/get_last_lr', engine.state.get_last_lr[0], epoch)
            # writer.add_scalar('learning_rate/get_lr', engine.state.get_lr, epoch)
            writer.add_scalar('learning_rate/opt_lr', engine.state.opt_lr, epoch)

            print(f'epoch: {epoch}')
            print(f'get_last_lr: {engine.state.get_last_lr[0]}')
            # print(f'get_lr: {engine.state.get_lr}')
            print(f'opt_lr: {engine.state.opt_lr}')

            max_train_acc = train_evaluator.state.opt_metrics['acc']
            max_valid_acc = val_evaluator.state.opt_metrics['acc']
            if max_train_acc > acc_thresh > 0 and max_valid_acc > acc_thresh:
                self._logger.info('Ending training as both  train_acc and valid_acc exceed acc_thresh {}'.format(
                    acc_thresh))
                engine.terminate()

            stasis_ratio = (epoch - engine.state.prev_save_epoch) / (epoch + 1)
            if stasis_ratio > max_stasis_ratio > 0 and epoch > min_epochs:
                self._logger.info('Ending training as stasis_ratio {} exceeds max_stasis_ratio {}'.format(
                    stasis_ratio, max_stasis_ratio))
                engine.terminate()

        train_evaluator.add_event_handler(Events.ITERATION_COMPLETED, on_iteration_eval)
        val_evaluator.add_event_handler(Events.ITERATION_COMPLETED, on_iteration_eval)

        start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self._logger.info(f'{start_time} :: starting batch training')

        trainer.run(train_loader, max_epochs=self._params.epochs)

        weights_paths = sorted([weights_path for weights_path in list(set(trainer.state.weights_paths.values()))
                                if weights_path is not None])

        if weights_paths:
            weights_names = [os.path.basename(weights_path) for weights_path in weights_paths]

            rows = [f'{k} train' for k in weights_names] + [f'{k} val' for k in weights_names]

            if self._n_test > 0:
                rows += [f'{k} test' for k in weights_names]

            cols = list(self._metrics_dict.keys())
            weights_status_df = pd.DataFrame(
                np.zeros((len(rows), len(cols)), dtype=np.float32),
                index=rows,
                columns=cols,
            )
            for _id, weights_path in enumerate(weights_paths):
                weights_name = weights_names[_id]
                chkpt = torch.load(weights_path, map_location=self._device)  # load checkpoint
                for data_type in opt_metrics:
                    weights_id = f'{weights_name} {data_type}'
                    data_opt_metrics = opt_metrics[data_type]
                    for _metric in data_opt_metrics.keys():
                        _metric_id = f'{data_type}_{_metric}'
                        try:
                            weights_status_df[_metric][weights_id] = chkpt[_metric_id] * 100
                        except KeyError:
                            weights_status_df[_metric][weights_id] = -1

                if self._n_test > 0:
                    weights_id = f'{weights_name} test'

                    self._net.load_state_dict(chkpt['net'])
                    val_evaluator.run(test_loader)
                    test_metrics = val_evaluator.state.metrics
                    process_class_metrics(self._class_metrics, test_metrics)

                    for _metric in test_metrics:
                        weights_status_df[_metric][weights_id] = test_metrics[_metric] * 100

            print_df(weights_status_df, fmt='.2f')
            print()

        self.is_trained = True
        self.is_loaded = False

        return save_dir

    def _load_weights(self, weights_path, weights_name, epoch=-1):
        """

        :param weights_path:
        :param weights_name:
        :param state_info:
        :param allow_missing:
        :param load_stats:
        :param epoch:
        :return:
        """

        if os.path.isdir(weights_path):
            if epoch >= 0:
                weights_path = linux_path(weights_path, f'{weights_name}.{epoch}')
                if not os.path.exists(weights_path):
                    raise IOError(f'Checkpoint file {weights_path} for epoch {epoch} does not exist')
            else:
                matching_ckpts = [k for k in os.listdir(weights_path) if
                                  os.path.isfile(linux_path(weights_path, k)) and
                                  k.startswith(weights_name)]
                if not matching_ckpts:
                    raise IOError('No checkpoints found matching {} in {}'.format(weights_name, weights_path))

                matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
                weights_path = linux_path(weights_path, matching_ckpts[-1])

        chkpt = torch.load(weights_path, map_location=self._device)  # load checkpoint

        if chkpt is None:
            raise IOError('Loading weights failed')

        rows = ['train', 'val']
        cols = list(self._metrics_dict.keys())
        weights_status_df = pd.DataFrame(
            np.zeros((len(rows), len(cols)), dtype=np.float32),
            index=rows,
            columns=cols,
        )

        for data_type in rows:
            for _metric in self._metrics_dict.keys():
                _metric_id = f'{data_type}_{_metric}'
                try:
                    weights_status_df[_metric][data_type] = chkpt[_metric_id] * 100
                except KeyError:
                    weights_status_df[_metric][data_type] = -1
                except ValueError:
                    weights_status_df[_metric][data_type] = -1

        load_txt = 'Loading weights from: {} with:\n' \
                   '\tepoch: {}\n' \
                   '\ttimestamp: {}\n'.format(
            weights_path,
            chkpt['epoch'],
            chkpt['timestamp'])

        try:
            load_txt += '\tn_train: {} - {} pos / {} neg\n' \
                        '\tn_valid: {}\n'.format(
                chkpt['n_train'],
                chkpt['n_train_pos'],
                chkpt['n_train_neg'],
                chkpt['n_valid'],
            )
        except KeyError:
            pass

        self._logger.info(load_txt)
        print_df(weights_status_df, fmt='.2f')
        print()

        self._net.load_state_dict(chkpt['net'])
        if self._params.batch.load_opt:
            self._optimizer.load_state_dict(chkpt['optimizer'])
        else:
            self._logger.info('Skipping optimizer parameter loading')

        chkpt['weights_path'] = weights_path

        return chkpt

    def save(self, save_dir):
        """
        :type save_dir: str
        :rtype: None
        """

        if self._params.batch.save_samples:
            self._save_train_samples(save_dir, reset=True)
            return

        if self.is_loaded:
            self._logger.warning(f'Skipping saving of loaded model')
            return

        if not self.is_trained:
            self._logger.warning('Not saving model as it is not trained yet')
            return

        assert self._params.enable_non_batch, "Non-batch training is disabled"

        save_path = linux_path(save_dir, self._params.batch.weights_name)

        self._logger.info('Saving weights to: {}'.format(save_path))

        chkpt = {'best_loss': self._losses,
                 'model': self._net.state_dict(),
                 'optimizer': self._optimizer.state_dict()}

        torch.save(chkpt, save_path)

    def load(self, load_dir):
        """
        :type load_dir: str
        :rtype: None
        """
        if os.path.isdir(load_dir):
            load_dir = linux_path(load_dir, 'model.bin')

        if self._params.batch.load_samples:
            self._load_train_samples(load_dir)
        else:
            load_path = self._params.batch.load_dir
            if not load_path:
                load_path = load_dir

            self._load_weights(load_path, self._params.batch.weights_name)

            self.is_trained = True
            self.is_loaded = True

        return True

    def _init_opt(self):

        if not self._params.pretrained:
            assert self._net is not None, "net must be created before initializing optimization"

            _init_type = paramparse.match_opt(self._params, 'init_type')

            if _init_type == 'Xavier':
                self._logger.info('Using Xavier weight initialization with zero bias')
                self._net.apply(init_xavier)
            elif _init_type == 'Kaiming':
                self._logger.info('Using Kaiming weight initialization with uniform bias')
                self._net.apply(init_kaiming)

        if self._optimizer is None:
            _opt_type, _opt_fn = self._params.opt.get_optimizer()

            self._logger.info('Using {} optimizer with params:\n{}'.format(
                _opt_type, pformat(_opt_fn.keywords)))

            self._optimizer = _opt_fn(self._net.parameters())

        if self._scheduler is None:
            _schd_type, _schd_fn = self._params.opt.get_scheduler()
            if _schd_fn is not None:
                self._logger.info('Using {} scheduler with params:\n{}'.format(
                    _schd_type, pformat(_schd_fn.keywords)))
                self._scheduler = _schd_fn(self._optimizer)
            else:
                self._scheduler = None

        if self._loss_fn is None:
            self._loss_fn = self._get_loss_fn()

    def _get_loss_fn(self, class_weights=None):
        _loss_type = str(self._params.loss_type)
        self._loss_types = paramparse.obj_from_docs(self._params, 'loss_type')

        _loss_type_str = [k for k in self._loss_types if _loss_type in self._loss_types[k]]
        if not _loss_type_str:
            raise IOError('Invalid loss_type: {}'.format(self._params.loss_type))
        _loss_type_str = _loss_type_str[0]

        ohem_ratio = self._params.ohem_ratio
        self._logger.info('Using {} loss'.format(_loss_type_str))
        if ohem_ratio > 0:
            self._logger.info('Using OHEM with ratio: {}'.format(ohem_ratio))

            if class_weights is not None:
                self._logger.warning('Weighted samples not supported with OHEM')

            if _loss_type_str == 'CrossEntropy':
                return CrossEntropyLossOHEM(ratio=ohem_ratio, device=self._device)
            elif _loss_type_str == 'NLL':
                return NLLLossOHEM(ratio=ohem_ratio, device=self._device)
            else:
                raise IOError('Invalid loss_type: {}'.format(self._params.loss_type))
        else:
            if class_weights is not None:
                class_weights = torch.Tensor(class_weights).to(self._device)
            if _loss_type_str == 'CrossEntropy':
                return nn.CrossEntropyLoss(weight=class_weights)
            elif _loss_type_str == 'NLL':
                return nn.NLLLoss(weight=class_weights)
            else:
                raise IOError('Invalid loss_type: {}'.format(self._params.loss_type))

    @contextmanager
    def profile(self, _id):
        if self._params.profile:
            start_t = time.time()
            yield None
            end_t = time.time()
            self._times[_id] = end_t - start_t
        else:
            yield None
