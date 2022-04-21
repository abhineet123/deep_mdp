# from sklearn import svm as sksvm
from models.libsvm import svmutil as libsvm, _svm as libsvm_low

import os
import numpy as np
from datetime import datetime
from pprint import pformat

try:
    import cPickle as pickle
except:
    import pickle

from models.model_base import ModelBase
from utilities import linux_path, profile


class SVM(ModelBase):
    """
    :type _params: SVM.Params
    :type _weights: libsvm.svm_train
    :type _problem: libsvm.svm_problem
    :type _libsvm_params: libsvm.svm_parameter
    :type _train_features: np.ndarray
    :type _train_labels: np.ndarray
    :type labels: np.ndarray
    :type probabilities: np.ndarray
    """

    class Params(ModelBase.Params):
        """
        :ivar implementation: SVM implementation to use: 0: libsvm (high level)
        1: libsvm (low level) 2: sklearn.svm 3: lasvm

        :ivar verbose: Show detailed diagnostic messages, if any, from the underlying
        library

        """

        def __init__(self):
            ModelBase.Params.__init__(self)

            self.implementation = 0
            self.verbose = 0

    def __init__(self, params, logger, feature_shape, name):
        """
        :type params: SVM.Params
        :type logger: logging.RootLogger
        :rtype: None
        """
        ModelBase.__init__(self, params, logger, feature_shape, name, 'SVM', n_classes=2)

        self._params = params

        self._weights = None
        self._problem = None
        self._libsvm_params = None

        self._train_opts = '-c 1 -g 1 -b 1 -h 0'
        self._predict_opts = '-b 1'

        if not self._params.verbose:
            # self._train_opts += ' -q'
            self._predict_opts += ' -q'

        self._labels = None
        # self.accuracies = None
        self._probabilities = None
        self.accuracies = None

        if self._params.implementation == 0:
            self._logger.info('Using high level implementation')
            self._problem_fn = libsvm.svm_problem
        elif self._params.implementation == 1:
            self._logger.info('Using low level implementation')
            self._problem_fn = libsvm_low.svm_problem
        else:
            raise IOError('Invalid implementation type: {}'.format(self._params.implementation))

        if not self._params.accumulative:
            self._logger.warning('accumulative training is disabled')

        self._params.min_samples = int(self._params.min_samples)

        if self._params.min_samples > 0:
            self._logger.info('skipping training with less than {:d} samples'.format(self._params.min_samples))

        if self._params.min_samples_ratio > 0:
            self._logger.info(
                'skipping training with ratio of new samples less than {:.3f}'.format(self._params.min_samples_ratio))

        self._params.max_samples = int(self._params.max_samples)

        if self._params.max_samples > 0:
            self._logger.info('limiting the maximum number of training samples to {:d}'.format(self._params.max_samples))

    def train(self, labels, features):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :rtype: None
        """

        if not self._check_train(labels, features):
            return

        n_samples = self._train_labels.size

        _train_labels, _train_labels = self._train_labels, self._train_labels

        max_samples = int(self._params.max_samples)

        if max_samples > 0 and max_samples < n_samples:
            self._logger.info('randomly selecting {} / {} training samples'.format(
                max_samples, n_samples))

            all_idx = list(range(n_samples))
            random_idx = list(np.random.permutation(all_idx))
            random_sampled_idx = random_idx[:max_samples]

            _train_features = self._train_features[random_sampled_idx, ...]
            _train_labels = self._train_labels[random_sampled_idx]

            n_samples = max_samples
        else:
            _train_features, _train_labels = self._train_features, self._train_labels

        features_list = list(map(list, _train_features))
        labels_list = list(_train_labels)

        # params = libsvm.svm_parameter('-c 1 -q')
        # prob = libsvm.svm_problem([1,-1], [{1:1, 3:1}, {1:-1,3:-1}])

        start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self._logger.info(f'{start_time} :: training on {n_samples} samples')

        times = {}
        with profile('svm_train', times, show=1):
            self._problem = self._problem_fn(labels_list, features_list)
            self._libsvm_params = libsvm.svm_parameter(self._train_opts)
            self._weights = libsvm.svm_train(self._problem, self._libsvm_params)

        self._logger.info('time taken: {}'.format(times['svm_train']))

        self.is_trained = True
        self.is_loaded = False

        # self.logger.debug('Done')

    def predict(self, features, gt_labels=None, vis=None):
        """
        :type features: np.ndarray
        :type gt_labels: np.ndarray | None
        :type vis bool | int | None
        :rtype: None
        """
        n_samples = features.shape[0]
        if self._params.verbose:
            self._logger.info('predicting for {} samples'.format(n_samples))

        features_list = list(map(list, features))
        labels = -np.ones((n_samples,))
        labels_list = list(labels)

        _labels, _accuracies, _probabilities = libsvm.svm_predict(
            labels_list, features_list, self._weights, self._predict_opts)

        self._labels = np.array(_labels)
        self._probabilities = np.array(_probabilities)
        self.accuracies = np.array(_accuracies)
        return self._labels, self._probabilities

    def batch_train(self):
        """

        :return: str
        """
        _batch_params = self._params.batch  # type: ModelBase.Params.BatchTrain

        self._load_samples_recursive(_batch_params.db_path, non_negative_labels=False)

        self._split_samples(_batch_params)

        save_path = _batch_params.save_dir
        load_path = _batch_params.load_dir
        if not load_path:
            load_path = save_path

        if _batch_params.load_weights:
            self.load(load_path)

        # self.train(self._all_labels, self._all_features)

        n_samples = self._all_labels.size
        start_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        self._logger.info(f'{start_time} :: batch training on {n_samples} samples')

        times = {}
        with profile('svm_train', times, show=1):
            features_list = list(map(list, self._all_features))
            labels_list = list(self._all_labels)

            self._problem = self._problem_fn(labels_list, features_list)
            self._libsvm_params = libsvm.svm_parameter(self._train_opts)
            self._weights = libsvm.svm_train(self._problem, self._libsvm_params)

        self._logger.info('time taken: {}'.format(times['svm_train']))

        self.is_trained = True
        self.is_loaded = False

        if not save_path:
            save_root_dir = os.path.dirname(self._db_path[0])
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            save_path = linux_path(save_root_dir, f'svm_batch_{timestamp}')

        os.makedirs(save_path, exist_ok=True)

        model_save_path = self.save(save_path, save_samples=False)

        model_dict = {
            'model_save_path': model_save_path,
            'n_samples': n_samples,
            'timestamp': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        }
        pickle.dump(model_dict, open('{:s}/model_info.bin'.format(save_path), "wb"))

        return model_save_path

    def save(self, save_dir, save_samples=True):
        """
        :type save_path: str
        :type save_samples: bool | int
        :rtype: None
        """
        save_dir = self._check_save(save_dir)
        if save_dir is None:
            return

        model_save_path = linux_path(save_dir, 'model.bin')

        self._logger.info(f'Saving weights to {model_save_path}')
        libsvm.svm_save_model(model_save_path, self._weights)

        assert os.path.isfile(model_save_path), "model saving failed"

        if save_samples and self._params.accumulative:
            samples_save_dir = linux_path(save_dir, 'train_samples')
            self._save_train_samples(samples_save_dir, reset=not self._params.accumulative)

        return model_save_path

    def load(self, load_dir):
        """
        :type load_dir: str
        :type load_samples: bool | int
        :rtype: None
        """

        if self._params.batch.load_samples:
            self._load_train_samples(load_dir)
            return True
        """
        hack to allow custom load_path for the model independently of the target
        """
        if self._params.batch.load_dir:
            load_dir = self._params.batch.load_dir

        if os.path.isdir(load_dir):
            model_load_path = linux_path(load_dir, 'model.bin')
        else:
            model_load_path = load_dir

        if not os.path.exists(model_load_path):
            msg = 'weights file does not exist: {}'.format(model_load_path)
            raise IOError(msg)

        self._logger.info('Loading weights from: {}'.format(model_load_path))
        self._weights = libsvm.svm_load_model(model_load_path)

        assert self._weights is not None, "Model loading failed"

        model_info_path = linux_path(load_dir, 'model_info.bin')

        if os.path.isfile(model_info_path):
            model_info = pickle.load(open(model_info_path, "rb"))
            self._logger.info('Loaded model with train info: {}'.format(pformat(model_info)))

        # if load_samples and self._params.accumulative:
        #     samples_load_path = linux_path(load_dir, 'train_samples')
        #     self._load_train_samples(samples_load_path)

        self.is_trained = True
        self.is_loaded = True

        return True
