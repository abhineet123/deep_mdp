import scipy
import numpy as np

from models.model_base import ModelBase
from utilities import CustomLogger


class Random(ModelBase):
    class Params(ModelBase.Params):
        """
        """

        def __init__(self):
            super(Random.Params, self).__init__()

    def __init__(self, params, logger, feature_shape, name='', n_classes=2):
        """
        :type logger: CustomLogger
        :rtype: None
        """
        ModelBase.__init__(self, params, logger, feature_shape, name, 'RandomClassifier', n_classes)
        self._logger = logger
        self._probabilities = None
        self._labels = None
        self.is_trained = 1

    def batch_train(self):
        self._logger.warning('RandomClassifier cannot be trained')

    def train(self, labels, features):
        self._logger.warning('RandomClassifier cannot be trained')

    def predict(self, features, gt_labels=None, vis=None):
        n_samples = features.shape[0]

        rand_vals = np.random.rand(n_samples, self._n_classes)

        # rand_vals_sum = np.sum(rand_vals, axis=1)
        # rand_vals_norm = rand_vals, rand_vals_sum[:, None]

        rand_vals_norm = scipy.special.softmax(rand_vals, axis=1)

        self._labels = np.argmax(rand_vals_norm, axis=1)
        self._labels[self._labels == 0] = -1

        """policies expect first column to correspond to +1 and second one to -1"""
        self._probabilities = rand_vals_norm[:, [1, 0]]

        return self._labels, self._probabilities

    def save(self, fname, save_samples=True):
        """
        :type fname: str
        :type save_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to save')

    def load(self, fname, ignore_missing=False, load_samples=True):
        """
        :type fname: str
        :type ignore_missing: bool
        :type load_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to load')
        return True


class Negative(ModelBase):
    class Params(ModelBase.Params):
        """
        """

        def __init__(self):
            super(Negative.Params, self).__init__()

    def __init__(self, params, logger, feature_shape, name='', n_classes=2):
        """
        :type logger: CustomLogger
        :rtype: None
        """
        ModelBase.__init__(self, params, logger, feature_shape, name, 'NegativeClassifier', n_classes)
        self._logger = logger
        self._probabilities = None
        self._labels = None
        self.is_trained = 1

    def batch_train(self):
        self._logger.warning('NegativeClassifier cannot be trained')

    def train(self, labels, features):
        self._logger.warning('NegativeClassifier cannot be trained')

    def predict(self, features, gt_labels=None, vis=None):
        n_samples = features.shape[0]

        self._labels = np.full((n_samples,), -1)

        """policies expect first column to correspond to +1 and second one to -1"""
        self._probabilities = np.zeros((n_samples, 2), dtype=np.float32)
        self._probabilities[:, 1] = 1

        return self._labels, self._probabilities

    def save(self, fname, save_samples=True):
        """
        :type fname: str
        :type save_samples: bool
        :rtype: None
        """
        self._logger.warning('Nothing to save')

    def load(self, fname, ignore_missing=False, load_samples=True):
        """
        :type fname: str
        :type ignore_missing: bool
        :type load_samples: bool
        :rtype: None
        """
        self._logger.warning('Nothing to load')
        return True


class Positive(ModelBase):
    class Params(ModelBase.Params):
        """
        """

        def __init__(self):
            super(Positive.Params, self).__init__()

    def __init__(self, params, logger, feature_shape, name='', n_classes=2):
        """
        :type logger: CustomLogger
        :rtype: None
        """
        ModelBase.__init__(self, params, logger, feature_shape, name, 'PositiveClassifier', n_classes)
        self._logger = logger
        self._probabilities = None
        self._labels = None
        self.is_trained = 1

    def batch_train(self):
        self._logger.warning('PositiveClassifier cannot be trained')

    def train(self, labels, features):
        self._logger.warning('PositiveClassifier cannot be trained')

    def predict(self, features, gt_labels=None, vis=None):
        n_samples = features.shape[0]

        self._labels = np.full((n_samples,), 1)

        """policies expect first column to correspond to +1 and second one to -1"""
        self._probabilities = np.zeros((n_samples, 2), dtype=np.float32)
        self._probabilities[:, 0] = 1

        return self._labels, self._probabilities

    def save(self, fname, save_samples=True):
        """
        :type fname: str
        :type save_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to save')

    def load(self, fname, ignore_missing=False, load_samples=True):
        """
        :type fname: str
        :type ignore_missing: bool
        :type load_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to load')
        return True


class Oracle(ModelBase):
    class Types:
        none, relative, absolute = range(3)
        to_str = {
            none: 'none',
            relative: 'relative',
            absolute: 'absolute',
        }

    class Params(ModelBase.Params):
        """
        """

        def __init__(self):
            super(Oracle.Params, self).__init__()
            self.type = Oracle.Types.relative

    def __init__(self, params, logger, feature_shape, name='', n_classes=2):
        """
        :type params: Oracle.Params
        :type logger: CustomLogger
        :rtype: None
        """
        assert params.type != Oracle.Types.none, "invalid Oracle type"

        model_name = '{} oracle'.format(Oracle.Types.to_str[params.type])

        ModelBase.__init__(self, params, logger, feature_shape, name, model_name, n_classes)
        self._logger = logger
        self._probabilities = None
        self._labels = None
        self.is_trained = 1

    def batch_train(self):
        self._logger.warning('Oracle cannot be trained')

    def train(self, labels, features):
        self._logger.warning('Oracle cannot be trained')

    def predict(self, features, gt_labels=None, vis=None):
        assert gt_labels is not None, "gt_labels must be provided"

        n_samples = features.shape[0]

        self._labels = np.zeros((n_samples,), dtype=np.int32)
        self._labels[:] = gt_labels

        pos_labels = np.flatnonzero(gt_labels == 1)

        """policies expect first column to correspond to +1 and second one to -1"""
        self._probabilities = np.zeros((n_samples, 2), dtype=np.float32)
        self._probabilities[pos_labels, 0] = 1
        self._probabilities[:, 1] = 1 - self._probabilities[:, 0]

        return self._labels, self._probabilities

    def save(self, fname, save_samples=True):
        """
        :type fname: str
        :type save_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to save')

    def load(self, fname, ignore_missing=False, load_samples=True):
        """
        :type fname: str
        :type ignore_missing: bool
        :type load_samples: bool
        :rtype: None
        """
        self._logger.warning('No model to load')
        return True
