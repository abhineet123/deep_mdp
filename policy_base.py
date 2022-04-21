import os
import numpy as np
import pandas as pd

from models.model_base import ModelBase
from models.svm import SVM
from models.xgb import XGB
from models.mlp import MLP
from models.cnn import CNN
from models.dummy import Oracle, Random, Positive, Negative

from utilities import CustomLogger, AnnotationStatus, print_df, BaseParams, SaveModes, linux_path


class PolicyBase:
    """
    base class for all state policies

    :type _params: PolicyBase.Params
    :type _n_train_samples: int
    :type _stats: pd.Series
    :type _model: ModelBase

    """

    class Params(BaseParams):
        """
        base class for all state policy parameters
        :type oracle: Oracle.Params

        :ivar model: 'learning method used for decision making in the policy: '
                 'svm: Support Vector Machine,'
                 'xgb: XGBoost,'
                 'mlp: Multi Layer Perceptron,'
                 'cnn: Convolutional Neural Network',

        :ivar syn_samples: Number of synthetic samples to generate for each detection corresponding
        to a true positive object

        :ivar syn_neg_iou: (minimum, maximum)  IOU between a synthetic box and its anchor box
         for it to be considered a negative synthetic example

        :ivar syn_pos_iou: (minimum, maximum)  IOU between a synthetic box and its anchor box
         for it to be considered a positive synthetic example

        :ivar svm: 'parameters for the SVM module',
        :ivar xgb: 'parameters for the XGB module',
        :ivar mlp: 'parameters for the MLP module',
        :ivar cnn: 'parameters for the CNN module',
        :ivar vis: 'Enable diagnostic visualization',
        :ivar pause_for_debug: 'pause execution for debugging',

        """

        def __init__(self):
            self.model = 'svm'
            self.always_train = 0
            self.save_mode = 0

            self.syn_samples = 0
            self.syn_neg_iou = (0, 0.3)
            self.syn_pos_iou = (0.5, 0.8)

            self.enable_stats = 0

            """models
            """
            self.svm = SVM.Params()
            self.xgb = XGB.Params()
            self.mlp = MLP.Params()
            self.cnn = CNN.Params()
            self.random = Random.Params()
            self.oracle = Oracle.Params()
            self.pos = Positive.Params()
            self.neg = Negative.Params()

            self.verbose = 0
            self.profile = 0
            self.vis = 0
            self.pause_for_debug = 0

    class Decision:
        types = (
            'total',
            'correct',
            'incorrect',
            'ambiguous',
            'correct_positive',
            'correct_negative',
            'incorrect_positive',
            'incorrect_negative',
            'ambiguous_positive',
            'ambiguous_negative',
        )
        total, correct, incorrect, ambiguous, \
        correct_positive, correct_negative, \
        incorrect_positive, incorrect_negative, \
        ambiguous_positive, ambiguous_negative = types

    class Samples:
        labels = None
        features = None
        count = 0

    def __init__(self, params, name, logger, parent):
        """
        :type params: PolicyBase.Params
        :type logger: CustomLogger
        :type parent: PolicyBase
        :rtype: None
        """
        self._params = params
        self.__logger = CustomLogger(logger, names=(name,))
        self._name = name
        self._model = None
        self._n_train_samples = 0

        if parent is not None:
            self.stats = parent.stats
            self.pc_stats = parent.pc_stats
            self._test_mode = parent._test_mode
            self._oracle_type = parent._oracle_type
            self.samples = parent.samples
            self.synthetic_samples = parent.synthetic_samples
            # self._test_features = parent._test_features
            # self._test_labels = parent._test_labels
            # self.n_test_samples = parent.n_test_samples
            # self.n_all_test_samples = parent.n_all_test_samples
            self._prev_db_paths = parent._prev_db_paths
            self._writer = parent._writer
        else:
            self._test_mode = 0
            self._oracle_type = Oracle.Types.none
            self.samples = PolicyBase.Samples()
            self.synthetic_samples = PolicyBase.Samples()
            self._writer = None

            # self._test_features = None
            # self._test_labels = None
            # self.n_test_samples = 0
            # self.n_all_test_samples = 0
            self._prev_db_paths = []

            # percent_stats = pd.DataFrame(
            #     np.zeros((len(Policy.Decision.types),), dtype=np.int32),
            #     index=Policy.Decision.types)

            """absolute number of decisions for each type of target"""
            self.stats = pd.DataFrame(
                np.zeros((len(PolicyBase.Decision.types), len(AnnotationStatus.types)), dtype=np.int32),
                columns=AnnotationStatus.types,
                index=PolicyBase.Decision.types,
            )
            """Relative or percent of decisions by target type"""
            self.pc_stats = pd.DataFrame(
                np.zeros((len(PolicyBase.Decision.types) - 1, len(AnnotationStatus.types)), dtype=np.float32),
                columns=AnnotationStatus.types,
                index=PolicyBase.Decision.types[1:],
            )
        self._stats = None
        self._pc_stats = None
        self._cmb_stats = None
        self._cmb_pc_stats = None

    def get_model(self):
        """

        :rtype: ModelBase
        """
        return self._model

    def set_model(self, model):
        """

        :param ModelBase model:
        :return:
        """
        self._model = model

    def set_tb_writer(self, writer):
        self._writer = writer

    def _create_model(self, feature_shape, logger):
        if self._params.model == 'svm':
            self._model = SVM(self._params.svm, logger, feature_shape, name=self._name)
        elif self._params.model == 'xgb':
            self._model = XGB(self._params.xgb, logger, feature_shape, name=self._name)
        elif self._params.model == 'mlp':
            self._model = MLP(self._params.mlp, logger, feature_shape, name=self._name)
        elif self._params.model == 'cnn':
            self._model = CNN(self._params.cnn, logger, feature_shape, name=self._name)
        elif self._params.model == 'random':
            self._model = Random(self._params.random, logger, feature_shape, name=self._name)
        elif self._params.model == 'oracle':
            self._model = Oracle(self._params.oracle, logger, feature_shape, name=self._name)
            if self._params.oracle.type == Oracle.Types.absolute:
                assert self._params.save_mode == SaveModes.none, \
                    "Saving samples with absolute oracle is not supported yet"
                self._oracle_type = Oracle.Types.absolute
            else:
                self._oracle_type = Oracle.Types.relative

        elif self._params.model == 'pos':
            self._model = Positive(self._params.pos, logger, feature_shape, name=self._name)
        elif self._params.model == 'neg':
            self._model = Negative(self._params.neg, logger, feature_shape, name=self._name)

        elif self._params.model == 'none':
            self._model = None
        else:
            raise RuntimeError('Invalid model type: {}'.format(self._params.model))

        if self._model is not None:
            self.__logger.info('Using {} for policy decisions'.format(self._model.name))
        else:
            self.__logger.info('Using model-free heuristics for policy decisions')

    def _set_stats(self, ann_status):
        self._stats = self.stats[ann_status]
        self._pc_stats = self.pc_stats[ann_status]
        self._cmb_stats = self.stats['combined']
        self._cmb_pc_stats = self.pc_stats['combined']

    def _update_stats(self, correctness, decision):
        self._stats['total'] += 1
        self._cmb_stats['total'] += 1

        for _id in (correctness, correctness + '_' + decision):
            self._stats[_id] += 1
            self._cmb_stats[_id] += 1

        self._pc_stats.values[:] = (self._stats.values[1:] / self._stats['total']) * 100.0
        self._cmb_pc_stats.values[:] = (self._cmb_stats.values[1:] / self._cmb_stats[
            'total']) * 100.0

        if self._params.verbose:
            print_df(self.stats, self._name)
            print_df(self.pc_stats, self._name)

        # pause = 1

    def reset_stats(self):
        for col in self.stats.columns:
            self.stats[col].values[:] = 0
        for col in self.pc_stats.columns:
            self.pc_stats[col].values[:] = 0

    def test_mode(self):
        """

        :return:
        """
        self._test_mode = 1

    def train_mode(self):
        self._test_mode = 0

    def _add_test_samples(self, features, labels, pred_labels, is_synthetic):

        assert labels is not None and labels[0] is not None, "labels must be provided to add samples"
        assert labels.shape[0] == features.shape[0], "mismatch between features and labels sizes"

        if self._params.save_mode == SaveModes.none:
            return

        if not is_synthetic and self._params.save_mode == SaveModes.error:
            assert pred_labels is not None, "pred_labels must be provided to add samples in error mode"

            assert labels.shape[0] == pred_labels.shape[0], "mismatch between labels and pred_labels sizes"

            error_idx = np.flatnonzero(labels != pred_labels)
            if error_idx.size == 0:
                """no mistakes so do not save samples"""
                return

            features = features[error_idx, ...]
            labels = labels[error_idx]

        if is_synthetic:
            samples = self.synthetic_samples
        else:
            samples = self.samples

        if samples.features is not None:
            samples.features = np.concatenate((samples.features, features), axis=0)
            samples.labels = np.concatenate((samples.labels, labels))
        else:
            samples.features = np.copy(features)
            samples.labels = np.copy(labels)

        samples.count += labels.size
        # self.n_test_samples = self.samples.labels.size


    def batch_train(self):
        assert self._model is not None, "No model to batch train"

        self._model.batch_train()

    def reset_test_samples(self):
        self.samples.features = self.samples.labels = None
        self.synthetic_samples.features = self.synthetic_samples.labels = None

    def save_test_samples(self, save_dir):

        if self._params.save_mode == SaveModes.none:
            self.__logger.warning('save_test_samples called with save_mode none')
            return

        os.makedirs(save_dir, exist_ok=True)

        # assert self._test_features is not None, "test_features is None"

        # db_fname = 'model.bin' + '.npz'

        # db_path = linux_path(save_dir, db_fname)

        self._prev_db_paths.append(save_dir)

        _test_features = self.samples.features
        _test_labels = self.samples.labels

        if _test_features is None:
            self.__logger.warning('No test samples to save so saving empty arrays')
            _test_features = np.empty((0,))
            _test_labels = np.empty((0,))
        else:
            assert _test_labels is not None, "test_labels is None"

            assert _test_labels.shape[0] == _test_features.shape[0], f"Mismatch between n_samples in " \
                f"test_labels: {_test_labels.shape[0]} and test_features: {_test_features.shape[0]}"

            self.__logger.info('Saving {} test samples to: {}'.format(
                _test_labels.size, save_dir))

        # np.savez_compressed(db_path, features=_test_features, labels=_test_labels,
        #                     prev_db_paths=self._prev_db_paths)

        features_path = linux_path(save_dir, 'features.npy')
        labels_path = linux_path(save_dir, 'labels.npy')
        np.save(features_path, _test_features)
        np.save(labels_path, _test_labels)

        if self.synthetic_samples.features is not None:
            _test_syn_features = self.synthetic_samples.features
            _test_syn_labels = self.synthetic_samples.labels

            assert _test_syn_labels.shape[0] == _test_syn_features.shape[0], f"Mismatch between n_samples in " \
                f"test_labels: {_test_syn_labels.shape[0]} and test_features: {_test_syn_features.shape[0]}"

            self.__logger.info('Saving {} synthetic test samples to: {}'.format(
                _test_syn_labels.size, save_dir))
            features_path = linux_path(save_dir, 'synthetic_features.npy')
            labels_path = linux_path(save_dir, 'synthetic_labels.npy')
            np.save(features_path, _test_syn_features)
            np.save(labels_path, _test_syn_labels)

        prev_paths_path = linux_path(save_dir, 'prev_paths.npy')
        np.save(prev_paths_path, self._prev_db_paths)

    def load(self, load_dir):
        """
        :type load_dir: str
        :rtype: None
        """

        if self._model is None:
            self.__logger.warning('No model to load')
            return True

        if not self._model.load(load_dir):
            raise IOError('Failed to load model from {}'.format(load_dir))

        return True

    def _train(self, _features, _labels):

        self._model.train(_labels, _features)

        n_samples = _features.shape[0]
        self._n_train_samples += n_samples
