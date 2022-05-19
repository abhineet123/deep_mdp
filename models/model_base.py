import os
import multiprocessing
import functools
import time
import numpy as np
from pprint import pformat

import cv2

from utilities import CustomLogger, linux_path, BaseParams, load_samples_from_file, annotate_and_show, \
    get_nested_idx, get_class_idx


class ModelBase:
    """
    :type _logger: CustomLogger
    :type _parent_name: str

    """

    class Params(BaseParams):
        """
        base class for all model parameters
        :type accumulative: int
        :type batch: ModelBase.Params.BatchTrain
        :type verbose: int
        :type pause_for_debug: int

        :ivar accumulative: accumulate all samples available so far for each training,
        :ivar verbose: Show detailed diagnostic messages, if any,
        :ivar min_samples: Minimum number of training samples to collect before starting training;
                       only for online incremental training mode

        :ivar enable_non_batch: enable calling the train() function for standard non-batch training;
        Calling this function with this parameter disabled  raises an error

        :ivar parallel_load: Load saved samples from all sequences in parallel; seems to be buggy

        :ivar mem_mapped_load: Load saved samples as memory mapped arrays to avoid running out of memory when
        loading a large number of big features (e.g. raw image patches)
        """

        class BatchTrain:
            """

            :ivar save_criteria:  improvement in which metrics to use as cue to save a new checkpoint; one or more of:
            validation accuracy
            validation loss,
            training accuracy
            training loss

            :ivar splits: 3-tuple specifying the ratio of samples to use for training,
            validation and testing respectively;  data is first split into train and test based on test split and
            train data is then split into train and validation,

            :ivar load_weights: 0: Train from scratch;
            1: load previously saved weights and test;
            2: load previously saved weights and continue training;

            # following should only be set from IBT

            :ivar _db_path: path to load training samples from
            :ivar _save_samples: only store training samples for later batch training
            :ivar _load_dir: directory to load saved weights from
            :ivar _save_dir: directory to save weights to


            :ivar resample: 1: undersample class with more samples,
                            2: oversample class with fewer samples

            """

            def __init__(self):
                self.splits = (0.7, 0.3, 0.4)

                self.shuffle = 1
                self.n_workers = 1

                self.min_epochs = 100
                self.max_stasis_ratio = 0.
                self.acc_thresh = 0.

                self.resample = 1.

                self.val_batch_size = 0
                self.test_batch_size = 10

                self.print_gap = 1
                self.valid_gap = 1
                self.save_gap = 0

                self.load_weights = 0
                self.load_stats = 1
                self.load_opt = 0

                self.tb_vis = 0

                self.save_criteria = ['val_acc',
                                      'val_rec', 'val_rec_0', 'val_rec_1',
                                      'val_prec', 'val_prec_0', 'val_prec_1',
                                      'val_auroc']
                self.weights_name = 'weights.pt'

                """for debugging"""
                self.max_samples = 0

                """not user configurable - only for information passing in IBT"""
                self._save_samples = 0
                self._load_samples = 0
                self._save_dir = ''
                self._load_dir = ''
                self._db_path = ''

                # print()

            @property
            def save_samples(self): return self._save_samples

            @save_samples.setter
            def save_samples(self, val): self._save_samples = val

            @property
            def load_samples(self): return self._load_samples

            @load_samples.setter
            def load_samples(self, val): self._load_samples = val

            @property
            def save_dir(self): return self._save_dir

            @save_dir.setter
            def save_dir(self, val): self._save_dir = val

            @property
            def load_dir(self): return self._load_dir

            @load_dir.setter
            def load_dir(self, val): self._load_dir = val

            @property
            def db_path(self): return self._db_path

            @db_path.setter
            def db_path(self, db_path): self._db_path = db_path

        def __init__(self):
            self.accumulative = 1
            self.min_samples = 10
            self.max_samples = 0
            self.min_samples_ratio = 0.25

            self.batch = ModelBase.Params.BatchTrain()
            self.batch_size = 1000

            self.enable_non_batch = 0
            self.enable_tqdm = 1

            self.parallel_load = 0
            self.mem_mapped_load = 1
            self.nested_load = 1

            self.verbose = 0
            self.vis = 0
            self.pause_for_debug = 0

    def __init__(self, params, logger, feature_shape, parent_name, name, n_classes):
        """
        :type params: ModelBase.Params
        :type logger: CustomLogger
        :type feature_shape: list | None
        :type parent_name: str
        :type name: str
        :type n_classes: int
        :rtype: None
        """

        self.name = name

        self._logger = CustomLogger(logger, names=(parent_name, name))
        self._parent_name = parent_name
        self._n_classes = n_classes
        self._params = params

        self.feature_shape = list(feature_shape)
        self.n_features = int(np.prod(self.feature_shape))

        self._target_id = -1

        self._train_features = self._train_labels = None
        self._syn_features = self._syn_labels = None

        self._valid_features = self._valid_labels = None
        self._test_features = self._test_labels = None
        self._train_idx = self._val_idx = self._test_idx = None
        self._n_train = self._n_valid = self._n_test = 0
        self._n_samples = 0
        self._n_untrained_samples = 0

        self._all_features = self._all_labels = None
        self._db_path = None

        self.__prev_db_paths = []
        self._negative_label = 0

        self.is_trained = False
        self.is_loaded = False

    # def initialize(self, input_shape):
    #     self._input_shape = list(input_shape)
    #     self._batch_shape = [self._batch_params.batch_size, 1, ] + self._input_shape

    def set_id(self, target_id):
        self._target_id = target_id

    def _save_train_samples(self, save_dir, reset):
        # db_save_path = linux_path(save_dir, 'model.bin.npz')

        self.__prev_db_paths.append(save_dir)

        if self._train_features is None:
            self._logger.warning('No training samples to save so saving empty arrays')
            self._train_features = np.empty((0,))
            self._train_labels = np.empty((0,))
        else:
            assert self._train_labels is not None, "train_labels is None"

            assert self._train_labels.shape[0] == self._train_features.shape[0], f"Mismatch between n_samples in " \
                f"train_labels: {self._train_labels.shape[0]} and train_features: {self._train_features.shape[0]}"

            self._logger.info('Saving {} training samples to: {}'.format(
                self._train_labels.size, save_dir))

        os.makedirs(save_dir, exist_ok=True)

        features_path = linux_path(save_dir, 'features.npy')
        labels_path = linux_path(save_dir, 'labels.npy')

        np.save(features_path, self._train_features)
        np.save(labels_path, self._train_labels)

        if self._syn_features is not None:
            syn_features_path = linux_path(save_dir, 'synthetic_features.npy')
            syn_labels_path = linux_path(save_dir, 'synthetic_labels.npy')

            np.save(syn_features_path, self._syn_features)
            np.save(syn_labels_path, self._syn_labels)

        # np.savez_compressed(db_save_path, features=self._train_features, labels=self._train_labels,
        #                     prev_db_paths=self.__prev_db_paths)
        prev_paths_path = linux_path(save_dir, 'prev_paths.npy')
        np.save(prev_paths_path, self.__prev_db_paths)

        if reset:
            self._train_features = self._train_labels = None
            self._syn_features = self._syn_labels = None

    def _load_train_samples(self, load_path):
        """

        :param load_path:
        :return:
        """

        db_load_path = load_path + '.npz'

        if not os.path.isdir(db_load_path):
            load_dir = os.path.dirname(db_load_path)
        else:
            load_dir = db_load_path

        self._logger.info('Loading training samples from: {}'.format(load_dir))

        features_path = linux_path(load_dir, 'features.npy')
        self._train_features = np.load(features_path, mmap_mode='r')
        labels_path = linux_path(load_dir, 'labels.npy')
        self._train_labels = np.load(labels_path, mmap_mode='r')

        syn_features_path = linux_path(load_dir, 'synthetic_features.npy')
        if os.path.exists(syn_features_path):
            syn_labels_path = linux_path(load_dir, 'synthetic_labels.npy')
            assert os.path.exists(syn_labels_path), "syn_features_path exists but syn_labels_path does not"
            self._syn_features = np.load(syn_features_path, mmap_mode='r')
            self._syn_labels = np.load(syn_labels_path, mmap_mode='r')

            n_syn_samples = self._syn_features.shape[0]
            assert self._syn_labels.shape[0] == n_syn_samples, f"Mismatch between n_syn_samples in " \
                f"labels: {self._syn_labels.shape[0]} and features: {n_syn_samples}"

        prev_paths_path = linux_path(load_dir, 'prev_paths.npy')
        if os.path.exists(prev_paths_path):
            self.__prev_db_paths = np.load(prev_paths_path, mmap_mode='r')
            self.__prev_db_paths = list(self.__prev_db_paths)
        else:
            print(f'No prev_db_paths found')
            self.__prev_db_paths = []

        n_samples = self._train_features.shape[0]
        assert self._train_labels.shape[0] == n_samples, f"Mismatch between n_samples in " \
            f"labels: {self._train_labels.shape[0]} and features: {n_samples}"

        self._logger.info(f'Loaded {n_samples} samples')
        if self.__prev_db_paths:
            self._logger.info(f'Found prev_db_paths:\n{pformat(self.__prev_db_paths)}')
        else:
            self._logger.info(f'No prev_db_paths found')

        if db_load_path not in self.__prev_db_paths:
            self.__prev_db_paths.append(db_load_path)

        self._train_features = self._train_labels = None
        self._syn_features = self._syn_labels = None

    def _remove_none(self, samples, sample_counts):
        valid_samples_idx = [i for i, sample in enumerate(samples) if sample[0] is not None and sample[0].size > 0]
        samples = [samples[i] for i in valid_samples_idx]
        sample_counts = [sample_counts[i] for i in valid_samples_idx]

        return samples, sample_counts

    def _load_samples_recursive(self, db_path, non_negative_labels):
        """

        :param str db_path:
        :param bool non_negative_labels:
        :return:
        """
        assert db_path, "db_path must be provided for batch training"

        if ',' in db_path:
            self._db_path = db_path.split(',')
        else:
            self._db_path = [db_path, ]

        _start_t = time.time()

        load_func = functools.partial(
            load_samples_from_file,
            load_prev_paths=True,
            mem_mapped=self._params.mem_mapped_load,
        )

        if self._params.mem_mapped_load:
            print(f'Using memory mapped loading')
            self._params.nested_load = 1

        if self._params.parallel_load:
            n_cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(n_cpus)
            print(f'Loading samples using {n_cpus} threads')

            raw_db_data = pool.map(load_func, self._db_path)

        else:
            raw_db_data = map(load_func, self._db_path)

        samples, sample_counts, syn_samples, syn_sample_counts, prev_db_paths = zip(*raw_db_data)

        prev_db_paths = [item for sublist in prev_db_paths for item in sublist]

        if prev_db_paths:
            load_func = functools.partial(
                load_samples_from_file,
                load_prev_paths=False,
                mem_mapped=self._params.mem_mapped_load,
            )
            if self._params.parallel_load:
                recursive_db_data = pool.map(load_func, prev_db_paths)
            else:
                recursive_db_data = map(load_func, prev_db_paths)

            _samples, _sample_counts, _syn_samples, _syn_sample_counts = zip(
                *recursive_db_data)

            samples += _samples
            sample_counts += _sample_counts
            # class_idx += _class_idx

            syn_samples += _syn_samples
            syn_sample_counts += _syn_sample_counts
            # syn_class_idx += _syn_class_idx

        samples, sample_counts = self._remove_none(samples, sample_counts)

        all_features, all_labels = list(zip(*samples))

        all_labels = np.concatenate(all_labels, axis=0)

        # all_n_samples, all_n_pos_samples, all_n_neg_samples = list(zip(*sample_counts))
        # all_pos_idx, all_neg_idx = list(zip(*class_idx))

        if not self._params.nested_load:
            all_features = np.concatenate(all_features, axis=0)
            features_idx = None
        else:
            features_idx = get_nested_idx(sample_counts)

        _end_t = time.time()

        total_samples = sum(sample_counts)
        sps = total_samples / float(_end_t - _start_t)

        assert total_samples > 0, "No samples found in any db file"

        print('Done reading data for {} samples at {:.4f} samples/sec'.format(total_samples, sps))

        if non_negative_labels:
            all_labels[all_labels == -1] = 0

        self._all_features, self._all_labels, self._features_idx, self._n_samples = \
            all_features, all_labels, features_idx, total_samples

        if syn_samples[0] is not None:
            syn_samples, syn_sample_counts = self._remove_none(syn_samples, syn_sample_counts)

            self._syn_features, self._syn_labels = list(zip(*syn_samples))
            self._syn_labels = np.concatenate(self._syn_labels, axis=0)
            if non_negative_labels:
                self._syn_labels[self._syn_labels == -1] = 0

            self.n_sym_samples = sum(syn_sample_counts)
            self._syn_sample_counts = syn_sample_counts

            if not self._params.nested_load:
                self._syn_features = np.concatenate(self._syn_features, axis=0)
                self._syn_features_idx = None
            else:
                self._syn_features_idx = get_nested_idx(self._syn_sample_counts)

            # syn_n_samples, syn_n_pos_samples, syn_n_neg_samples = list(zip(*syn_sample_counts))
            # syn_pos_idx, syn_neg_idx = list(zip(*syn_class_idx))

    def _check_save(self, save_dir):

        if self._params.batch.save_dir:
            save_dir = self._params.batch.save_dir
            self._logger.warning(f'overriding save_path with that from batch_params: {save_dir}')

        if self._params.batch.save_samples:
            self._save_train_samples(save_dir, reset=True)
            return None

        if self.is_loaded:
            self._logger.warning(f'Skipping saving of loaded model')
            return None

        if not self.is_trained:
            self._logger.warning(f'Skipping saving of untrained model')
            return None

        return save_dir

    def _check_train(self, labels, features):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :rtype: bool
        """
        assert labels.shape[0] == features.shape[0], f"Mismatch between n_samples in " \
            f"labels: {labels.shape[0]} and features: {features.shape[0]}"

        if self._train_features is not None:
            if self._params.batch.save_samples or self._params.accumulative or \
                    self._n_untrained_samples > 0:
                self._train_features = np.concatenate((self._train_features, features), axis=0)
                self._train_labels = np.concatenate((self._train_labels, labels))
            else:
                self._train_features = np.copy(features)
                self._train_labels = np.copy(labels)
        else:
            self._train_features = np.copy(features)
            self._train_labels = np.copy(labels)

        self._n_untrained_samples += labels.shape[0]

        n_samples = self._train_labels.size

        assert self._train_features.shape[0] == n_samples, f"Mismatch between n_samples in " \
            f"features: {self._train_features.shape[0]} and labels: {n_samples}"

        if self._params.batch.save_samples:
            """only collecting training samples for saving"""
            if n_samples % 1000 == 0:
                self._logger.info(f'accumulated {n_samples} training samples')
            return False

        min_samples = int(self._params.min_samples)

        if self._params.min_samples_ratio > 0:
            min_samples_from_ratio = int((n_samples - self._n_untrained_samples) * self._params.min_samples_ratio)
            min_samples = min(min_samples, min_samples_from_ratio)

        if self._n_untrained_samples < min_samples:
            if n_samples % (min_samples/3) == 0:
                self._logger.info('Training is deferred since only {}/{} new samples are available'.format(
                    self._n_untrained_samples, min_samples))
            return False

        self._n_untrained_samples = 0


        return n_samples

    def _split_samples(self, _params):
        """

        :param ModelBase.Params.BatchTrain _params:
        :return:
        """

        n_samples = self._n_samples

        if n_samples == 0:
            self._logger.warning('No training samples found')
            self._train_idx = []
            self._valid_idx = []
            self._test_idx = []

            self._train_labels = []
            self._valid_labels = []
            self._test_labels = []

            self._n_train = self._n_valid = self._n_test = 0
            return

        pos_idx, neg_idx, n_pos_samples, n_neg_samples = get_class_idx(self._all_labels)

        self._logger.info(f'\nn_neg_samples: {n_neg_samples}\n'
                          f'n_pos_samples: {n_pos_samples}\n'
                          # f'class_weights: {class_weights}\n'
                          )

        if self._syn_features is not None and n_neg_samples != n_pos_samples:
            syn_pos_idx, syn_neg_idx, n_syn_pos_samples, n_syn_neg_samples = get_class_idx(self._syn_labels)

            self._logger.info(f'\nn_syn_neg_samples: {n_syn_neg_samples}\n'
                              f'n_syn_pos_samples: {n_syn_pos_samples}\n'
                              # f'class_weights: {class_weights}\n'
                              )
            if n_neg_samples > n_pos_samples:
                syn_samples_needed = n_neg_samples - n_pos_samples
                syn_type = 'positive'
                if syn_samples_needed > n_syn_pos_samples:
                    syn_samples_needed = n_syn_pos_samples
                syn_idx = syn_pos_idx
            else:
                syn_samples_needed = n_pos_samples - n_neg_samples
                syn_type = 'negative'
                if syn_samples_needed > n_syn_neg_samples:
                    syn_samples_needed = n_syn_neg_samples
                syn_idx = syn_neg_idx

            self._logger.info('Adding {} {} synthetic samples to make the class distribution balanced'.format(
                syn_samples_needed, syn_type
            ))
            random_syn_idx = list(np.random.permutation(syn_idx))
            random_syn_idx = random_syn_idx[:syn_samples_needed]

            syn_labels = self._syn_labels[random_syn_idx]

            n_real_samples = self._all_labels.size

            self._all_labels = np.concatenate((self._all_labels, syn_labels), axis=0)
            n_real_sample_sets = len(self._all_features)

            if self._params.nested_load:
                self._all_features += self._syn_features

                appended_syn_idx = {
                    n_real_samples + i: (n_real_sample_sets + self._syn_features_idx[k][0],
                                         self._syn_features_idx[k][1])
                    for i, k in enumerate(random_syn_idx)
                }
                self._features_idx.update(appended_syn_idx)
            else:
                syn_features = self._syn_features[random_syn_idx, ...]

                self._all_features = np.concatenate((self._all_features, syn_features), axis=0)

            self._n_samples = n_samples = self._all_labels.size

        train_split, valid_split, test_split = _params.splits

        assert train_split + valid_split == 1.0, \
            f"train_split: {train_split} and " \
                f"validation_split: {valid_split} " \
                f"do not add up to 1"

        n_test = int(n_samples * test_split)
        n_train = int((n_samples - n_test) * train_split)
        n_valid = int((n_samples - n_test - n_train))

        indices = list(range(n_samples))
        if _params.shuffle:
            self._logger.info('Shuffling dataset before splitting')
            indices = list(np.random.permutation(indices))

        indices = np.asarray(indices)

        train_idx, valid_idx, test_idx = indices[:n_train], indices[n_train:n_train + n_valid], \
                                         indices[n_train + n_valid:]

        self._train_labels = self._all_labels[train_idx]
        self._valid_labels = self._all_labels[valid_idx]
        self._test_labels = self._all_labels[test_idx]

        _, _, _n_train_pos, _n_train_neg = get_class_idx(self._train_labels)
        _, _, _n_valid_pos, _n_valid_neg = get_class_idx(self._valid_labels)
        _, _, _n_test_pos, _n_test_neg = get_class_idx(self._test_labels)

        self._logger.info(f'Loaded {n_samples} samples with following splits:\n'
                          f'train: {n_train} ({_n_train_pos} pos / {_n_train_neg} neg)\n'
                          f'validation: {n_valid} ({_n_valid_pos} pos / {_n_valid_neg} neg)\n'
                          f'test: {n_test} ({_n_test_pos} pos / {_n_test_neg} neg)\n'
                          )

        self._train_idx, self._val_idx, self._test_idx = train_idx, valid_idx, test_idx
        self._n_train, self._n_valid, self._n_test = n_train, n_valid, n_test

    def train(self, labels, features):
        """
        :type labels: np.ndarray
        :type features: np.ndarray
        :rtype: None
        """
        raise NotImplementedError()

    def predict(self, features, gt_labels, vis=0):
        """
        :type features: np.ndarray
        :type gt_labels: np.ndarray | None
        :type vis bool | int | None
        :rtype: np.ndarray, np.ndarray
        """
        raise NotImplementedError()

    def batch_train(self):
        raise NotImplementedError()

    def save(self, fname):
        """
        :type fname: str
        :rtype: None
        """
        raise NotImplementedError()

    def load(self, fname):
        """
        :type fname: str
        :rtype: None
        """
        raise NotImplementedError()

    def _vis_samples(self, features=None, labels=None, gt_labels=None, batch_size=None, return_only=0):
        raise NotImplementedError()

    def get_tb_image(self, features, labels, gt_labels, idx, batch_size):

        _features = features[idx]
        _labels = labels[idx]
        _gt_labels = gt_labels[idx]

        sort_idx = np.argsort(_gt_labels)

        vis_img = self._vis_samples(features=_features[sort_idx, ...], labels=_labels[sort_idx],
                                    gt_labels=_gt_labels[sort_idx],
                                    batch_size=batch_size, return_only=1)

        vis_img_uint8 = (vis_img * 255.0).astype(np.uint8)

        if self._params.vis:
            annotate_and_show('vis_img_uint8', vis_img_uint8, self._logger)

        vis_img_tb = cv2.cvtColor(vis_img_uint8, cv2.COLOR_BGR2RGB)
        vis_img_tb = np.transpose(vis_img_tb, axes=[2, 0, 1])

        return vis_img_uint8, vis_img_tb

    def add_tb_image(self, writer, features, labels, gt_labels, iteration, title, batch_size,
                     tb_vis_path, epoch, stats, show=0):

        if writer is not None:
            assert iteration is not None or stats is not None, \
                "at least one of iteration or stats must be provided for tb writing"

        correct = (labels == gt_labels)
        incorrect = np.logical_not(labels == gt_labels)

        pos = (labels == 1)
        neg = np.logical_not(pos)

        idx = {
            'correct_positive': np.flatnonzero(np.logical_and(correct, pos)),
            'correct_negative': np.flatnonzero(np.logical_and(correct, neg)),
            'incorrect_positive': np.flatnonzero(np.logical_and(incorrect, pos)),
            'incorrect_negative': np.flatnonzero(np.logical_and(incorrect, neg)),
        }

        assert sum(idx[k].size for k in idx) == labels.size, \
            "weird mismatch in correct / incorrect labels sizes"

        for _decision_type in idx:
            if not idx[_decision_type].size:
                continue
            if stats is not None:
                iteration = stats[_decision_type]

            vis_img, vis_img_tb = self.get_tb_image(features, labels, gt_labels, idx[_decision_type], batch_size)
            if writer is not None:
                writer.add_image('{}/{}'.format(title, _decision_type), vis_img_tb, iteration)

            if tb_vis_path:
                img_name = '{}_{}_{}_{}.jpg'.format(title, _decision_type, iteration, epoch)
                cv2.imwrite(os.path.join(tb_vis_path, img_name), vis_img)

            if show:
                annotate_and_show(f'{_decision_type}', vis_img)
