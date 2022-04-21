import copy
import functools
import time
import os
import sys
import math
import importlib
import numpy as np
from scipy.signal import argrelmax
import cv2

import paramparse

from trackers.tracker_base import TrackerBase

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pytracking'))

from pytracking.tracker.atom import ATOM
from pytracking.tracker.eco import ECO
from pytracking.tracker.dimp import DiMP

from utilities import TrackingStatus, nms, get_neighborhood, annotate_and_show


class PYT(TrackerBase):
    """
    :type _params: PYT.Params

    """

    class Params(TrackerBase.Params):
        """
        :type variant: int


        :ivar variant: {
            'ECO': ('0', 'eco'),
            'ATOM': ('1', 'atom'),
            'DiMP': ('2', 'dimp'),
            'PrDiMP': ('3', 'prdimp'),
        }

        :ivar score_size: 'size to which to resize the tracker score map - needed to provide uniformity across '
                      'all variants that not only have different shapes but also determine these during '
                      'tracker initialization',
        :ivar stacked: 'use stacked implementation of LK - here all patches are stacked onto a single large image',
        :ivar feature_type: '0: scores flattened '
                '1: maximum values in each row and column of scores concatenated '
                '2: top <n_features> maxima values in scores (with nms for resized scores) ',
        :ivar n_features: 'only used if feature_type = 2',
        :ivar nms_dist_ratio: 'fraction of the maximum dimension of the score map used as the distance threshold '
                  'while performing non-maximum suppression for feature and status extraction',
        :ivar vis: 'Enable diagnostic visualization',

        """

        def __init__(self):
            TrackerBase.Params.__init__(self)

            self.variant = 'atom'
            self.config = 'default'

            self.stacked = 0
            self.score_size = 288

            self.feature_scale = 0.
            self.n_features = 10

            self.nms_method = 0
            self.nms_dist = 0
            self.nms_dist_ratio = 0
            self.max_type = 2
            self.conf_thresh = 0.90
            self.non_best_scores = 1
            self.normalize_scores = 0

            self.feat_thickness = 1
            self.feat_ops = '0'

            # self.atom = ATOMParams()
            # self.eco = ECOParams()

    class Result(TrackerBase.Result):
        """
        :type _tracker: PYT
        """

        def __init__(self, locations, features, status, conf, scores, tracker):
            TrackerBase.Result.__init__(self, locations, features, status, conf, None, 1)
            self._scores = scores
            self._tracker = tracker
            self._pause = 1

        def get_summarized_features(self, template_ids, track_id):
            """
            :param np.ndarray template_ids:
            :param int track_id:
            :return:
            """

            assert template_ids is not None and template_ids.size > 0, "template_ids is None or empty"

            if template_ids.size == 1:
                mean_score = self._scores[track_id][template_ids.item()]
            else:
                template_ids = template_ids.squeeze()

                curr_scores = [self._scores[track_id][_id.item()] for _id in template_ids]

                _scores = np.stack(curr_scores, axis=2)

                mean_score = np.mean(_scores, axis=2)

            # self._pause = annotate_and_show('siamese:get_summarized_features', [mean_score, ] + curr_scores,
            #                                 grid_size=(3, 4), pause=self._pause)

            score_id = np.argmax(mean_score)
            score_id = np.unravel_index(score_id, mean_score.shape)

            features = PYT.features_from_score_map(self._tracker, mean_score, score_id)

            features = features[0]

            # print()

            return features

        def set_status(self, valid_idx, invalid_idx, track_id=0):
            # self.flags[valid_idx, :] = flags[valid_idx].reshape((valid_idx.size, 1))
            # self.features[valid_idx, :] = features[valid_idx, :]

            assert self.heuristics, "tracker heuristics are disabled"

            if valid_idx.size == 0:
                self.status[track_id, :].fill(TrackingStatus.failure)
            else:
                self.status[track_id, invalid_idx] = TrackingStatus.failure

        def get_scores(self, track_id):
            assert self.heuristics, "tracker heuristics are disabled"

            return self.conf[track_id, :]

        def _get_best_template(self, track_id, template_ids=None):
            """
            Returns the ID of the best tracked patch
            :rtype: int
            """

            if template_ids is not None:
                _conf = np.copy(self.conf[track_id, :])
                mask = np.ones(len(_conf), np.bool)
                mask[template_ids] = 0
                _conf[mask] = 0
                return np.argmax(_conf)

            """object with the maximum confidence is the best one"""
            return np.argmax(self.conf[track_id, :])

        def _get_worst_id(self, exclude_id=None, track_id=0):
            """
            Returns the ID of the worst tracked patch with an optional exclusion

            :param exclude_id: optional ID of the patch to exclude
            :type exclude_id: int | None
            :type track_id: int
            :rtype: int
            """

            # object with the minimum confidence is the worst one
            if exclude_id is None:
                return np.argmin(self.conf[track_id, :])
            else:
                _conf = np.copy(self.conf[track_id, :])
                _conf[exclude_id] = 1
                return np.argmin(_conf)

    def __init__(self, **kwargs):
        TrackerBase.__init__(self, 'siamese', **kwargs)

        if self._parent is None:
            self._tracker_params = next(self._spawn_ids_gen)
            self.score_shape = next(self._spawn_ids_gen)
            self._score_sz = next(self._spawn_ids_gen)
            self._scaled_score_sz = next(self._spawn_ids_gen)
            self._feat_ops = next(self._spawn_ids_gen)
            self._feat_ops_str = next(self._spawn_ids_gen)
            self._n_neighborhoods = next(self._spawn_ids_gen)
            self._n_features = next(self._spawn_ids_gen)
            self._feat_cy = next(self._spawn_ids_gen)
            self._feat_cx = next(self._spawn_ids_gen)
            self._nms_dist = next(self._spawn_ids_gen)
            self._max_dist_sqr = next(self._spawn_ids_gen)

            self._create_tracker = next(self._spawn_ids_gen)
            self._tracker = next(self._spawn_ids_gen)

            self._register()
        else:
            self._spawn()

        """needed for intellisense"""
        self._params = kwargs["params"]  # type: PYT.Params

        if self._parent is None:
            pyt_variants = paramparse.obj_from_docs(self._params, 'variant')
            try:
                variant = [k for k in pyt_variants if self._params.variant in pyt_variants[k]][0]
            except IndexError:
                raise AssertionError('Invalid variant: {}'.format(self._params.variant))

            self._logger.info(f'Using variant {variant} with config {self._params.config}')

            if self._update_location:
                self._logger.warning('Location updating is enabled')

            tracker_name = variant.lower()
            param_name = self._params.config
            self._tracker_params = self.get_parameters(tracker_name, param_name)
            params = dict(
                params=self._tracker_params, update_location=self._update_location
            )
            if self._params.variant in pyt_variants['ECO']:
                self._create_tracker = functools.partial(ECO, **params)
            elif self._params.variant in pyt_variants['ATOM']:
                self._create_tracker = functools.partial(ATOM, **params)
            elif self._params.variant in pyt_variants['DiMP']:
                self._create_tracker = functools.partial(DiMP, **params)
            else:
                raise IOError('Invalid variant: {}'.format(self._params.variant))

            self._score_sz = self._params.score_size
            self.score_shape = (self._score_sz, self._score_sz)
            self.init_features()

        self._trackers = {}
        for i in range(self._n_templates):
            self._trackers[i] = self._create_tracker()

        self._pause = 1
        self.locations = None
        self._features = None
        self._status = None
        self._conf = None

        self.update = self.initialize

        if self._params.stacked:
            raise NotImplementedError('stacked not implemented yet')

    def init_features(self):
        if self._params.feature_scale > 0:
            self._scaled_score_sz = int(self._score_sz / self._params.feature_scale)

        if self._params.nms_dist >= 0:
            self._nms_dist = self._params.nms_dist
        else:
            self._nms_dist = int(self._params.nms_dist_ratio * self._score_sz)
        self._nms_dist_sqr = self._nms_dist ** 2

        feat_ops_dict = {
            '0': (np.amax, 'max'),
            '1': (np.mean, 'mean'),
            '2': (np.median, 'median'),
            '3': (np.amin, 'min'),
        }

        self._feat_ops = [feat_ops_dict[_op][0] for _op in self._params.feat_ops]
        self._feat_ops_str = [feat_ops_dict[_op][1] for _op in self._params.feat_ops]

        self._n_neighborhoods = self._params.n_features - 1

        if self._params.feature_type == 0:
            self._logger.info('Using flattened score map directly as features')
            if self._scaled_score_sz is not None:
                self._logger.info(f'Scaling it down to {self._scaled_score_sz}x{self._scaled_score_sz}')
                self.feature_shape = (self._scaled_score_sz, self._scaled_score_sz)
            else:
                self.feature_shape = (self._score_sz, self._score_sz)

        elif self._params.feature_type == 1:
            self._logger.info('Using maximum values in each row and column of score map as features')

            n_features = self._score_sz + self._score_sz
            self.feature_shape = (n_features,)
        elif self._params.feature_type == 2:
            assert self._params.nms_method != 1, "feature_type 2 is not supported with nms_method 1"

            if self._params.nms_method == 0:
                n_features = 1 + self._n_neighborhoods * len(self._feat_ops)
                self.feature_shape = (n_features,)
                self._logger.info(
                    'Using distance from center combined with {} over neighborhood for total feature size {} '.format(
                        self._feat_ops_str, n_features))
            else:
                self.feature_shape = (self._params.n_features,)
                self._logger.info(
                    'Using top {} maxima values in scores (without nms) as features'.format(self.n_features))
                if self._nms_dist > 0:
                    self._logger.info(f'Using nms_dist: {self._nms_dist}')

        if self._params.nms_method == 0:
            self._logger.info('NMS is disabled')
        elif self._params.nms_method == 1:
            self._logger.info('Using standard NMS')
        elif self._params.nms_method == 2:
            self._logger.info('Using fast approximate NMS')

        _h = _w = self._score_sz
        self._feat_cy, self._feat_cx = int(_h / 2), int(_w / 2)

        max_offset_x = float(max(self._feat_cx, _w - self._feat_cx - 1))
        max_offset_y = float(max(self._feat_cy, _h - self._feat_cy - 1))

        self._max_dist_sqr = max_offset_x ** 2 + max_offset_y ** 2

        self._logger.info('feature_shape: {}'.format(self.feature_shape))
        self._logger.info('_nms_dist_sqr: {}'.format(self._nms_dist_sqr))

    @staticmethod
    def get_parameters(name, parameter_name):
        """Get parameters."""
        param_module = importlib.import_module('pytracking.parameter.{}.{}'.format(name, parameter_name))
        params = param_module.parameters()
        return params

    def set_region(self, template_id, frame, bbox):
        self._trackers[template_id].set_region(frame, bbox)

    def initialize(self, template_id, frame, bbox):
        self._trackers[template_id].initialize(frame, bbox)

        # if self.score_shape is None:
        #     self.score_shape = tuple(self.trackers[template_id].output_sz.detach().cpu().numpy().astype(np.int32))
        #
        #     """dummy tracking to get score size - temporary hack till more detailed code examination can be
        #     completed"""
        #     # tracker_out = self.trackers[template_id].track(frame)
        #     # _, _score = tracker_out['bbox'], tracker_out['score_map']
        #     # _score = _score.squeeze()
        #     # self.score_shape = _score.shape
        #
        #     self._score_sz = self.score_shape[0]
        #     self.init_features()

    def track(self, patches, heuristics, vis=None):
        """

        :param np.ndarray patches:
        :param bool | int heuristics:
        :param bool | int vis:
        :return:
        :rtype Siamese.Result
        """

        self._heuristics = heuristics

        if vis is None:
            vis = self._params.vis

        n_patches = patches.shape[0]
        n_features = np.prod(self.feature_shape)

        self._locations = np.zeros((n_patches, self._n_templates, 4))
        self._features = np.zeros((n_patches, self._n_templates, n_features), dtype=np.float32)
        self._conf = self._status = None

        if self._heuristics:
            self._conf = np.zeros((n_patches, self._n_templates), dtype=np.float32)
            self._status = np.zeros((n_patches, self._n_templates), dtype=np.int32)

        scores = {}

        for patch_id in range(n_patches):
            patch_2 = np.squeeze(patches[patch_id, :, :])
            # annotate_and_show(f'Siamese :: patch {patch_id}', patch_2, f'siamese:_track:track{patch_id}')

            scores[patch_id] = []
            for template_id in range(self._n_templates):
                tracker_out = self._trackers[template_id].track(patch_2)
                bbox, _score = tracker_out['bbox'], tracker_out['score_map']

                _score = _score.squeeze()

                if _score.shape != self.score_shape:
                    """temporary hack to account for the weird behaviour in ECO that causes it to change 
                    its score shape after the first target"""
                    # self._logger.warning(f'score_shape mismatch: {_score.shape} and {self.score_shape}')
                    # self.score_shape = _score.shape
                    # self._score_sz = self.score_shape[0]
                    # self.init_features()
                    _score = cv2.resize(_score, self.score_shape)

                min_score, max_score = np.amin(_score), np.amax(_score)

                # if min_score < 0 or max_score > 1:
                assert max_score != min_score, f"min_score and min_score are equal: {min_score}"

                _score = (_score - min_score) / (max_score - min_score)

                _score_id = np.argmax(_score)
                _score_id = np.unravel_index(_score_id, _score.shape)

                scores[patch_id].append(_score)

                # if isinstance(_score_id, int):
                #     _score_id = np.unravel_index(_score_id, _score.shape)

                _features, _flag, _conf = self._get_features(_score, _score_id)

                if vis:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[0] + bbox[2]),
                           int(bbox[1] + bbox[3]))

                    frame_disp = np.copy(patch_2)
                    cv2.rectangle(frame_disp, pt1, pt2, (0, 0, 255), thickness=2)

                    # cv2.imshow('frame {}'.format(tracker_id), frame_disp)
                    # cv2.imshow('raw_score_img {}'.format(tracker_id), raw_score_)
                    # cv2.imshow('score_img {}'.format(tracker_id), score_)

                    self._logger.info(f'tracker{template_id}')
                    self._pause = annotate_and_show('siamese:_get_features', [frame_disp, _score],
                                                    self._logger, pause=self._pause)

                    patch_2_disp = np.copy(patch_2)
                    feature_img = _features.reshape(self.feature_shape)
                    self._logger.info(f'target_id {self._target_id} patch_id {patch_id} template_id {template_id}')
                    self._pause = annotate_and_show('frame_disp',
                                                    [patch_2_disp, feature_img],
                                                    self._logger, pause=self._pause, grid_size=(1, -1))

                if self._heuristics:
                    self._conf[patch_id, template_id] = _conf
                    if _flag:
                        self._status[patch_id, template_id] = TrackingStatus.success
                    else:
                        self._status[patch_id, template_id] = TrackingStatus.failure

                # bbox, score_ = self.trackers[p1].update(patch_2)7
                self._locations[patch_id, template_id, :] = bbox
                self._features[patch_id, template_id, :] = _features

        return PYT.Result(self._locations, self._features, self._status, self._conf, scores, self)

    @staticmethod
    def features_from_score_map(self, score_, score_id_):

        _features = np.zeros((self.n_features,), dtype=np.float32)
        py, px = score_id_
        _status = _conf = None

        if self._params.feature_type == 0:
            """flattened raw score map (optionally resized) 
            """
            if self._scaled_score_sz is not None:
                score_ = cv2.resize(score_, (self._scaled_score_sz, self._scaled_score_sz),
                                    interpolation=cv2.INTER_AREA)
            _features[:] = score_.flatten()
            if not self._heuristics:
                return _features, _status, _conf
        elif self._params.feature_type == 1:
            """maximum in each row and column
            """
            _features[:] = np.concatenate((
                np.amax(score_, axis=0).reshape((1, self._score_sz)),
                np.amax(score_, axis=1).reshape((1, self._score_sz)),
            ), axis=1)
            if not self._heuristics:
                return _features, _status, _conf

        if self._params.nms_method == 0:
            """
            statistics from concentric neighborhoods of increasing radii around the maximum
            """
            diff = self._params.feat_thickness

            neighborhood = [get_neighborhood(score_, px, py, self._nms_dist + diff * k, self._score_sz,
                                             self._params.max_type, diff)
                            for k in range(self._n_neighborhoods)]

            feat_conf = []
            for _op in self._feat_ops:
                feat_conf += list(map(lambda x: 1 - math.exp(-1.0 / _op(x)), neighborhood))

            if self._params.feature_type == 2:
                _features[:-1] = feat_conf

                feat_dist = 1. - ((px - self._feat_cx) ** 2 + (py - self._feat_cy) ** 2) / self._max_dist_sqr

                _features[-1] = feat_dist

                if not self._heuristics:
                    return _features, _status, _conf

            _conf = np.mean(feat_conf)

            if _conf > self._params.conf_thresh:
                _status = 1
            else:
                _status = 0

            return _features, _status, _conf

        best_score = score_[py, px]

        if self._params.nms_method == 1:
            # start_t = time.time()

            """set a fixed squared region around the maximum to zero"""

            start_x = int(max(0, px - self._nms_dist))
            start_y = int(max(0, py - self._nms_dist))

            end_x = int(min(self._score_sz - 1, px + self._nms_dist))
            end_y = int(min(self._score_sz - 1, py + self._nms_dist))

            score_[start_y:end_y, start_x:end_x] = 0
            second_best_score = np.amax(score_)

            # end_t = time.time()
            # maxima_time_taken = end_t - start_t
        else:
            # start_t = time.time()
            # raw_score_maxima_loc = argrelextrema(score_, np.greater, order=5)
            raw_score_maxima_loc = argrelmax(score_, order=5)

            # raw_score_maxima = filters.maximum_filter(score_, self._nms_dist)
            # cv2.imshow('raw_score_maxima', raw_score_maxima)

            # end_t = time.time()
            # maxima_time_taken = end_t - start_t

            # cv2.imshow('score_', score_)
            #
            # k = cv2.waitKey(0)
            # if k == 27:
            #     sys.exit()

            raw_score_maxima = score_[raw_score_maxima_loc[0], raw_score_maxima_loc[1]]
            raw_score_maxima_sorted_idx = np.argsort(raw_score_maxima)[::-1]
            raw_score_maxima_loc_sorted = [k[raw_score_maxima_sorted_idx] for k in raw_score_maxima_loc]

            if self._nms_dist_sqr > 0:
                # start_t = time.time()
                raw_score_maxima_loc_sorted_nms = nms(raw_score_maxima_loc_sorted, self._nms_dist_sqr)
                # end_t = time.time()
                # nms_time_taken = end_t - start_t
            else:
                raw_score_maxima_loc_sorted_nms = raw_score_maxima_loc_sorted

            raw_score_maxima_sorted = score_[
                raw_score_maxima_loc_sorted_nms[0], raw_score_maxima_loc_sorted_nms[1]]

            if self._params.feature_type == 2:
                n_maxima = len(raw_score_maxima_sorted)
                if n_maxima < self.n_features:
                    _features[:n_maxima] = raw_score_maxima_sorted
                else:
                    _features = raw_score_maxima_sorted[:self.n_features]

            if not self._heuristics:
                return _features, _status, _conf

            if self._params.non_best_scores > 1:
                second_best_score = np.mean(raw_score_maxima_sorted[1:self._params.non_best_scores + 1])
            else:
                second_best_score = raw_score_maxima_sorted[1]

        assert best_score >= second_best_score, \
            f"second_best_score: {second_best_score} cannot be > best_score: {best_score}"

        second_best_score_ratio = second_best_score / best_score

        if second_best_score_ratio > 0:
            _conf = 1 - math.exp(-1.0 / second_best_score_ratio)
        else:
            _conf = 1

        if _conf > self._params.conf_thresh:
            _status = 1
        else:
            _status = 0

        if self._params.vis:
            # print('score_maxima: {}'.format(score_maxima))
            # print('maxima_time_taken: {}'.format(maxima_time_taken))
            # print('nms_time_taken: {}'.format(nms_time_taken))

            k = cv2.waitKey(1 - self._pause)
            if k == 32:
                self._pause = 1 - self._pause

        # if self.run.visualization:
        #     show_frame(image_, bbox[0, :], 1)

        if not np.any(_features):
            print('all zeros features')

        return _features, _status, _conf
