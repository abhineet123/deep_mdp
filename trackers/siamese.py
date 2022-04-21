import functools
import math
import numpy as np
from scipy.signal import argrelmax
import cv2
from collections import OrderedDict

import paramparse

from trackers.siamfc_tf.siamfc_tf import SiamFCTF
from trackers.da_siam_rpn.da_siam_rpn import DaSiamRPN
from trackers.siamX.siamx import SiamX

from trackers.tracker_base import TrackerBase
from utilities import TrackingStatus, nms, get_neighborhood, draw_box, annotate_and_show, CustomLogger

class Siamese(TrackerBase):
    """
    :type _params: Siamese.Params
    :type _pre_cmpt_params: (tuple, int, tuple)
    :type _tracker: SiamFCTF | DaSiamRPN | SiamX
    """

    class Params(TrackerBase.Params):
        """
        :type variant: int


        :ivar variant: {
            'SiamFC': ('0', 'fc', 'siam_fc', 'siamfc'),
            'DaSiamRPN': ('1', 'da', 'da_rpn', 'da_siam_rpn'),
            'SiamX': ('2', 'x', 'siamx'),
        }
        :ivar stacked: 'use stacked implementation of LK - here all patches are stacked onto a single large image',
        :ivar nms_method: '0: no NMS - distance from center combined with <feat_ops> over concentric neighborhoods of '
                      'increasing radii around the maximum '
                      '1: slow accurate NMS '
                      '2: fast approximate NMS ',
        :ivar feature_type: '0: scores flattened '
                '1: maximum values in each row and column of scores concatenated '
                '2: top <n_features> maxima values in scores (with nms for resized scores)'
                ' with nms_method 2 ',
        :ivar feature_scale: 'factor by which to scale down the raw scores; only applicable if feature_type=0',
        :ivar n_features: 'only used if feature_type = 2',
        :ivar n_init_samples: 'Number of idealized training samples to use for initial training;'
                  'set to 0 to disable initial training',

        :ivar feat_ops: 'only for nms_method 0; one or more statistical operations over neighborhood score map values '
                    'specified as a string of integers with each representing one operation: '
                    '0: max, '
                    '1: mean, '
                    '2: median, '
                    '3: min, '
        ,
        :ivar nms_dist_ratio: 'fraction of the maximum dimension of the score map used as the distance threshold '
                  'while performing non-maximum suppression for feature and status extraction',
        :ivar normalize_scores: 'normalize score map so that the maximum becomes 1',
        :ivar vis: 'Enable diagnostic visualization',
        :ivar siam_fc: 'SiamFCParams',
        :ivar da_rpn: 'DaSiamRPN.Params',
        :ivar siamx: 'SiamX.Params',

        """

        def __init__(self):
            TrackerBase.Params.__init__(self)

            self.variant = 'siam_fc'

            self.stacked = 0

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

            self.max_frames_to_store = 5
            self.n_init_samples = 1

            self.siam_fc = SiamFCTF.Params()
            self.da_rpn = DaSiamRPN.Params()
            self.siamx = SiamX.Params()

            self.profile = 0
            self.verbose = 0
            self.vis = 0

    def __init__(self, **kwargs):
        """
        :type params: Siamese.Params
        :type update_location: int
        :type logger: logging.RootLogger
        :type parent: Siamese | None
        :rtype: None
        """

        TrackerBase.__init__(self, 'siamese', **kwargs)

        if self._parent is None:
            self._scaled_score_sz = next(self._spawn_ids_gen)
            self._nms_dist = next(self._spawn_ids_gen)
            self._nms_dist_sqr = next(self._spawn_ids_gen)
            self._feat_ops = next(self._spawn_ids_gen)
            self._feat_ops_str = next(self._spawn_ids_gen)
            self._n_neighborhoods = next(self._spawn_ids_gen)
            self._n_features = next(self._spawn_ids_gen)
            self._feat_cy = next(self._spawn_ids_gen)
            self._feat_cx = next(self._spawn_ids_gen)
            self._max_dist_sqr = next(self._spawn_ids_gen)
            self._pause = next(self._spawn_ids_gen)
            self._create_tracker = next(self._spawn_ids_gen)
            self._tracker = next(self._spawn_ids_gen)

            self._features = next(self._copy_ids_gen)
            self._status = next(self._copy_ids_gen)
            self._conf = next(self._copy_ids_gen)

            self._register()
        else:
            self._spawn()

        # self._create_object = functools.partial(Siamese, **kwargs)

        """needed for intellisense"""
        self._params = kwargs["params"]  # type: Siamese.Params

        if self._parent is None:
            siamese_variants = paramparse.obj_from_docs(self._params, 'variant')

            variant_name = [k for k in siamese_variants if self._params.variant in siamese_variants[k]]
            if not variant_name:
                raise IOError('Invalid variant: {}'.format(self._params.variant))
            variant_name = variant_name[0]

            self._logger.info(f'Using variant: {variant_name}')

            _logger = CustomLogger(self._logger, names=(variant_name.lower(),))

            if self._params.variant in siamese_variants['SiamFC']:
                self._create_tracker = functools.partial(SiamFCTF,
                                                         params=self._params.siam_fc,
                                                         logger=_logger)
            elif self._params.variant in siamese_variants['DaSiamRPN']:
                self._create_tracker = functools.partial(DaSiamRPN,
                                                         params=self._params.da_rpn,
                                                         logger=_logger)
            elif self._params.variant in siamese_variants['SiamX']:
                self._create_tracker = functools.partial(SiamX,
                                                         params=self._params.siamx,
                                                         logger=_logger)
            else:
                raise IOError('Invalid variant: {}'.format(self._params.variant))

        self._tracker = self._create_tracker(n_trackers=self._n_templates,
                                             update_location=self._update_location,
                                             parent=self._tracker)
        self._score_sz = int(self._tracker.score_sz)

        if self._parent is None:
            if not self._params.roi.enable:
                if self._update_location:
                    self._logger.warning('Disabling location updating as that is not supported in ROI-free mode')
                    self._update_location = 0
            elif self._update_location:
                self._logger.warning('Location updating is enabled')

            self._scaled_score_sz = None
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
                    self._n_features = self._scaled_score_sz * self._scaled_score_sz
                    self.feature_shape = (self._scaled_score_sz, self._scaled_score_sz)
                else:
                    self._n_features = self._score_sz * self._score_sz
                    self.feature_shape = (self._score_sz, self._score_sz)

            elif self._params.feature_type == 1:
                self._logger.info('Using maximum values in each row and column of score map as features')

                self._n_features = self._score_sz + self._score_sz
                self.feature_shape = (self._n_features,)
            elif self._params.feature_type == 2:
                assert self._params.nms_method != 1, "feature_type 2 is not supported with nms_method 1"

                if self._params.nms_method == 0:
                    self._n_features = 1 + self._n_neighborhoods * len(self._feat_ops)
                    self.feature_shape = (self._n_features,)
                    self._logger.info(
                        'Using distance from center combined with {} over neighborhood for total feature size {} '
                        ''.format(
                            self._feat_ops_str, self._n_features))
                else:
                    self._n_features = self._params.n_features
                    self.feature_shape = (self._n_features,)
                    self._logger.info(
                        'Using top {} maxima values in scores (without nms) as features'.format(self._n_features))
                    if self._nms_dist > 0:
                        self._logger.info(f'Using nms_dist: {self._nms_dist}')
            elif self._params.feature_type != -1:
                raise AssertionError(f'invalid feature_type: {self._params.feature_type}')

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
            self._logger.info('n_features: {}'.format(self._n_features))
            self._logger.info('_nms_dist_sqr: {}'.format(self._nms_dist_sqr))

            self._pause = 1

        self._locations = None
        self._features = None
        self._status = None
        self._conf = None
        self._templates = [None] * self._n_templates

        self.frame_ids = []
        self.frame_id = None
        self.track_res_by_fram = OrderedDict()

        self._update = self._initialize

    def _initialize_roi(self, template_id, roi):
        self._initialize(template_id, frame=roi, bbox=None)

    def _initialize(self, template_id, frame, bbox):
        """

        :param int template_id:
        :param frame:
        :param np.ndarray | None bbox:
        :return:
        """

        self._tracker.initialize(template_id, frame, bbox)

        self._templates[template_id] = (frame, bbox)

    def update_frame(self, frame, frame_id):
        if self._params.roi.enable:
            return

        self._track(frame, frame_id, locations=None)

    def set_region(self, template_id, frame, bbox):
        self._tracker.set_region(template_id, frame, bbox)

    def _track(self, frame, frame_id, locations):
        """
        ignoring locations for now and simply reusing _track_roi with full frame instead of ROIs

        :param frame:
        :param locations:
        :return:
        """

        assert not self._params.roi.enable, "_track cannot be called in ROI mode"

        if frame_id <= self.frame_id:
            """frame already tracked"""
            try:
                track_res = self.track_res_by_frame(frame_id)
            except KeyError:
                raise AssertionError(f'Invalid frame {frame_id} with latest frame_id {self.frame_id} and max_frames_to_store {self._params.max_frames_to_store}')
            return track_res

        assert frame_id == self.frame_id + 1, f"missing frame between frame_ids {self.frame_id} and {frame_id}"

        self.frame_ids.append(frame_id)
        self.frame_id = frame_id

        frames = np.expand_dims(frame, axis=0)
        track_res = self._track_roi(frames)

        if locations is not None:
            """replicate locations, conf and status for all locations though ideally there ought 
            to be some sort of IOU based matching to infer what input locations are viable but leaving 
            it to the caller for now"""

            n_patches = locations.shape[0]
            track_res.locations = np.tile(track_res.locations, (n_patches, 1, 1))
            if not self._patch_as_features:
                track_res.features = np.tile(track_res.features, (n_patches, 1, 1))
                if self._heuristics:
                    track_res.conf = np.tile(track_res.conf, (n_patches, 1))
                    track_res.status = np.tile(track_res.status, (n_patches, 1))

        self.track_res_by_frame[self.frame_id] = track_res

        if len(self.track_res_by_frame) > self._params.max_frames_to_store:
            """remove data for oldest frame"""
            self.track_res_by_frame.popitem(last=False)

        return track_res

    def _track_roi(self, patches):
        """

        :param np.ndarray patches:
        :return:
        :rtype Siamese.Result
        """

        vis = self._params.vis

        n_patches = patches.shape[0]

        self._locations = np.zeros((n_patches, self._n_templates, 4))
        self._conf = self._status = self._features = None

        if not self._patch_as_features:
            self._features = np.zeros((n_patches, self._n_templates, self._n_features), dtype=np.float32)

            if self._heuristics:
                self._conf = np.zeros((n_patches, self._n_templates), dtype=np.float32)
                self._status = np.zeros((n_patches, self._n_templates), dtype=np.int32)

        self._pause = 1

        scores = {}

        for patch_id in range(n_patches):
            patch_2 = np.squeeze(patches[patch_id, :, :])

            scores[patch_id] = []
            for template_id in range(self._n_templates):
                bbox, _score, _score_id = self._tracker.update(template_id, patch_2)
                self._locations[patch_id, template_id, :] = bbox

                scores[patch_id].append(_score)

                if self._patch_as_features:
                    continue

                _features, _flag, _conf = Siamese.features_from_score_map(self, _score, _score_id)

                if vis:
                    pt1 = (int(bbox[0]), int(bbox[1]))
                    pt2 = (int(bbox[0] + bbox[2]),
                           int(bbox[1] + bbox[3]))

                    frame_disp = np.copy(patch_2)
                    cv2.rectangle(frame_disp, pt1, pt2, (0, 0, 255), thickness=2)

                    self._logger.info(f'tracker{template_id}')
                    self._pause = annotate_and_show('siamese:features_from_score_map', [frame_disp, _score],
                                                    self._logger, pause=self._pause)

                    patch_2_disp = np.copy(patch_2)
                    template_disp = np.copy(self._templates[template_id][0])
                    template2_disp = np.copy(self._tracker.templates[template_id][0])
                    draw_box(template2_disp, self._tracker.templates[template_id][1], color='red')
                    feature_img = _features.reshape(self.feature_shape)
                    self._logger.info(f'target_id {self._target_id} patch_id {patch_id} template_id {template_id}')
                    self._pause = annotate_and_show('frame_disp',
                                                    [patch_2_disp, template_disp, template2_disp, feature_img],
                                                    self._logger, pause=self._pause, grid_size=(1, -1))

                if self._heuristics:
                    self._conf[patch_id, template_id] = _conf
                    if _flag:
                        self._status[patch_id, template_id] = TrackingStatus.success
                    else:
                        self._status[patch_id, template_id] = TrackingStatus.failure

                self._features[patch_id, template_id, :] = _features

        track_res = Siamese.Result(self._locations, self._features, self._status, self._conf, scores, self)

        return track_res

    @staticmethod
    def features_from_score_map(self, score_, score_id_):
        """

        :param Siamese self:
        :param score_:
        :param score_id_:
        :return:
        """

        _features = np.zeros((self._n_features,), dtype=np.float32)
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

        else:
            raw_score_maxima_loc = argrelmax(score_, order=5)

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
                if n_maxima < self._n_features:
                    _features[:n_maxima] = raw_score_maxima_sorted
                else:
                    _features = raw_score_maxima_sorted[:self._n_features]

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
            k = cv2.waitKey(1 - self._pause)
            if k == 32:
                self._pause = 1 - self._pause

        if not np.any(_features):
            print('all zeros features')

        return _features, _status, _conf

    def copy(self):
        """
        :rtype: dict
        """
        obj_dict = TrackerBase.copy(self)
        obj_dict['tracker'] = self._tracker.copy()
        return obj_dict

    def restore(self, obj_dict, deep_copy=False):
        """
        :type obj_dict: dict
        :type deep_copy: bool
        """
        TrackerBase.restore(self, obj_dict, deep_copy)
        self._tracker.restore(obj_dict['tracker'])

    """
    ************************************************************
    end of Siamese
    ************************************************************
    """

    class Result(TrackerBase.Result):
        """
        :type _tracker: Siamese
        """

        def __init__(self, locations, features, status, conf, scores, tracker):
            """

            :param locations:
            :param features:
            :param status:
            :param conf:
            :param scores:
            :param Siamese tracker:
            """
            TrackerBase.Result.__init__(self, locations, features, status, conf)

            self._tracker = tracker
            self._scores = scores
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

            score_id = np.argmax(mean_score)
            score_id = np.unravel_index(score_id, mean_score.shape)

            features = Siamese.features_from_score_map(self._tracker, mean_score, score_id)

            features = features[0]

            return features

        def set_status(self, valid_idx, invalid_idx, track_id=0):
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

        # def _get_worst_id(self, exclude_id=None, track_id=0):
        #     """
        #     Returns the ID of the worst tracked patch with an optional exclusion
        #
        #     :param exclude_id: optional ID of the patch to exclude
        #     :type exclude_id: int | None
        #     :type track_id: int
        #     :rtype: int
        #     """
        #
        #     # object with the minimum confidence is the worst one
        #     if exclude_id is None:
        #         return np.argmin(self._conf[track_id, :])
        #     else:
        #         _conf = np.copy(self._conf[track_id, :])
        #         _conf[exclude_id] = 1
        #         return np.argmin(_conf)
