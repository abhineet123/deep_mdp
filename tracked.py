import numpy as np
import os
import time
import cv2
import functools
from datetime import datetime
from contextlib import contextmanager

import paramparse

from objects import Annotations
from templates import Templates
from policy_base import PolicyBase
from models.model_base import ModelBase
from models.dummy import Oracle

from trackers.tracker_base import TrackerBase
from trackers.siamese import Siamese
from trackers.lk import LK
from trackers.pyt import PYT

from utilities import MDPStates, TrackingStatus, draw_box, resize_ar, \
    stack_images, col_bgr, copy_rgb, compute_overlaps_multi, compute_overlap, annotate_and_show, CustomLogger


class Tracked(PolicyBase):
    """

    :type _model: ModelBase
    :type _templates: Templates
    :type _params: Tracked.Params
    :type _rgb_input: int | bool
    :type _logger: logging.RootLogger | CustomLogger
    :type _target_id: int
    :type _feature_shape: tuple(int)
    :type _annotations: Annotations
    :type _ctm_tracker: TrackerBase
    :type _tracker: TrackerBase
    """

    class Params(PolicyBase.Params):
        """

        :ivar iou_det_state: 'threshold of the mean overlap of all templates with the best matching detection'
                 ' for the tracking to be considered successful; if the mean overlap is less than this, '
                 'then the object is considered to have been lost; '
                 'only matters if ignore_detection is 0'
        :ivar iou_det_box: 'minimum overlap (IOU) between the location of the anchoring template and the best '
                       'matching detection for the final object location to be computed as a weighted average of '
                       'the two; if the overlap is less than this, then the detection itself is used as this '
                       'location;
                       only matters if ignore_detection is 0'
        :ivar iou_pos: 'min IOU between tracked / predicted box and detection for positive label when training ',
        :ivar iou_gt: 'min IOU between tracked box and GT for positive GT label'
        :ivar weight_tracking: 'weight given to the anchoring template tracked location while computing the '
                   'weighted average of this and the best matching detection as the final object '
                   'location'
        :ivar weight_detection: 'weight given to the best matching detection while computing the weighted average '
                    'of this and the anchoring template tracked location as the final object location'
        :ivar exit_threshold: 'minimum overlap between an object location and the frame extents for the object to be '
                  'considered to be within the scene'
        :ivar tracker: {
            'LK': ('0', 'lk'),
            'Siamese': ('1', 'siam', 'siamese'),
            'PYT': ('1', 'pyt', 'pytracking'),
        }

        """

        def __init__(self):
            PolicyBase.Params.__init__(self)

            self.model = 'none'

            self.ignore_det = 0

            self.iou_det_state = 0.7
            self.iou_det_box = 0.5
            self.iou_pos = 0.5
            self.iou_gt = 0.2

            self.weight_tracking = 1
            self.weight_detection = 1

            self.track_heuristics = 1

            """
            minimum overlap with the frame for the object to be considered to be
            present in the scene"""
            self.exit_threshold = 0.95

            self.summarize_templates = 1

            """use custom tracker for continuous tracking to bypass templates module completely
            """
            self.tracker = ''
            self.lk = LK.Params()
            self.siamese = Siamese.Params()
            self.pyt = PYT.Params()

    def __init__(self, templates, params, rgb_input, external_model, logger, parent=None):
        """
        :type templates: Templates
        :type params: Tracked.Params
        :type rgb_input: int | bool
        :type logger: logging.RootLogger | CustomLogger
        :type parent: Tracked | None
        :rtype: None
        """
        PolicyBase.__init__(self, params, 'tracked', logger, parent)

        self._templates = templates
        self._tracker = self._templates.tracker  # type: TrackerBase
        self._params = params
        self._rgb_input = rgb_input
        self._logger = logger
        self._external_model = external_model
        self._parent = parent

        self._target_id = -1
        self._prev_id = -1

        if self._parent  is not None:
            ctm_tracker = self._parent ._ctm_tracker  # type: TrackerBase
            # self.frame_size = parent.frame_size
            # self.frame_box = parent.frame_box
            self._model = self._parent ._model
            self._external_model = self._parent ._external_model
            self._track_heuristics = self._parent ._track_heuristics
            self._n_features = self._parent ._n_features
            # self.tracker_type = parent.tracker_type
            # self.tracker_types = parent.tracker_types
            self._ctm_info = self._parent ._ctm_info
            self._ctm_name = self._parent ._ctm_name
        else:
            ctm_tracker = None
            # self.frame_size = (0, 0)
            # self.frame_box = np.zeros((1, 4))
            self._track_heuristics = self._params.track_heuristics

            tracker_type = str(self._params.tracker)
            if tracker_type:
                tracker_types = paramparse.obj_from_docs(self._params, 'tracker')
                try:
                    tracker_name = [k for k in tracker_types if tracker_type in tracker_types[k]][0]
                except IndexError:
                    raise AssertionError('Invalid tracker_type: {}'.format(tracker_type))
                tracker_func_dict = {
                    'LK': (LK, self._params.lk),
                    'Siamese': (Siamese, self._params.siamese),
                    'PYT': (PYT, self._params.pyt),
                }
                self._ctm_info = tracker_func_dict[tracker_name]

                self._logger.info(f'Using continuous tracking mode with {tracker_name} tracker')
                self._ctm_name = tracker_name
            else:
                self._ctm_info = None
                self._ctm_name = None
                if not self._params.summarize_templates:
                    self._logger.info('feature summarization is disabled')

            if self._params.ignore_det:
                self._logger.warning(f'detections are ignored in decision making')
            else:
                self._logger.info(
                    f'threshold_box: {self._params.iou_det_state} overlap_box: {self._params.iou_det_box}')

        if self._ctm_info is not None:
            ctm_type, ctm_params = self._ctm_info  # type: type(TrackerBase), TrackerBase.Params
            ctm_params.roi.enable = 0
            _params = dict(
                params=ctm_params,
                rgb_input=self._rgb_input,
                update_location=1,
                n_templates=1,
                parent=ctm_tracker,
                logger=self._logger,
                policy_name='tracked',
            )
            self._ctm_tracker = ctm_type(**_params)
            self._feature_shape = self._ctm_tracker.feature_shape
            self._n_features = np.prod(self._feature_shape)
        else:
            self._ctm_tracker = None

        if parent is None:
            """need feature_shape for model so must be done after creating tracker
            """
            if self._external_model:
                self._logger.info('using external model')
                self._features = self._n_features = None
            else:
                if self._params.model == 'none':
                    assert self._track_heuristics, "track_heuristics must be enabled" \
                                                   " with model free policy"
                    self._n_features = 2
                    self._feature_shape = (2,)
                else:
                    self._feature_shape = self._tracker.feature_shape
                    self._n_features = np.prod(self._feature_shape)

                    if not self._params.summarize_templates:
                        self._n_features *= self._templates.count
                        self._feature_shape = [self._templates.count, ] + list(self._feature_shape)

                self._create_model(self._feature_shape, self._logger)

            if not self._params.enable_stats:
                self._logger.warning('stats are disabled')

        self.max_iou_det_idx = None
        self.max_iou = None
        self.max_iou_det_scores = None
        self.mean_det_iou = None

        if not self._external_model:
            self._features = np.array((self._n_features, 1), dtype=np.float32)

        self.streak = 0
        self._frame_id = 0
        self._frame = None
        self._detections = None
        self._predicted_location = np.zeros((1, 4), dtype=np.float32)

        """tracked locations of all templates in the reference frame of the input image"""
        self.locations = np.zeros((self._templates.count, 4))
        # self.locations_mat = np.zeros((self.templates.count, 4))
        self.location = np.zeros((1, 4), dtype=np.float32)
        self.transform = np.zeros((1, 4), dtype=np.float32)
        self.state = MDPStates.inactive
        self.state_str = MDPStates.to_str[self.state]
        self.status = None
        self.status_str = None

        """holds the ROI around the predicted target location into which the templates are tracked"""
        # self.roi = np.empty((1,) + self._templates.roi_shape, dtype=np.uint8)

        self._tracking_result = None
        self._learn_features = None
        self._learn_labels = None

        self._n_total_assoc = 0
        self._n_correct_assoc = self._correct_assoc = 0
        self._n_incorrect_assoc = self._incorrect_assoc = 0

        self._pause = 0

        self._annotations = None
        self._curr_ann_idx = None
        self._ann_status = None

        self._gt_label = None

        if not self._track_heuristics:
            if not self._external_model and self._model is None:
                self._logger.warning(f'no model so enabling track_heuristics')
                self._track_heuristics = 1
            # elif self._is_oracle:
            #     self._logger.warning(f'oracle model does not support score prediction so enabling track_heuristics')
            #     self._track_heuristics = 1

    def _get_label(self):
        """

        :param np.ndarray  location:
        :return:
        """

        location = self.location

        self._gt_label = np.array((-1,), dtype=np.int32)
        _gt_location = None

        invalid_pred = np.any(np.isnan(location))

        if self._curr_ann_idx is not None:
            _gt_location = self._annotations.data[self._curr_ann_idx[0], 2:6]

            # is_occluded = self._annotations.data[self._curr_ann_idx[0], 11]
            # if not is_occluded:

            if not invalid_pred:
                # pred_cx, pred_cy = self._predicted_location[:2] + self._predicted_location[2:] / 2.0
                # gt_cx, gt_cy = self._gt_location[:2] + self._gt_location[2:] / 2.0
                #
                # gt_dist = np.sqrt((gt_cx-pred_cx)**2 + (gt_cy-pred_cy)**2)
                #
                # gt_size = (self._gt_location[3] + self._gt_location[4]) / 2.0
                # if gt_dist < gt_size*self._params.gt_dist_factor:
                #     self._gt_label = 1

                location_iou = np.empty((1,))
                """
                iou between GT and predicted locations
                """
                compute_overlap(location_iou, None, None, _gt_location.reshape((1, 4)), location)

                if location_iou > self._params.iou_gt:
                    """
                    tracked location corresponds to GT
                    """
                    self._gt_label[0] = 1

        if self._params.vis:
        # if 1:
            frame_disp = copy_rgb(self._frame)

            # if np.any(np.isnan(self.location)):
            #     self._logger.warning('invalid location found: {}'.format(self.location))
            # else:
            #     draw_box(frame_disp, self.location, color='red')

            if _gt_location is not None:
                draw_box(frame_disp, _gt_location, color='blue')

            if self._gt_label[0] == 1:
                col = 'green'
            else:
                col = 'red'

            if invalid_pred:
                self._logger.warning('invalid location found: {}'.format(location))
            else:
                draw_box(frame_disp, location, color=col)

            annotate_and_show('frame_disp', frame_disp, f'frame {self._frame_id} target {self._target_id}')

            # print()

    def initialize(self, target_id, frame_id, frame, location,
                   annotations=None, curr_ann_idx=None, ann_status=None):
        """

        :param int target_id:
        :type frame_id: int
        :type frame: np.ndarray
        :param np.ndarray location:
        :type annotations: Annotations | None
        :type curr_ann_idx: tuple | None
        :return:
        """

        self._target_id = target_id
        self._frame_id = frame_id
        self._frame = frame

        self._annotations = annotations
        self._curr_ann_idx = curr_ann_idx
        self._ann_status = ann_status

        if annotations is not None:
            self._set_stats(ann_status)

        if self._model is not None:
            if not self._external_model:
                self._model.set_id(self._target_id)

            if not self._model.is_trained and not self._track_heuristics:
                if self._parent is None:
                    self._logger.warning(f'model is not trained so enabling track_heuristics')
                self._track_heuristics = 1

        self.location[:] = location

        if self._ctm_tracker is not None:
            self._ctm_tracker.initialize(frame, location.reshape((1, 4)))

            if self._feature_shape is None:
                self._feature_shape = self._ctm_tracker.feature_shape
                self._n_features = np.prod(self._feature_shape)

    @contextmanager
    def profile(self, _id):
        if self._params.profile:
            start_t = time.time()
            yield None
            end_t = time.time()
            _times = end_t - start_t
            _fps = 1.0 / _times if _times > 0 else 0
            self._logger.info('{} fps: {:.3f}'.format(_id, _fps))
        else:
            yield None

    def reinitialize(self, frame, frame_id, detections, location):
        """
        reinitialize tracker state in continuous tracking mode for transition from lost to racked
        to account for frame discontinuity while target was lost

        :param np.ndarray frame:
        :param int frame_id:
        :param np.ndarray detections:
        :param np.ndarray location:
        :return:
        """

        if self._ctm_tracker is None:
            return

        # if frame_id <= self.frame_id + 1:
        #     """no missing frames so resetting not needed"""
        #     return

        """tracker might alternately be completely initialized but letting set_region approximate that for speed"""
        self._ctm_tracker.set_region(0, frame, location.squeeze())

    def _weird_ctm_pseudo_heuristics(self, features, mean_det_iou):
        """if model is not trained, _track_heuristics must be on so use TrackingStatus from there as
        proxy for label from model

        if model is trained but _track_heuristics is still on and says that tracking was successful,
        we confirm this prediction with the model before deciding;
        not sure if this makes sense or helps in any way
        """
        if self._model.is_trained and self._track_heuristics:
            labels, _ = self._model.predict(features, vis=0, gt_labels=self._gt_label)
            if labels[0] == 1:
                self.status = TrackingStatus.success
            else:
                self.status = TrackingStatus.failure

        if self.status == TrackingStatus.success and \
                (self._params.ignore_det or mean_det_iou > self._params.iou_det_state):
            self.state = MDPStates.tracked
        else:
            self.state = MDPStates.lost

    def _add_test_samples(self, features, labels, pred_labels, is_synthetic):
        PolicyBase._add_test_samples(self, features, labels, pred_labels, is_synthetic=is_synthetic)
        # print()

    def _get_scores(self, track_res, features, gt_label=None):

        labels = np.zeros((features.shape[0],), dtype=np.int32)

        if self._track_heuristics:
            status = track_res.get_status(0, track_id=0)
            if status == TrackingStatus.success:
                labels[0] = 1
            else:
                labels[0] = -1

            success_ids = track_res.get_success_ids(0)
            tracking_scores = track_res.get_scores(0)
        else:
            labels, probabilities = self._model.predict(features, gt_label, vis=0)

            success_ids = np.argwhere(labels == 1)
            if labels[0] == 1:
                status = TrackingStatus.success
            else:
                status = TrackingStatus.failure
            tracking_scores = probabilities[:, 0]

        if self._params.save_mode and gt_label is not None:
            self._add_test_samples(features, gt_label, labels, is_synthetic=0)

        return tracking_scores, success_ids, status

    def update_ctm(self, frame, frame_id, detections, predicted_location):
        assert self._ctm_tracker is not None, "CTM cannot be used when tracker is None"

        frame_exp = np.expand_dims(frame, axis=0)
        with self.profile('tracking'):
            track_res = self._ctm_tracker.track(frame=frame_exp,
                                                frame_id=frame_id,
                                                locations=None,
                                                n_objs=1,
                                                heuristics=self._track_heuristics
                                                )  # type: TrackerBase.Result
            if track_res is None:
                self._logger.warning('Tracker indicates that object has exited the scene')
                self.state = MDPStates.inactive
                self.state_str = MDPStates.to_str[self.state]

                return None

        # self._logger.debug(f'{self._target_id} :: {tracking_result._conf}')

        self.location[:] = track_res.locations.squeeze()
        self._tracking_result = track_res

        if self._annotations is not None:
            self._get_label()

        self.max_iou = np.zeros((1, 1), dtype=np.float32)

        if detections.shape[0] == 0:
            """no detections"""
            self.max_iou = 0
            self.max_iou_det_idx = None
            # self.mean_det_iou = 0
            self.max_iou_det_scores = -1
        elif detections.shape[0] == 1:
            """single detection"""
            iou = np.empty((1, 1))
            compute_overlap(iou, None, None, detections[0, 2:6].reshape((1, 4)),
                            self.location, self._logger)
            self.max_iou_det_idx = 0
            self.max_iou = iou
            # self.mean_det_iou = iou.item()
            self.max_iou_det_scores = detections[0, 6]
        else:
            """get detection with maximum overlap with the tracked location"""
            iou = np.empty((detections.shape[0], 1))
            compute_overlaps_multi(iou, None, None, detections[:, 2:6],
                                   self.location, self._logger)
            self.max_iou_det_idx = np.argmax(iou, axis=0).item()
            self.max_iou = iou[self.max_iou_det_idx, 0]
            self.max_iou_det_scores = detections[self.max_iou_det_idx, 6]

        max_det_iou = self.max_iou
        # mean_det_iou = self.mean_det_iou
        max_iou_det_id = self.max_iou_det_idx

        if not self._params.ignore_det and max_det_iou > self._params.iou_det_box:
            """adjust the location to be a weighted average of the maximally overlapping detection 
            and the tracked location if the former has high overlap
            """
            max_iou_det = detections[max_iou_det_id, 2:6].reshape((1, 4))

            """weighted average of locations of the main template and its maximally overlapping detection"""
            self.location[:] = np.average(np.concatenate((max_iou_det, self.location), axis=0), axis=0,
                                          weights=(self._params.weight_detection, self._params.weight_tracking))

            self._ctm_tracker.set_region(0, frame, self.location.squeeze())

        if self._params.vis:
            self._vis(frame, predicted_location, self.location, detections, max_iou_det_id, show_dets=1)

        """make decision about the state"""
        if self._model is None:
            """decide using annoying heuristics
            """
            self.status = track_res.get_status(0, track_id=0)
            tracking_scores = track_res.get_scores(0)

            self._features[0] = self.status

            self._features[1] = max_det_iou
            if self._features[0] == TrackingStatus.success and \
                    (self._params.ignore_det or self._features[1] > self._params.iou_det_state):
                self.state = MDPStates.tracked
            else:
                self.state = MDPStates.lost
            self.state_str = MDPStates.to_str[self.state]
        else:
            """decide using model
            """
            features = track_res.features[0, ...]

            tracking_scores, success_ids, status = self._get_scores(track_res, features, self._gt_label)

            self.status = status
            self._features = features

            if self.status == TrackingStatus.failure:
                self.state = MDPStates.lost
            else:
                self._weird_ctm_pseudo_heuristics(features, max_det_iou)

            self.state_str = MDPStates.to_str[self.state]

        self.status_str = TrackingStatus.to_str[self.status]

        self.__update_stats()

        return tracking_scores

    def update(self, frame, frame_id, detections, predicted_location,
               curr_ann_idx=None):
        """
        :type frame: np.ndarray
        :type frame_id: int
        :type detections: np.ndarray
        :type predicted_location: np.ndarray
        :type curr_ann_idx: tuple
        :rtype: np.ndarray
        """

        annotations = self._annotations

        self.streak += 1
        self._frame_id = frame_id
        self._frame = frame
        self._detections = np.copy(detections)
        self._curr_ann_idx = curr_ann_idx
        self._predicted_location[:] = predicted_location

        # gt_location = None

        if annotations is not None:
            if self._curr_ann_idx is not None:
                gt_location = annotations.data[self._curr_ann_idx[0], 2:6]

        if self._oracle_type == Oracle.Types.absolute:
            assert annotations is not None, "annotations must be available for absolute oracle"

            if self._curr_ann_idx is not None:
                self.location[:] = annotations.data[self._curr_ann_idx[0], 2:6]
                self.state = MDPStates.tracked
            else:
                self.state = MDPStates.lost

            self.state_str = MDPStates.to_str[self.state]
            self.__update_stats()
            tracking_scores = np.ones((self._templates.count,), dtype=np.float32)
            return tracking_scores

        """check if object has left the scene"""
        """temporarily disabled to ensure consistency with the annoying ad-hoc heuristics in the original code"""
        # ioa_1 = np.empty((1,))
        # computeOverlap(None, ioa_1, None, self.location, self.frame_box)
        # if ioa_1[0] < self.params.exit_threshold:
        #     self.state = MDPStates.inactive
        #     return

        if self._ctm_tracker is not None:
            """continuous tracking mode
            """
            return self.update_ctm(frame, frame_id, detections, predicted_location)

        """track all templates into the predicted location in the current frame"""
        track_res = self._tracker.track(frame, frame_id, predicted_location, 1,
                                        heuristics=self._track_heuristics)  # type: TrackerBase.Result

        if track_res is None:
            # self._logger.warning('\nTracker indicates that target {} has exited the scene'.format(self._target_id))
            self.state = MDPStates.inactive
            self.state_str = MDPStates.to_str[self.state]

            return None

        if track_res.locations is None:
            self._logger.warning('Predicted location tracking failed: {}\n'
                                 'Assuming failed prediction'.format(predicted_location))
            self.state = MDPStates.lost
            self.state_str = MDPStates.to_str[self.state]

            return None

        self.locations[:] = track_res.locations.squeeze()

        if np.any(np.isnan(self.locations)):
            self._logger.warning('invalid locations found: {}'.format(self.locations))

        self._templates.apply_heuristics(frame_id, detections, self.locations, track_res,
                                         check_ratio=True, track_id=0)

        """location and similarity are updated irrespective of tracking success 
        seems dodgy at best
        """
        anchor_location = self.locations[self._templates.anchor_id, :].reshape((1, 4))
        if np.isfinite(anchor_location).all() and not np.any(np.isnan(anchor_location)):
            self.location[:] = anchor_location

        if annotations is not None:
            self._get_label()

        max_det_iou = self._templates.max_iou[self._templates.anchor_id]
        # mean_det_iou = np.mean(self._templates.anchor_iou)

        max_iou_det_id = self._templates.max_iou_det_idx[self._templates.anchor_id]

        # self._assoc_det_id = max_iou_det_id

        if not self._params.ignore_det and max_det_iou > self._params.iou_det_box:
            """adjust the location to be a weighted average of the maximally overlapping detection 
            and the tracked location if the former has high overlap
            """
            # self.logger.critical('curr_det_data: %(1)s', {'1': curr_det_data})
            # self.logger.critical('self.templates.indices: %(1)s', {'1': self.templates.indices})
            # self.logger.critical('self.templates.anchor_id: %(1)s', {'1': self.templates.anchor_id})

            max_iou_det = detections[max_iou_det_id, 2:6].reshape((1, 4))

            """weighted average of locations of the main template and its maximally overlapping detection"""
            self.location[:] = np.average(np.concatenate((max_iou_det, self.location), axis=0), axis=0,
                                          weights=(self._params.weight_detection, self._params.weight_tracking))
        if self._params.vis:
            # if 1:
            self._vis(frame, predicted_location, self.location, detections, max_iou_det_id, show_dets=0)

        """make decision about the state"""
        if self._model is None:
            """decide using annoying heuristics
            """
            tracking_scores = track_res.get_scores(0)

            self._features[0] = track_res.get_status(self._templates.anchor_id, track_id=0)
            status_str = TrackingStatus.to_str[self._features[0]]

            self._features[1] = max_det_iou
            if self._features[0] == TrackingStatus.success and \
                    (self._params.ignore_det or self._features[1] > self._params.iou_det_state):
                self.state = MDPStates.tracked
            else:
                self.state = MDPStates.lost
            self.state_str = MDPStates.to_str[self.state]
        else:
            """decide using model
            """
            all_features = track_res.features[0, ...]

            if self._params.summarize_templates:
                """summarize_templates 
                """
                tracking_scores, success_ids, status = self._get_scores(track_res, all_features,
                                                                        gt_label=self._gt_label)

                if success_ids.size == 0:
                    """all templates failed to get tracked
                    """
                    self.state = MDPStates.lost
                    """anchor template features as proxy summary features (only for training)
                    """
                    self._features = all_features[self._templates.anchor_id, :]
                else:
                    """summarize features of successful templates to make final decision
                    """
                    self._features = track_res.get_summarized_features(success_ids, 0)

                    _pred_label = np.ones((1,))
                    if self._model.is_trained:
                        _pred_label, _ = self._model.predict(self._features.squeeze(),
                                                             gt_labels=self._gt_label, vis=0)
                    if self._params.save_mode:
                        self._add_test_samples(self._features, self._gt_label, _pred_label, is_synthetic=0)

                    if _pred_label[0] == 1 and (self._params.ignore_det or max_det_iou > self._params.iou_det_state):
                        self.state = MDPStates.tracked
                    else:
                        self.state = MDPStates.lost
            else:
                """no summarize_templates 
                """
                self._features = all_features

                if self._track_heuristics:
                    assert not self._model.is_trained, \
                        "tracking heuristics cannot be used with trained model and no summarized_features"

                    success_ids = track_res.get_success_ids(0)
                    tracking_scores = track_res.get_scores(0)

                    if success_ids.size == 0:
                        self.state = MDPStates.lost
                    else:
                        _label = 1

                        if self._params.ignore_det or max_det_iou > self._params.iou_det_state:
                            self.state = MDPStates.tracked
                        else:
                            self.state = MDPStates.lost
                else:
                    """no track_heuristics 
                    """

                    _pred_label, _probabilities = self._model.predict(self._features.squeeze(), vis=self._params.vis,
                                                                      gt_labels=self._gt_label
                                                                      )

                    if self._params.save_mode:
                        self._add_test_samples(self._features, self._gt_label, _pred_label, is_synthetic=0)

                    tracking_scores = _probabilities[:, 0]
                    if _pred_label[0] == 1 and (self._params.ignore_det or max_det_iou > self._params.iou_det_state):
                        self.state = MDPStates.tracked
                    else:
                        self.state = MDPStates.lost

            self.state_str = MDPStates.to_str[self.state]

        # if self._params.pause_for_debug:
        #     self._logger.debug('paused')

        state_str = MDPStates.to_str[self.state]

        self.__update_stats()

        return tracking_scores

    def __update_stats(self):
        if not self._params.enable_stats or self._gt_label is None:
            return

        label = np.zeros_like(self._gt_label)

        if self.state == MDPStates.tracked:
            decision = 'positive'
            label[0] = 1
            if self._gt_label[0] == 1:
                correctness = 'correct'
            else:
                correctness = 'incorrect'
        else:
            label[0] = -1
            decision = 'negative'
            if self._gt_label[0] == 1:
                correctness = 'incorrect'
            else:
                correctness = 'correct'

        self._update_stats(correctness, decision)

        if self._writer is not None and self._model is not None and not self._external_model:
            self._model.add_tb_image(self._writer, self._features, label, self._gt_label,
                                     iteration=None, stats=self._cmb_stats, batch_size=1,
                                     tb_vis_path=None, epoch=None, title='tracked')


    def train_async(self, frame, frame_id, detections, predicted_location):
        """
        asynchronous training

        :param np.ndarray frame:
        :param int frame_id:
        :param np.ndarray detections:
        :param np.ndarray predicted_location:
        :param Annotations annotations:
        :param np.ndarray ann_idx: index of the annotation belonging to the current trajectory
        :return: None
        """
        annotations = self._annotations
        assert annotations is not None, "annotations must be provided to train_async"

        if self._model is None:
            self._logger.warning('No model to train')
            return

        if self._external_model:
            self._logger.warning('Skipping training external model')
            return True

        if self._curr_ann_idx[0] is None:
            self._logger.debug('"no annotation for training in frame {} for target {}'.format(
                frame_id, self._target_id))
            return

        if annotations.data[self._curr_ann_idx[0], 11]:
            """only training on clearly visible annotations for now
            """
            if self._params.verbose:
                self._logger.debug('annotation not clearly visible in frame {} for target {}'.format(
                    frame_id, self._target_id))
            return

        if self._ctm_tracker is not None:
            frame_exp = np.expand_dims(frame, axis=0)
            with self.profile('tracking'):
                track_res = self._ctm_tracker.track(frame=frame_exp,
                                                    frame_id=frame_id,
                                                    locations=None,
                                                    n_objs=1,
                                                    heuristics=self._track_heuristics
                                                    )

            tracked_location = track_res.locations.squeeze()
            _learn_features = track_res.features[0, ...].squeeze()
        else:
            """track all templates into the predicted location in the current frame"""
            track_res = self._tracker.track(
                frame, frame_id, predicted_location, 1,
                heuristics=self._track_heuristics)  # type: TrackerBase.Result

            self._templates.apply_heuristics(frame_id, detections, track_res.locations, track_res,
                                             check_ratio=True, track_id=0)

            all_features = track_res.features[0, ...]

            if self._params.summarize_templates:
                if self._track_heuristics:
                    success_ids = track_res.get_success_ids(0)
                else:
                    all_labels, all_probabilities = self._model.predict(all_features.squeeze(),
                                                                        vis=0, gt_labels=None)
                    success_ids = np.argwhere(all_labels == 1)

                if success_ids.size == 0:
                    """anchor template features as proxy summary features for training
                    """
                    _learn_features = all_features[self._templates.anchor_id, :]
                else:
                    _learn_features = track_res.get_summarized_features(success_ids, 0)
            else:
                _learn_features = all_features

            tracked_location = predicted_location

        """iou between GT and predicted location
        """
        tracked_iou = np.empty((1,))
        compute_overlap(tracked_iou, None, None, annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4)),
                        tracked_location.reshape((1, 4)))

        if tracked_iou[0] > self._params.iou_pos:
            """predicted_location corresponds to GT
            """
            label = 1
        else:
            """predicted_location doesn't corresponds to GT - should have transitioned to lost
            """
            label = -1

        _learn_labels = np.array((label,))
        self._train(_learn_features, _learn_labels)

    def train(self):
        """
        :type annotations: Annotations
        :type ann_idx: np.ndarray
        :rtype: (bool, int)
        """
        assert self._annotations is not None, "annotations must be provided to train"

        if self._model is None:
            self._logger.warning('No model to train')
            return True

        if self._external_model:
            self._logger.warning('Skipping training external model')
            return True

        annotations = self._annotations

        if self._curr_ann_idx is None:
            """no annotations in this frame"""
            return

        if annotations.data[self._curr_ann_idx[0], 11]:
            """only training on clearly visible annotations for now
            """
            return

        """iou between GT and predicted location
        """
        predicted_iou = np.empty((1,))
        compute_overlap(predicted_iou, None, None, annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4)),
                        self._predicted_location.reshape((1, 4)))

        _n_correct_assoc = _n_incorrect_assoc = 0

        if predicted_iou[0] > self._params.iou_pos:
            """predicted_location corresponds to GT
            """
            label = 1
            if self.state == MDPStates.tracked:
                """made correct decision by remaining in tracked state
                """
                reward = 1
                _n_correct_assoc = 1
            else:
                """made incorrect decision by transitioning to lost state
                """
                reward = -1
                _n_incorrect_assoc = 1
        else:
            """predicted_location doesn't corresponds to GT - should have transitioned to lost
            """
            label = -1
            if self.state == MDPStates.tracked:
                """made incorrect decision by remaining in tracked state
                """
                reward = -1
                _n_incorrect_assoc = 1
            else:
                """made correct decision by transitioning to lost state
                """
                reward = -1
                _n_correct_assoc = 1

        self._n_correct_assoc += _n_correct_assoc
        self._n_incorrect_assoc += _n_incorrect_assoc
        self._n_total_assoc += 1

        if reward == -1 or self._params.always_train:
            """
            only train when we made an incorrect decision
            """
            self._learn_features = self._features.reshape((1, -1))
            self._learn_labels = np.array((label,))
            self._train(self._learn_features, self._learn_labels)

        if self._params.verbose == 2 or self._target_id != self._prev_id:
            self._prev_id = self._target_id
            self._correct_assoc = float(self._n_correct_assoc) / float(self._n_total_assoc) * 100
            self._incorrect_assoc = float(self._n_incorrect_assoc) / float(self._n_total_assoc) * 100
            self._logger.info('target {:d}: total: {:d}, correct: {:.2f}%, incorrect: {:.2f}%'.format(
                self._target_id, self._n_total_assoc, self._correct_assoc, self._incorrect_assoc))

        return reward

    def reset_streak(self):
        self.streak = 0

    def _vis(self, frame, predicted_location, location, detections, max_iou_det_id,
             show_dets=1, frame_label=''):
        """

        :param np.ndarray frame:
        :param np.ndarray annotations:
        :param np.ndarray detections:
        :param np.ndarray indices:
        :param np.ndarray labels:
        :param str frame_label:
        :return: np.ndarray
        """
        if len(frame.shape) == 2:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_vis = np.copy(frame)

        frames = [frame_vis, ]

        if np.any(np.isnan(location)):
            self._logger.warning('skipping drawing invalid location: {}'.format(location))
        else:
            draw_box(frame_vis, location, color='green', thickness=2)

        if self._ctm_tracker is None:
            if np.any(np.isnan(predicted_location)):
                self._logger.warning('skipping drawing invalid predicted_location: {}'.format(predicted_location))
            else:
                frame_vis_pred = np.copy(frame)
                draw_box(frame_vis_pred, predicted_location, color='purple', thickness=2)
                frames.append(frame_vis_pred)

        if max_iou_det_id is not None:
            frame_vis_det = np.copy(frame)
            draw_box(frame_vis_det, detections[max_iou_det_id, 2:6], color='red', thickness=2)
            frames.append(frame_vis_det)

        if frame_label:
            frame_vis_label = np.copy(frame)
            cv2.putText(frame_vis_label, frame_label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col_bgr['green'], 1, cv2.LINE_AA)
            frames.append(frame_vis_label)

        if self._ctm_tracker is None and self._tracker._params.roi.enable:
            templates = self._tracker.get_stacked_roi(
                self._templates.anchor_id, self._templates.frame_ids, self._templates.frames, self._templates.locations,
                show=0, grid_size=[-1, 3])[0]

            # if predicted_roi is not None:
            #     predicted_roi = predicted_roi.squeeze()
            #     draw_box(predicted_roi, self._tracker._std_box, color='black', thickness=2)
            #     templates = np.concatenate((templates, predicted_roi), axis=1)

            templates = resize_ar(templates, width=frame_vis.shape[1])
            if len(templates.shape) == 2:
                templates = cv2.cvtColor(templates, cv2.COLOR_GRAY2BGR)
            stacked_img = stack_images((templates, frame_vis), grid_size=[2, 1])
            stacked_img = resize_ar(stacked_img, height=2000)
        else:
            # templates = np.copy(frame)
            stacked_img = stack_images(frames, grid_size=None)

        img_list = [stacked_img, ]

        if show_dets:
            frame_vis_det = np.copy(frame)

            for i in range(detections.shape[0]):
                if max_iou_det_id is not None and i == max_iou_det_id:
                    color = 'green'
                else:
                    color = 'red'

                draw_box(frame_vis_det, detections[i, 2:6],
                         _id='{}'.format(i),
                         color=color, thickness=2)
            img_list.append(frame_vis_det)

        self._pause = annotate_and_show('tracked_vis', img_list,
                                        f'frame {self._frame_id} target {self._target_id}\n'
                                        f'purple: predicted, green: location, red: max_iou_det',
                                        pause=self._pause, max_width=1800, max_height=2000)

        return stacked_img

    def get_model(self):
        return self._model

    def set_model(self, model):
        """

        :param ModelBase model:
        :param int n_features:
        :param tuple feature_shape:
        :return:
        """
        self._logger.info(f'Setting external model of type: {model.name} with feature_shape: {model.feature_shape}')
        self._external_model = 1

        self._model = model
        self._n_features = model.n_features
        self._features = np.array((self._n_features, 1), dtype=np.float32)
        self._feature_shape = model.feature_shape

    def load(self, load_dir):
        """
        :type load_dir: str
        :rtype: None
        """

        if self._external_model:
            self._logger.warning('skipping loading external model')
            return True

        if self._model is None:
            self._logger.warning('No model to load')
            return True

        if not self._model.load(load_dir):
            raise IOError('Failed to load model from {}'.format(load_dir))

        return True

    def save(self, save_dir, summary_dir):
        """
        :type save_dir: str
        :type summary_dir: str
        :rtype: None
        """
        if self._model is not None and not self._external_model:
            os.makedirs(save_dir, exist_ok=True)
            self._model.save(save_dir)

        os.makedirs(summary_dir, exist_ok=True)

        stats_file = '{}/tracked.txt'.format(summary_dir)
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

        with open(stats_file, 'a') as fid:
            fid.write('{}\t{}\t{:d}\t{:.3f}\t{:.3f}\n'.format(
                save_dir, time_stamp, self._n_total_assoc, self._correct_assoc,
                self._incorrect_assoc))
