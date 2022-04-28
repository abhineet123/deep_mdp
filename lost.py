import sys
import os
from datetime import datetime
from pprint import pformat
from tabulate import tabulate

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import pandas as pd

import cv2

from templates import Templates
from input import Annotations
from policy_base import PolicyBase
from models.model_base import ModelBase
from models.dummy import Oracle
from models.cnn import CNN

from trackers.tracker_base import TrackerBase

from utilities import MDPStates, compute_overlap, draw_box, col_bgr, resize_ar, stack_images, write_to_files, \
    annotate_and_show, CVConstants, AnnotationStatus, CustomLogger, TrackingStatus, write_df, \
    get_shifted_boxes, draw_boxes


class Lost(PolicyBase):
    """
    :type _params: Lost.Params
    :type _logger: logging.RootLogger | CustomLogger
    :type _templates: Templates
    :type _max_score: float
    :type _n_features: int
    :type score: float
    :type templates: Templates
    :type _features: np.ndarray
    :type _labels: np.ndarray
    :type _probabilities: np.ndarray
    :type _model: ModelBase
    :type streak: int
    :type _det_centers: np.ndarray
    :type _distances: np.ndarray
    :type _height_ratios: np.ndarray
    :type _indices: np.ndarray
    :type _model_heuristics_flags: np.ndarray
    :type _roi: np.ndarray
    :type _det_location: np.ndarray
    :type location: np.ndarray
    :type state: int
    :type score: float
    :type _frame_id: int
    :type _frame: np.ndarray
    :type assoc_det_id: int
    :type assoc_det_id: int
    :type _annotations: Annotations
    """

    class Params(PolicyBase.Params):
        """

        :ivar max_streak: 'maximum number of consecutive frames for which the target is allowed to remain in '
                 'the Lost state before it transitions to inactive',
        :ivar threshold_ratio: 'aspect ratio threshold in target association - this is the minimum ratio '
                   'between the '
                   'heights of the last known location and a candidate detection for the latter to be '
                   'considered a possible match',
        :ivar threshold_dist: ' distance threshold in target association, multiple of the width of target - '
                  'this is the maximum ratio of the Euclidean distance between the centers of the last '
                  'known target location and a candidate detection with the width of the former '
                  'for the latter to be considered a possible match',
        :ivar iou_det_box: 'minimum overlap (IOU) between the LK result and the best matching detection '
                       'for the final '
                       'object location to be computed as a weighted average of the two; if the overlap is '
                       'less than this, then the detection itself is used as this location;',
        :ivar iou_pos: 'minimum overlap (IOU) of the ground truth location with the best matching detection and '
                   'the computed object location for the corresponding feature vector to be regarded '
                   'as a positive training sample for learning',
        :ivar iou_neg: 'maximum overlap (IOU) of the ground truth location with the best matching detection or '
                   'the computed object location for the corresponding feature vector to be regarded '
                   'as a negative training sample for learning'
        :ivar iou_gt: 'min overlap between max IOU detection and GT for association to be considered doable ',
        :ivar heuristic_features: 'augment tracker/templates features with heuristics to '
                      'construct policy classifier features; '
                      'setting to 0 will use only tracker features'
        :ivar copy_while_training: 'original code did not update templates state matrices while getting '
                       'learning features for some unknown annoying reason; toggle this behavior'
        :ivar weight_tracking: 'weight given to the tracked location while computing the weighted average '
                   'of this and the best matching detection as the final object location '
                   'during association',
        :ivar weight_association: 'weight given to the best matching detection while computing '
                      'the weighted average '
                      'of this and the tracked location as the final object location during association',
        :ivar similarity_type_id: 'ID of the similarity type used to obtain similarity feature '
                      'from template patches - indexes into Utilities.CVConstants.similarity_types'
        :ivar summarize_templates: heuristically combine features from all templates to create summarized features
        representing all of them; if disabled, all template features are stacked together so the model
        must be template-aware

        :ivar syn_method:
        0: Generate synthetic sample directly from each tracked box that is actually added as a sample
        1: generate a synthetic sample from each detection and make it undergo the entire
        tracking/Feature extraction/classification pipeline before adding the corresponding tracked box as a sample

        :ivar gt_type: 0: consider IOU of each detection with GT box to decide on its GT labels;
        only max-IOU detection can possibly be positive
        1: consider IOU of tracking result with GT box to decide on each detection's GT label;
        multiple detections can be positive

        """

        def __init__(self):
            PolicyBase.Params.__init__(self)

            self.max_streak = 10

            self.threshold_ratio = 0.6
            self.threshold_dist = 1

            self.iou_det_box = 0.5
            self.iou_pos = 0.5
            self.iou_neg = 0.2
            self.iou_gt = 0.5

            self.ann_ioa_thresh = 0.0

            self.heuristic_features = 1
            self.model_heuristics = 1
            self.track_heuristics = 1

            self.similarity_type_id = 2

            self.weight_tracking = 1
            self.weight_association = 1

            self.syn_method = 0

            self.gt_type = 1

            self.vis_train = 0
            self.copy_while_training = 1
            self.summarize_templates = 1

    def __init__(self, templates, params, logger, parent=None):
        """
        :type templates: Templates
        :type params: Lost.Params
        :type logger: llogging.RootLogger | CustomLogger
        :type parent: Lost | None
        :rtype: None
        """
        PolicyBase.__init__(self, params, 'lost', logger, parent)

        self._templates = templates
        self._params = params
        self._logger = logger

        self._target_id = 0
        self._prev_id = -1

        self._tracker = self._templates.tracker  # type: TrackerBase

        self._feature_shape = self._tracker.feature_shape
        self._n_track_features = np.prod(self._feature_shape)

        self.assoc_probabilities = []

        self._track_heuristics = self._params.track_heuristics

        if self._params.heuristic_features:
            """
            append annoying home-grown lost heuristics to tracker features 
            """
            self._n_features = self._n_track_features + 6
        else:
            self._n_features = self._n_track_features

        if not self._params.summarize_templates:
            self._n_track_features *= self._templates.count
            self._n_features *= self._templates.count
            self._feature_shape = [self._templates.count, ] + list(self._feature_shape)

        if parent is not None:
            """make sure that the model is not trained online in any manner that is target dependent
            """
            self._model = parent._model
            # self._train_features = lost._train_features
            # self._train_labels = lost._train_labels
            self._n_train_samples = parent._n_train_samples
            self._max_score = parent._max_score
            self._pause = parent._pause

            self.train_stats = parent.train_stats
            self.train_pc_stats = parent.train_pc_stats

            # self._n_total_assoc = parent._n_total_assoc
            # self._n_correct_assoc = parent._n_correct_assoc
            # self._n_incorrect_assoc = parent._n_incorrect_assoc
            # self._n_unknown_assoc = parent._n_unknown_assoc
            # self._track_heuristics = parent._track_heuristics

        else:
            self.train_stats = pd.DataFrame(
                np.zeros((len(PolicyBase.Decision.types), len(AnnotationStatus.types)), dtype=np.int32),
                columns=AnnotationStatus.types,
                index=PolicyBase.Decision.types,
            )
            self.train_pc_stats = pd.DataFrame(
                np.zeros((len(PolicyBase.Decision.types) - 1, len(AnnotationStatus.types)), dtype=np.float32),
                columns=AnnotationStatus.types,
                index=PolicyBase.Decision.types[1:],
            )

            if not self._params.heuristic_features:
                self._logger.info('heuristic features are disabled')
            else:
                assert self._templates.ratio_heuristics, \
                    "templates ratio_heuristics must be on to use heuristic_features"

            if not self._params.model_heuristics:
                self._logger.info('model heuristics are disabled')
            if not self._track_heuristics:
                self._logger.info('track heuristics are disabled')

            if not self._params.summarize_templates:
                self._logger.info('feature summarization is disabled')

                assert not self._params.heuristic_features, \
                    "heuristic_features should not be on without summarized_features"
                assert not self._params.model_heuristics, \
                    "model_heuristics should not be on without summarized_features"

            self._create_model(self._feature_shape, self._logger)
            assert not (self._params.model == 'cnn' and self._params.heuristic_features), \
                "heuristic features cannot be used with CNN"

            if self._params.syn_samples:
                if self._params.syn_method == 0:
                    self._logger.info(
                        f'generating {self._params.syn_samples} synthetic samples directly from tracked boxes')
            elif self._params.syn_method == 1:
                self._logger.info(
                    f'generating {self._params.syn_samples} synthetic samples from detections')

            # self._train_features = np.zeros((self.n_init_samples, self.n_features), dtype=np.float32)
            # self._train_labels = np.zeros((self.n_init_samples,), dtype=np.float32)

            # self._train_features = None
            # self._train_labels = None

            self._n_train_samples = 0

            self._max_score = 1
            self._pause = 0

            if not self._params.enable_stats:
                self._logger.warning('stats are disabled')

            # self._n_total_assoc = 0
            # self._n_correct_assoc = self._correct_assoc = 0
            # self._n_incorrect_assoc = self._incorrect_assoc = 0
            # self._n_unknown_assoc = self._unknown_assoc = 0

            self._test_mode = 0

        self._similarity_type = CVConstants.similarity_types[self._params.similarity_type_id]

        self._features = np.zeros((1, self._n_features), dtype=np.float32)
        self._labels = np.zeros((1,), dtype=np.float32)

        self._learn_features = np.zeros((1, self._n_features), dtype=np.float32)
        self._learn_labels = np.zeros((1,), dtype=np.float32)

        self._probabilities = np.zeros((1, 2), dtype=np.float32)
        self.streak = 0

        self._assoc_success_ids = {}

        """for pre-filtering of associations"""
        self._predicted_center = None
        self._det_centers = None
        self._distances = None
        self._height_ratios = None
        self._indices = None
        self._model_heuristics_flags = None
        self._transform = None
        # self.stacked_roi = None
        """location of the currently processed detection"""
        self._det_location = np.zeros((1, 4))
        """ tracked locations of templates wrt the predicted target location"""
        self._track_locations = np.zeros((self._templates.count, 4), dtype=np.float32)
        """tracked locations of templates wrt the currently processed detection"""
        self._locations_multi = None

        """current estimated location of the target"""
        self.location = np.zeros((1, 4), dtype=np.float32)
        self.state = MDPStates.inactive
        self.score = 0

        """truncated features for debugging"""
        self._train_features_trunc = None
        self._features_trunc = None
        if self._params.heuristic_features:
            """similarity (e.g. NCC) between template patches and and the main detection / predicted location"""
            self._similarity = np.zeros((self._templates.count, 1), dtype=np.float32)
            """get rid of the similarity feature in correspondence check to avoid mismatches due to 
            insidious floating point errors
            """
            self._trunc_idx = np.delete(np.arange(self._n_features), self._n_track_features + 1)
        else:
            self._similarity = None
            self._trunc_idx = None

        self._frame_id = 0
        self._frame = None
        self._detections = None
        self._predicted_location = None

        """detection with which the target is associated"""
        self.assoc_det_id = -1
        """index of the same detection within the local valid detections"""
        self._local_assoc_det_id = -1
        """Keep track of whether associations are being done using annotations"""
        self._ann_assoc = 0

        self._gt_labels = None
        self._gt_label_absolute = None
        self._gt_location = None

        self._annotations = None
        # self._traj_id = None
        self._ann_status = None
        # self._traj_idx_by_frame = None
        self._curr_ann_idx = None

        self._train_stats = None
        self._train_pc_stats = None
        self._cmb_train_stats = None
        self._cmb_train_pc_stats = None


    def _get_labels(self, tracked_locations, track_res, indices):
        """

        :param np.ndarray | None ann_idx: global index into annotations of the currently tracked target
        in the current frame
        :param TrackerBase.Result track_res:
        :param np.ndarray indices:
        :return:
        """
        assert indices.size > 0, "No filtered detections to get GT labels for"

        if self._annotations is None:
            _gt_labels = [None, ] * indices.size
            return _gt_labels

        negative_type = 'none'

        _gt_labels = np.full((indices.size,), -1)
        if self._curr_ann_idx is None:

            """fp_background target - should always give negative prediction since there is no object to associate"""
            negative_type = 'fp_background'
        else:

            is_occluded = self._annotations.data[self._curr_ann_idx[0], 11]
            gt_location = self._annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4))

            if not is_occluded:

                if self._params.gt_type == 0:
                    """
                    Only the single detection with the maximum IOU with the GT  will have a positive label
                    """
                    max_det_iou_idx, max_det_iou = self._get_max_iou_det(indices)
                    self._gt_assoc_det_id = max_det_iou_idx
                    if max_det_iou > self._params.iou_gt:
                        """
                        max iou detection corresponds to GT
                        """
                        _gt_labels[max_det_iou_idx] = 1
                    else:
                        negative_type = 'max_det_iou {:.2f}'.format(max_det_iou)
                else:
                    """
                    Any detection whose tracking result has high IOU with GT will have a positive label
                    """
                    self.max_tracked_iou_list = []
                    self.tracked_iou_dict = {}
                    for det_id in range(indices.size):
                        _tracked_locations = tracked_locations[det_id, :, :].reshape(
                            (self._templates.count, 4))  # type: np.ndarray
                        tracked_iou = np.empty((_tracked_locations.shape[0], 1))
                        compute_overlap(tracked_iou, None, None, gt_location, _tracked_locations)

                        self.tracked_iou_dict[det_id] = tracked_iou

                        max_tracked_iou_idx = np.argmax(tracked_iou)
                        max_tracked_iou = tracked_iou[max_tracked_iou_idx].item()

                        self.max_tracked_iou_list.append(max_tracked_iou)

                        if max_tracked_iou > self._params.iou_gt:
                            """
                            max iou tracked result corresponds to GT
                            """
                            _gt_labels[det_id] = 1

                    overall_max_tracked_iou = np.amax(self.max_tracked_iou_list)
                    if overall_max_tracked_iou < self._params.iou_gt:
                        """
                        max iou tracked result corresponds to GT
                        """
                        negative_type = 'overall_max_tracked_iou {:.2f}'.format(overall_max_tracked_iou)
            else:
                negative_type = 'is_occluded'

        if self._params.vis:
            frame_disp = np.copy(self._frame)

            draw_box(frame_disp, self.location, color='red')
            if self._curr_ann_idx is not None:
                draw_box(frame_disp, self._annotations.data[self._curr_ann_idx[0], 2:6], color='green')

            all_concat_imgs = []
            for i, det_idx in enumerate(indices):
                gt_label = self._gt_labels[i]
                if gt_label == 1:
                    col = 'white'
                else:
                    col = 'black'
                draw_box(frame_disp, self._detections[det_idx, 2:6], color=col)
                draw_boxes(frame_disp, tracked_locations[i, :, :].squeeze(axis=0),
                           color=col, is_dotted=1)

                features = track_res.features[i, ...]

                vis_gen = CNN.vis_samples(features, None, self._gt_labels, self._feature_shape, batch_size=1)
                for batch_id, concat_imgs in vis_gen:
                    all_concat_imgs.append(concat_imgs)

            txt = f'frame {self._frame_id} target {self._target_id} '
            if negative_type != 'none':
                txt += f' negative_type: {negative_type}'
            annotate_and_show('_get_labels', frame_disp, txt)

            # if track_res.roi is not None:
            #     vis_image = track_res.roi.get_vis_image()
            #     roi_vis_image = resize_ar(vis_image, height=900)
            #     annotate_and_show('tracking_result roi_vis_image', roi_vis_image)

            all_concat_imgs = np.concatenate(all_concat_imgs, axis=0)

            all_concat_imgs = resize_ar(all_concat_imgs, height=1050, only_shrink=1)
            annotate_and_show('features and labels', all_concat_imgs, self._logger, pause=self._pause)

        return _gt_labels

    # def add_sequence(self, frame_size):
    #     """
    #     :rtype: None
    #     """
    #     # self._max_score = max_score
    #     return True

    def _get_synthetic_samples(self, frame, frame_id, bboxes, indices, n_bboxes, syn_type):

        if indices is None:
            indices = np.asarray(list(range(n_bboxes)))

        assert n_bboxes == indices.size, "indices and n_bboxes mismatch"

        bboxes = bboxes.reshape((-1, 4))

        annotations = self._annotations

        all_sampled_boxes = []
        sampled_source_dets = []
        for det_id in range(n_bboxes):
            bbox = bboxes[indices[det_id], :].squeeze()
            if annotations is not None and annotations.idx[frame_id] is not None:
                gt_boxes = annotations.data[annotations.idx[frame_id], :][:, 2:6]
            else:
                gt_boxes = None

            if syn_type == 0:
                """
                negative samples
                """
                # min_shift_ratio = 1 - self._params.syn_neg_iou
                min_shift_ratio = 0
                min_anchor_iou, max_anchor_iou = self._params.syn_neg_iou
                max_gt_iou = max_anchor_iou
                name = 'lost neg'
            else:
                min_shift_ratio = 0
                min_anchor_iou, max_anchor_iou = self._params.syn_pos_iou
                max_gt_iou = max_anchor_iou
                name = 'lost pos'

            sampled_boxes = get_shifted_boxes(bbox, frame, self._params.syn_samples,
                                              min_anchor_iou=min_anchor_iou,
                                              max_anchor_iou=max_anchor_iou,
                                              min_shift_ratio=min_shift_ratio,
                                              max_shift_ratio=1.0,

                                              gt_boxes=gt_boxes,
                                              max_gt_iou=max_gt_iou,

                                              name=name,
                                              sampled_boxes=all_sampled_boxes,
                                              # vis=1,
                                              vis=self._params.vis,

                                              )
            all_sampled_boxes += sampled_boxes
            sampled_source_dets += [det_id, ] * len(sampled_boxes)
        return all_sampled_boxes, sampled_source_dets

    def _add_test_samples(self, features, labels, pred_labels, is_synthetic):
        PolicyBase._add_test_samples(self, features, labels, pred_labels, is_synthetic=is_synthetic)
        # print()

    def _set_train_stats(self, ann_status):
        self._train_stats = self.train_stats[ann_status]
        self._train_pc_stats = self.train_pc_stats[ann_status]
        self._cmb_train_stats = self.train_stats['combined']
        self._cmb_train_pc_stats = self.train_pc_stats['combined']

    def initialize(self, target_id, frame_id, frame, location,
                   annotations=None, curr_ann_idx=None, ann_status=None):
        """
        :type target_id: int
        :type frame_id: int
        :type frame: np.ndarray
        :type location: np.ndarray
        :type annotations: Annotations | None
        :type curr_ann_idx: tuple | None
        :rtype: None
        """
        self._target_id = target_id
        self._frame_id = frame_id
        self._frame = frame

        self._annotations = annotations
        self._curr_ann_idx = curr_ann_idx
        self._ann_status = ann_status

        if annotations is not None:
            self._set_stats(ann_status)

            if not self._test_mode:
                self._set_train_stats(ann_status)

        if self._model is not None:
            self._model.set_id(self._target_id)

        if (self._model is None or not self._model.is_trained) and not self._track_heuristics:
            # self._logger.warning(f'No model or model is not trained so enabling track_heuristics')
            self._track_heuristics = 1
        # elif self._is_oracle:
        #     self._logger.warning(f'oracle does not support score prediction so enabling track_heuristics')
        #     self._track_heuristics = 1

        self.assoc_probabilities = []

        self.location[:] = location.reshape((1, 4))

        # if self._params.heuristic_features:
        #     self._similarity.fill(0)

        if self._model is None:
            """no model"""
            return

        if self._model.is_trained:
            """model has already been trained"""
            return

        if not self._params.summarize_templates:
            """filthy heuristics not needed"""
            return

        # self.train_features = np.zeros((2, self.n_features))
        # self.train_labels = np.zeros((2,))

        """idealized training samples for both positive and negative cases - annoying hangover from the original code"""
        features, labels = self._tracker.get_init_samples()
        if features is not None and labels is not None:
            _train_features = features
            _train_labels = labels

            if self._params.heuristic_features:
                heuristic_features = np.ones((_train_features.shape[0], 6), dtype=np.float32)
                neg_idx = np.flatnonzero(np.equal(labels, -1))
                heuristic_features[neg_idx, :] = 0

                _train_features = np.concatenate((_train_features, heuristic_features), axis=1)

            self._model.train(_train_labels, _train_features)
        self.state = MDPStates.lost

    def update(self, frame, frame_id, detections, predicted_location, prev_location,
               curr_ann_idx=None):
        """
        :type frame: np.ndarray
        :type frame_id: int
        :type detections: np.ndarray
        :type predicted_location: np.ndarray
        :type prev_location: np.ndarray
        :type curr_ann_idx: tuple
        :rtype: None
        """

        self._frame = frame
        self._frame_id = frame_id
        self._detections = detections
        self._predicted_location = predicted_location
        self._curr_ann_idx = curr_ann_idx

        """these should not be written out to state info if there are no detections in this frame"""
        self._features = None
        self._labels = None
        self._probabilities = None
        self.assoc_det_id = -1
        self._local_assoc_det_id = -1

        """these should be empty if  there are no detections in this frame"""
        self._indices = np.array([])
        self._height_ratios = np.array([])
        self._distances = np.array([])

        self.streak += 1

        self.state = MDPStates.lost
        self.score = 0

        """removed temporarily from training for debugging behavioral consistency"""
        # if self.is_test and self.streak > self.params.max_occlusion:
        #     self.state = MDPStates.inactive
        #     return

        if self._detections.shape[0] == 0:
            return

        """coarse detection filtering"""
        self._indices = self._get_matching_detections(frame, frame_id, self._detections, predicted_location,
                                                      prev_location)

        n_detections = self._indices.size
        if n_detections == 0:
            """probably temporary measure to achieve correspondence"""
            self._height_ratios = np.array([])
            self._distances = np.array([])
            return

        is_synthetic = np.zeros((n_detections,), dtype=np.bool)
        if self._params.save_mode and self._params.syn_samples and self._params.syn_method == 1:
            all_sampled_boxes, synthetic_source_dets = self._get_synthetic_samples(frame, frame_id,
                                                                                   self._detections[:, 2:6],
                                                                                   self._indices, n_detections,
                                                                                   syn_type=0)
            if all_sampled_boxes:
                synthetic_dets = np.zeros((len(all_sampled_boxes), 10))
                synthetic_dets[:, 2:6] = np.asarray(all_sampled_boxes)
                n_syn = synthetic_dets.shape[0]
                syn_indices = np.asarray(list(range(n_detections, n_detections + n_syn)))

                self._detections = np.concatenate((self._detections, synthetic_dets), axis=0)
                self._indices = np.concatenate((self._indices, syn_indices), axis=0)
                is_synthetic = np.concatenate((is_synthetic, np.ones((n_syn,), dtype=np.bool)), axis=0)

                # self._logger.info('generated {} synthetic samples'.format(n_syn))

                n_detections += n_syn

        self._labels = -np.ones((n_detections,))
        self._probabilities = np.zeros((n_detections, 2))
        self._probabilities[:, 1] = 1

        """
        original code does not retain the changes made in tracker while extracting association features; 
        causes a mind bogglingly extraordinarily amazingly annoying insidious bug while testing so
        deserves to be exterminated
        """
        # with self._templates.immutable():

        self._features = self._get_features(self._frame, self._frame_id, self._detections, self._indices)

        if self._features is None:
            self.state = MDPStates.inactive
            return

        _gt_labels = None if self._gt_labels is None or self._gt_labels[0] is None else self._gt_labels

        if self._model is None:
            self._labels = np.full((n_detections,), -1, dtype=np.int32)

            _tracking_status = self._track_res.get_status(template_id=self._templates.anchor_id)
            success_ids = np.where(_tracking_status == TrackingStatus.success)
            self._labels[success_ids] = 1

            self._probabilities = np.zeros((n_detections, 2), dtype=np.float32)
            for det_id in range(n_detections):
                _tracking_scores = self._track_res.get_scores(det_id)
                mean_score = np.mean(_tracking_scores.flatten()).item()
                self._probabilities[det_id, :] = (mean_score, 1 - mean_score)
        else:

            if self._model.is_trained:
                self._labels[:], self._probabilities[:] = self._model.predict(self._features, _gt_labels)
            else:
                """move back to tracked state since nothing useful can be done without a trained model"""
                self._labels = np.ones((n_detections,))
                self._probabilities = np.ones((n_detections, 2), dtype=np.float32)
                self._probabilities[:, 1] = 0

            if self._params.save_mode:
                is_real = np.logical_not(is_synthetic)
                real_ids = np.nonzero(is_real)[0]
                n_real = real_ids.size
                if n_real > 0:
                    self._add_test_samples(self._features[real_ids, ...], _gt_labels[real_ids], self._labels[real_ids],
                                           is_synthetic=0)

                if self._params.syn_samples:
                    if self._params.syn_method == 1:
                        n_synthetic = np.count_nonzero(is_synthetic)
                        if n_synthetic > 0:
                            self._add_test_samples(self._features[is_synthetic, ...], _gt_labels[is_synthetic],
                                                   self._labels[is_synthetic],
                                                   is_synthetic=1)
                    else:
                        all_synthetic_boxes = []
                        for det_id in real_ids:
                            if _gt_labels[det_id] != 1:
                                """synthetic samples from negative real samples for later"""
                                continue
                            best_tracked_template = self._best_tracked_templates[self._indices[det_id]]

                            bbox = self._track_res.locations[det_id, best_tracked_template, :]

                            """negative samples"""
                            synthetic_boxes, _ = self._get_synthetic_samples(self._frame, frame_id, bbox,
                                                                             None, 1,
                                                                             # min_size=10, max_size=100
                                                                             syn_type=0
                                                                             )

                            if synthetic_boxes:
                                n_synthetic_boxes = len(synthetic_boxes)
                                synthetic_boxes_arr = np.expand_dims(np.asarray(synthetic_boxes), axis=1)
                                syn_features = self._tracker.get_features(self._frame, best_tracked_template,
                                                                          synthetic_boxes_arr, n_synthetic_boxes)
                                """all negative samples"""
                                syn_labels = -np.ones((n_synthetic_boxes,))

                                self._add_test_samples(syn_features, syn_labels, None, is_synthetic=1)

                                all_synthetic_boxes += synthetic_boxes

                            """positive samples"""
                            synthetic_boxes, _ = self._get_synthetic_samples(self._frame, frame_id, bbox,
                                                                             None, 1,
                                                                             # min_size=10, max_size=100
                                                                             syn_type=1
                                                                             )

                            if synthetic_boxes:
                                n_synthetic_boxes = len(synthetic_boxes)
                                synthetic_boxes_arr = np.expand_dims(np.asarray(synthetic_boxes), axis=1)
                                syn_features = self._tracker.get_features(self._frame, best_tracked_template,
                                                                          synthetic_boxes_arr, n_synthetic_boxes)
                                """all negative samples"""
                                syn_labels = np.ones((n_synthetic_boxes,))

                                self._add_test_samples(syn_features, syn_labels, None, is_synthetic=1)

                                all_synthetic_boxes += synthetic_boxes

        if self._params.model_heuristics:
            invalid_idx = np.flatnonzero(self._model_heuristics_flags == 0)

            """zero association probability for heuristically-determined invalid detections
            """
            self._probabilities[invalid_idx, 0] = 0
            self._probabilities[invalid_idx, 1] = 1
            self._labels[invalid_idx] = -1

        if self._test_mode:
            self.__update_stats()

        # if self._params.pause_for_debug:
        #     self._logger.debug('paused')

    def __update_stats(self):
        if not self._params.enable_stats:
            return

        if self._gt_labels is None or self._gt_labels[0] is None:
            return

        for idx, (label, gt_label) in enumerate(zip(self._labels, self._gt_labels)):
            if label == 1:
                decision = 'positive'
                if gt_label == 1:
                    correctness = 'correct'
                else:
                    correctness = 'incorrect'
            else:
                decision = 'negative'
                if gt_label == 1:
                    correctness = 'incorrect'
                else:
                    correctness = 'correct'

            self._update_stats(correctness, decision)

        if self._writer is not None and self._model is not None:
            self._model.add_tb_image(self._writer, self._features, self._labels, self._gt_labels,
                                     iteration=None, stats=self._cmb_stats, batch_size=1,
                                     tb_vis_path=None, epoch=None, title='lost')

    def _associate(self, assoc_det_id=None):
        """associate using policy model"""
        self._ann_assoc = 0
        if self._params.verbose == 2:
            self._logger.info(f'Associating using policy...')

        if assoc_det_id is None:
            if self._probabilities is None:
                return False
            """association to the detection with the maximum probability
            """
            local_assoc_det_id = np.argmax(self._probabilities[:, 0])
            assoc_det_id = self._indices[local_assoc_det_id]

            label = self._labels[local_assoc_det_id]
            probabilities = self._probabilities[local_assoc_det_id, :]

            if self._params.vis_train:
                stacked_img = self._vis_assoc(self._frame, self._predicted_location, self._detections,
                                              self._indices, self._labels, frame_label=f'{self._frame_id} assoc')

            if label > 0:
                """successful association"""
                if self._params.verbose == 2:
                    self._logger.info(f'successful association to detection'
                                      f' {local_assoc_det_id}: {probabilities}')
                self.state = MDPStates.tracked
            else:
                """no association"""
                if self._params.verbose == 2:
                    self._logger.info(f'no association:'
                                      f'\n labels: {pformat(self._labels)}'
                                      f'\n probabilities: {pformat(self._probabilities)}'
                                      )
                self.state = MDPStates.lost

            assoc_prob = self._probabilities[local_assoc_det_id, 0]
            self.assoc_probabilities.append(assoc_prob)

        elif assoc_det_id < 0:
            """unsuccessful external association"""
            self.state = MDPStates.lost
            return False
        else:
            """successful external association"""
            local_assoc_det_id = np.flatnonzero(self._indices == assoc_det_id).item()
            self.state = MDPStates.tracked

        self.assoc_det_id = assoc_det_id
        self._local_assoc_det_id = local_assoc_det_id

        return True

    def _get_scores(self, track_res, det_id=0, gt_label=None):
        """

        :param TrackerBase.Result track_res:
        :param int det_id:
        :param np.ndarray gt_label:
        :return:
        """

        if self._track_heuristics:
            tracking_scores = track_res.get_scores(track_id=det_id)
            success_ids = track_res.get_success_ids(det_id)
        else:
            _features = track_res.features[det_id, ...]
            # _features = _features.squeeze(axis=0)

            all_gt_labels = np.full((_features.shape[0],), gt_label, dtype=np.int32) if gt_label is not None else None
            _labels, _probabilities = self._model.predict(_features, all_gt_labels)
            tracking_scores = _probabilities[:, 0]
            success_ids = np.argwhere(_labels == 1)

        return tracking_scores, success_ids

    def apply_heuristics(self):
        # if self._oracle_type == Oracle.Types.absolute:
        #     if self.streak > 0:
        #         self.state = MDPStates.inactive

        if self.streak > self._params.max_streak:
            self.state = MDPStates.inactive
            if self._params.verbose:
                print('target {:d} exits due to long time occlusion'.format(
                    self._target_id))

    def associate(self, assoc_with_ann, assoc_det_id=None):
        """
        :param int assoc_det_id: optional ID for external association - only used during hungarian association in Tester
        :param bool assoc_with_ann:
        :rtype: np.ndarray
        """
        # local_assoc_det_id = None

        if assoc_with_ann:
            if not self._associate_with_annotations():
                return None
        else:
            if self._oracle_type == Oracle.Types.absolute:
                self._gt_label_absolute = -1
                self._gt_location = None
                if self._curr_ann_idx is not None:
                    # is_occluded = self._annotations.data[self._curr_ann_idx[0], 11]
                    # if not is_occluded:

                    self._gt_label_absolute = 1
                    self._gt_location = self._annotations.data[self._curr_ann_idx[0], 2:6]

                label = self._gt_label_absolute
                if label > 0:
                    self.state = MDPStates.tracked
                    self.location = self._gt_location
                else:
                    self.state = MDPStates.lost
                """if association was successful, all scores become perfect, otherwise scores don't matter anyway"""
                tracking_scores = np.ones((self._templates.count,), dtype=np.float32)
                return tracking_scores

            if not self._associate(assoc_det_id):
                return None

        assoc_det_data = self._detections[self.assoc_det_id, :].reshape((1, 10))

        self.score = self._probabilities[self._local_assoc_det_id, 0]

        """           
        postprocessing that is done independently of whether or not the policy-based association was successful 
        however, it is not done if external association is being called for with an invalid ID
        
        2020-05-08 8:06:04 AM :: finally figured out that its main point is for training purposes   
        
        its only point seems to be to update the similarity feature that is only used for training so 
        such a disregard for the decision might make sense; 
        however, there seems to be a lag between the computation 
        of the similarity and its use for training since the latter will only happen in the next frame while 
        the former happens in the current frame; 
        this might simply be used to indicate exactly how sure we are that the last known location actually 
        correspondence to the correct object
        
        another point might be to choose the anchor template but only in case of successful association 
        since tracking_scores are used in subsequent call to templates.update
        
        """

        self.location[:] = assoc_det_data[0, 2:6].reshape((1, 4))

        """        
        track all the templates into the best matching detection to check if the tracked location of the 
        best matched template has sufficiently high overlap with this detection and, if so,  replace the location 
        with a weighted average of the detection and this tracked location
        """
        self._track_locations = self._locations_multi[self._local_assoc_det_id, ...]

        """for some foul annoying buggy reason, this tracking was done again in the original code"""
        # tracking_result = self._templates.track(np.expand_dims(self._track_locations, 0),
        #                                         np.expand_dims(self._roi[self._local_assoc_det_id, :, :], 0),
        #                                         np.expand_dims(self._transform[self._local_assoc_det_id, :], 0),
        #                                         debug=0,
        #                                         heuristics=self._track_heuristics
        #                                         )  # type: Tracker.Result

        # is_eq = np.isclose(self._track_locations2, self._track_locations)
        # is_same = np.all(is_eq)
        # assert is_same, "weird discrepancy between _track_locations and _track_locations2"

        if not self._params.summarize_templates:
            tracking_scores, _ = self._get_scores(self._track_res,
                                                  det_id=self._local_assoc_det_id,
                                                  gt_label=self._gt_labels[self._local_assoc_det_id])

            return tracking_scores

        self._templates.apply_heuristics(self._frame_id, assoc_det_data, self._track_locations, self._track_res,
                                         check_ratio=False,
                                         track_id=self._local_assoc_det_id)
        tracking_scores, success_ids = self._get_scores(self._track_res,
                                                        det_id=self._local_assoc_det_id,
                                                        gt_label=self._gt_labels[self._local_assoc_det_id])
        self._assoc_success_ids[self._frame_id] = success_ids

        """
        an amazingly annoying bug in the original code – location is being updated independently of the policy 
        on the basis of annoying heuristics and yet the same location is used while training to decide whether or not 
        the policy was successful with an equally annoying hard threshold heuristic that can easily be failed by 
        slight modifications of the location which is exactly what this does
        
        as a result, even if the original location was obtained using the GT and should therefore result in 
        a correct decision as far as training goes, this annoying modification can cause it to be classified as 
        an incorrect decision
        """
        if not assoc_with_ann or self._test_mode:
            best_tracked_idx = np.argmax(tracking_scores)
            if self._templates.max_iou[best_tracked_idx] > self._params.iou_det_box:
                """weighted average of locations of the minimum error template and its 
                maximally overlapping detection"""
                self.location[:] = np.average(np.concatenate(
                    (self.location, self._track_locations[best_tracked_idx, :].reshape((1, 4))), axis=0), axis=0,
                    weights=(self._params.weight_association, self._params.weight_tracking))

        # if self._params.heuristic_features:
        #     self._templates.get_similarity(self._similarity, self._frame, self.location, self.similarity_type)

        return tracking_scores

    def _associate_with_annotations(self):

        assert self._annotations is not None, "annotations must be provided to associate using them"

        """associate using annotations - proxy for an ideal policy"""
        self._ann_assoc = 1

        if self._curr_ann_idx is None:
            if self._params.verbose:
                self._logger.warning(
                    f'target {self._target_id}, frame {self._frame_id} :: no annotations for association')
            return False

        indices = self._indices
        n_detections = indices.size
        if n_detections == 0:
            if self._params.verbose:
                self._logger.warning(f'target {self._target_id}, frame {self._frame_id} :: '
                                     f'No valid detections for association')
            return False

        frame_id = self._frame_id
        if self._params.verbose == 2:
            self._logger.info(f'Associating using annotations...')

        """Associate using annotations"""

        """index of current annotation within all annotations in this frame"""
        max_det_iou_idx, max_det_iou = self._get_max_iou_det(indices)
        # det_iou = self._annotations.cross_iou[frame_id][:, self._curr_ann_idx[1]]
        # max_det_iou_idx = np.argmax(det_iou[indices])
        # max_det_iou = det_iou[indices[max_det_iou_idx]].item()

        self._local_assoc_det_id = max_det_iou_idx
        self.assoc_det_id = indices[self._local_assoc_det_id]

        self._probabilities[:, 0] = 0
        self._probabilities[:, 1] = 1

        """conservative association heuristic – successful association only happens when the annotation is 
        absolutely clearly visible and the nearest detection has high overlap"""
        if self._annotations.max_ioa[self._curr_ann_idx[0]] <= self._params.ann_ioa_thresh:
            if max_det_iou > self._params.iou_pos:
                """successful association"""
                self.state = MDPStates.tracked
                self._probabilities[self._local_assoc_det_id, 0] = 1
                self._probabilities[self._local_assoc_det_id, 1] = 0
                self._labels[self._local_assoc_det_id] = 1
            else:
                """no association"""
                self.state = MDPStates.lost

        return True

    def train_async(self, frame, frame_id, detections, predicted_location, prev_location):
        """
        asynchronous training - train even when the previous target state wasn't lost

        :param np.ndarray frame:
        :param int frame_id:
        :param np.ndarray detections:
        :param np.ndarray predicted_location:
        :param np.ndarray prev_location:
        :param Annotations annotations: All annotations in the current sequence
        :param np.ndarray ann_idx: index of the annotation belonging to the current trajectory
        :return: None
        """

        self._detections = detections
        annotations = self._annotations

        # ann_idx = self._curr_ann_idx

        assert annotations is not None, "annotations must be provided to train_async"

        if self._model is None:
            self._logger.warning('No model to train')
            return

        if self._curr_ann_idx is None:
            if self._params.verbose:
                self._logger.warning(
                    f'No annotations for asynchronous training in frame {frame_id} for target {self._target_id}')
            return

        n_detections = detections.shape[0]
        if n_detections == 0:
            if self._params.verbose:
                self._logger.warning(
                    f'No detections for asynchronous training in frame {frame_id} for target {self._target_id}')
            return

        if annotations.max_ioa[self._curr_ann_idx[0]] > self._params.ann_ioa_thresh:
            """annotation not clearly visible - either occludes or is occluded by another"""

            if self._params.verbose:
                """skip training for now"""
                self._logger.warning(
                    f'annotation not clearly visible in frame {frame_id} for target {self._target_id}')
                return

        """no coarse filtering"""
        # indices = np.array(list(range(n_detections)), dtype=np.int32)
        indices = self._get_matching_detections(frame, frame_id, detections, predicted_location, prev_location)

        n_detections = indices.size
        if n_detections == 0:
            if self._params.verbose:
                self._logger.warning(f'No valid detections for asynchronous training in frame {frame_id} '
                                     f'for target {self._target_id}')
            return

        with self._templates.immutable():
            features = self._get_features(frame, frame_id, detections, indices)

        learn_features = np.zeros((n_detections, self._n_features), dtype=np.float32)
        learn_features[:] = features

        """all detections are negatives by default"""
        learn_labels = np.full((n_detections,), -1, dtype=np.float32)

        max_det_iou_idx, max_det_iou = self._get_max_iou_det(indices)

        # det_iou = annotations.cross_iou[frame_id][:, self._curr_ann_idx[1]]
        # max_det_iou_idx = np.argmax(det_iou[indices])
        # max_det_iou = det_iou[indices[max_det_iou_idx]].item()

        if max_det_iou > self._params.iou_pos:
            """annotation detected correctly"""

            """the matching detection is a positive sample, all others remain negatives"""
            learn_labels[max_det_iou_idx] = 1
        else:
            """annotation not detected correctly - false negative"""
            pass

        if self._params.vis_train:
            stacked_img = self._vis_train(frame, annotations.data[self._curr_ann_idx[0], 2:6], detections, indices,
                                          learn_labels, frame_label=f'{frame_id} async')

        self._train(learn_features, learn_labels)
        return

    def train(self):
        """
        :type annotations: Annotations
        :param np.ndarray ann_idx: index of the annotation belonging to the current trajectory
        and the current frame
        :rtype: (bool, int)
        """

        if self._model is None:
            self._logger.warning('No model to train')
            return True

        annotations = self._annotations
        # ann_idx = self._curr_ann_idx

        assert annotations is not None, "annotations must be provided to train"

        reward = 0
        is_end = False

        label = 0
        max_det_iou_idx = None
        max_det_iou = 0

        # _n_correct_assoc = _n_incorrect_assoc = _n_unknown_assoc = 0

        # correctness = None

        if self.state == MDPStates.tracked:
            decision = 'positive'
        else:
            decision = 'negative'

        if self._features is not None:
            """use only the associated detection features for learning"""
            self._learn_features[:] = self._features[self._local_assoc_det_id, :].reshape((1, self._n_features))

        if self._curr_ann_idx is None:
            """no annotations in this frame"""
            if self._params.verbose:
                self._logger.debug(f'no annotation for object {self._target_id} in frame {self._frame_id}')
        else:
            is_occluded = annotations.data[self._curr_ann_idx[0], 11]
            if not is_occluded and self._indices.size > 0:
                max_det_iou_idx, max_det_iou = self._get_max_iou_det(self._indices)

            if self._params.always_train:
                """
                train on all filtered detections and even if the policy didn't make a mistake
                """
                # if annotations.max_ioa[self._curr_ann_idx[0]] <= self._params.ann_ioa_thresh:
                #     """annotation clearly visible by not having been overlapped by any others"""

                """train on all valid detections"""
                n_valid_det = self._indices.size
                self._learn_features = np.zeros((n_valid_det, self._n_features), dtype=np.float32)
                self._learn_features[:] = self._features
                """all detections are negatives by default"""
                self._learn_labels = np.full((n_valid_det,), -1, dtype=np.float32)
                if max_det_iou > self._params.iou_pos:
                    """annotation detected correctly"""

                    """only the best matching detection is a positive sample, all others remain negatives"""
                    self._learn_labels[max_det_iou_idx] = 1
                else:
                    """target not detected correctly - false negative"""
                    pass

                if self._params.vis_train:
                    stacked_img = self._vis_train(self._frame, annotations.data[self._curr_ann_idx[0], 2:6],
                                                  self._detections,
                                                  self._indices, self._learn_labels, frame_label=f'{self._frame_id}')

                self._train(self._learn_features, self._learn_labels)
                return is_end, reward

                # else:
                #     """annotation not clearly visible - either occludes or is occluded by another"""
                #     if self._params.verbose:
                #         self._logger.debug('annotation not clearly visible - '
                #                            'either occludes or is occluded by another')
                #     """skip training for now"""
                #     return is_end, reward

        """annoying cornucopia of heuristics to decide if and how to train
        """
        if max_det_iou > self._params.iou_pos:
            """
            max iou detection corresponds to GT
            """
            if self.state == MDPStates.tracked:
                """
                max iou detection corresponds to GT
                successful association during update
                """
                tracked_iou = np.empty((1,))
                """
                iou between GT and tracked location
                """
                compute_overlap(tracked_iou, None, None, annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4)),
                                self.location.reshape((1, 4)))
                if tracked_iou[0] > self._params.iou_pos:
                    """
                    max iou detection corresponds to GT
                    successful association during update
                    correct association since tracked location has high IOU with GT
                    """
                    reward = 1
                    """should transition to tracked state since that was a correct decision
                    """
                    label = 1
                    # _n_correct_assoc = 1
                    correctness = 'correct'
                else:
                    """max iou detection corresponds to GT
                    successful association during update
                    incorrect association since tracked location has low IOU with GT
                    """
                    reward = -1
                    """should remain in lost state since transitioning to tracked state was an incorrect decision
                    """
                    label = -1
                    is_end = True
                    if self._params.verbose:
                        self._logger.debug(f'target {self._target_id}, frame {self._frame_id} :: '
                                           f'associated to wrong target'
                                           f' ({"{:.3f}".format(max_det_iou)},'
                                           f'{"{:.3f}".format(tracked_iou[0])})'
                                           f'! Game over')
                    # _n_incorrect_assoc = 1
                    correctness = 'incorrect'
            else:
                """
                max iou detection corresponds to GT
                no association during update
                """
                if annotations.max_ioa[self._curr_ann_idx[0]] <= self._params.ann_ioa_thresh:
                    """max iou detection corresponds to GT
                    no association during update
                    annotation does not overlap any others
                    """

                    """should transition to tracked state as object is absolutely clearly visible with 0 occlusion
                    """
                    label = 1

                    if self._params.model_heuristics and self._assoc_success_ids[self._frame_id].size == 0:
                        """all templates failed to get tracked
                        """
                        reward = 0
                        """
                        don't know if this decision was correct or not
                        """
                        # _n_unknown_assoc = 1
                        correctness = 'ambiguous'
                        if self._params.verbose == 2:
                            self._logger.debug('all templates failed to get tracked')

                        # stacked_img = self._vis_train(self._frame, annotations.data[ann_idx, 2:6], self._detections,
                        #                               self._indices, self._learn_labels,
                        #                               frame_label=f'{self._frame_id}')
                    else:
                        """no association during update
                        annotation does not overlap any others                        
                        """
                        reward = -1
                        """should transition to tracked state as object is absolutely clearly visible with 0 occlusion
                        """
                        label = 1
                        is_end = True
                        self._logger.debug('Failed to associate')
                        # _n_incorrect_assoc = 1
                        correctness = 'incorrect'

                        """assuming that the current detections are the same as those used in the
                        previous call to update where ratios and distances were computed
                        """
                        best_det_id = np.array(self._indices[max_det_iou_idx], ndmin=1)

                        with self._templates.immutable(self._params.copy_while_training):
                            """use features from the detection that best matches the GT for learning
                            """
                            self._learn_features[:] = self._get_features(self._frame, self._frame_id, self._detections,
                                                                         best_det_id)

                        """figured out the reason for the weird discrepancy in LK output – apparently this
                        is the only case where multi track is called in the original code without
                        being followed by single track before state variables are written so that the
                        changes in the original code are only made to the multi version of the output
                        matrix and therefore not written to the file by the unified tracking matrix
                        """
                        if self._params.verbose:
                            self._logger.debug('Missed association!')
                else:
                    """no association during update
                    annotation has non-zero overlap with others
                    """
                    reward = 1
                    # _n_correct_assoc = 1
                    correctness = 'correct'

        else:
            """
            max iou detection does not correspond to GT - either a false negative or  mistake in filtering or 
            no GT at all
            """

            """
            should remain lost in all 3 of the cases
            """
            label = -1

            if self.state == MDPStates.lost:
                """
                max iou detection does not correspond to GT - either a false negative or mistake in filtering or no GT
                no association during update - correct decision
                """
                reward = 1
                # _n_correct_assoc = 1
                correctness = 'correct'
            else:
                """
                max iou detection does not correspond to GT - either a false negative or mistake in filtering or no GT
                successful association during update
                """
                tracked_iou = np.zeros((1,))
                if self._curr_ann_idx is not None:
                    """
                    iou between GT and tracked location
                    """
                    compute_overlap(tracked_iou, None, None,
                                    annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4)),
                                    self.location.reshape((1, 4)))
                if tracked_iou[0] < self._params.iou_neg or max_det_iou < self._params.iou_neg:
                    """
                    max iou detection does not correspond to GT - either a false negative or mistake in filtering or 
                    no GT
                    successful association during update
                    neither tracked location nor max iou detection overlaps even slightly well with the GT so this is
                    a negative sample, i.e. one where correct decision is to remain in Lost state
                    """
                    reward = -1
                    label = -1
                    is_end = True
                    # _n_incorrect_assoc = 1
                    correctness = 'incorrect'
                    if self._params.verbose:
                        self._logger.debug('associated to wrong target! Game over')
                else:
                    """
                    max iou detection does not correspond to GT - either a false negative or mistake in filtering or 
                    no GT
                    successful association during update
                    either tracked location or max iou detection overlaps at least slightly well with the GT though the 
                    latter definitely does not overlap well
                    """
                    reward = 0
                    # _n_unknown_assoc = 1
                    correctness = 'ambiguous'
                    if self._params.verbose == 2:
                        self._logger.debug('ambiguous decision')

        if not self._ann_assoc:
            self._update_train_stats(correctness, decision)

            # self._n_correct_assoc += _n_correct_assoc
            # self._n_incorrect_assoc += _n_incorrect_assoc
            # self._n_unknown_assoc += _n_unknown_assoc
            # self._n_total_assoc += 1

            # assert self._n_correct_assoc + self._n_incorrect_assoc + self._n_unknown_assoc == self._n_total_assoc, \
            #     "association status counts do not match the total:\n" \
            #     "total: {:d}, correct: {:d}, incorrect: {:d}, unknown: {:d}".format(
            #         self._n_total_assoc, self._n_correct_assoc, self._n_incorrect_assoc, self._n_unknown_assoc)

        if self._params.verbose:
            if self._ann_assoc:
                assert correctness == 'correct', "incorrect association even when using annotations"
            elif self._params.verbose == 2 or self._target_id != self._prev_id:
                self._prev_id = self._target_id
                # self._correct_assoc = float(self._n_correct_assoc) / float(self._n_total_assoc) * 100
                # self._incorrect_assoc = float(self._n_incorrect_assoc) / float(self._n_total_assoc) * 100
                # self._unknown_assoc = float(self._n_unknown_assoc) / float(self._n_total_assoc) * 100
                # self._logger.info('target {:d}: total: {:d}, correct: '
                #                   '{:.2f}%, incorrect: {:.2f}%, unknown: {:.2f}% '.format(
                #     self._target_id, self._n_total_assoc, self._correct_assoc, self._incorrect_assoc,
                #     self._unknown_assoc))

                # print_stats(self.train_stats, self._name)
                # print_stats(self.train_pc_stats, self._name)

        if reward == -1 and self._features is not None:
            """
            only train when we associated incorrectly
            """
            # print 'self.labels.shape: ', self.labels.shape
            # print 'self.labels: ', self.labels
            # print 'label: ', label
            self._learn_labels = np.array((label,))
            self._train(self._learn_features, self._learn_labels)

        return is_end, reward

    def _update_train_stats(self, correctness, decision):
        self._train_stats['total'] += 1
        self._cmb_train_stats['total'] += 1

        for _id in (correctness, correctness + '_' + decision):
            self._train_stats[_id] += 1
            self._cmb_train_stats[_id] += 1

            self._train_pc_stats[_id] = (self._train_stats[_id] / self._train_stats['total']) * 100.0
            self._cmb_train_pc_stats[_id] = (self._cmb_train_stats[_id] / self._cmb_train_stats[
                'total']) * 100.0

    def _get_max_iou_det(self, indices):
        """
        find maximally overlapping detection with the current annotation
        :return:
        """
        det_iou = np.empty((indices.size, 1))
        compute_overlap(det_iou, None, None,
                        self._annotations.data[self._curr_ann_idx[0], 2:6].reshape((1, 4)),
                        self._detections[indices, 2:6].reshape((indices.size, 4)), self._logger)
        max_det_iou_idx = np.argmax(det_iou)
        max_det_iou = det_iou[max_det_iou_idx].item()

        return max_det_iou_idx, max_det_iou


    def get_distances(self, dist):
        """
        :type dist: np.ndarray
        :rtype: None
        """
        dist[:] = np.inf
        if self._probabilities is not None:
            """distance is equal to the probability of non-association
            """
            dist[self._indices] = self._probabilities[:, 1]

    def _get_matching_detections(self, frame, frame_id, detections, predicted_location, prev_location):
        """
        superficial heuristical filtering to filter detections that obviously don't correspond to the target
        based on location and shape as compared to the predicted target location

        :type frame: np.ndarray
        :type frame_id: int
        :type predicted_location: np.ndarray
        :type prev_location: np.ndarray
        :rtype: np.ndarray
        """

        predicted_location = predicted_location.reshape((1, 4))
        self._predicted_center = predicted_location[0, :2] + predicted_location[0, 2:] / 2.0
        """
        inconsistent behavior retained for the sake of conformity with the original code
        """
        self._det_centers = detections[:, 2:4] + detections[:, 4:6] / 2.0

        # self.distances = np.linalg.norm(self.det_centers - self.predicted_center, axis=1) / predicted_location[0, 2]
        # self.ratios = predicted_location[0, 3] / det_data[:, 5]

        prev_mean_size = (prev_location[2] + prev_location[3]) / 2.0

        """
        annoying ad-hoc heuristics from the original code
        """

        """normalized distance of predicted box from detections
        """
        self._distances = np.linalg.norm(self._det_centers - self._predicted_center, axis=1) / prev_mean_size
        """ratio of previous height to detection height
        """
        self._height_ratios = prev_location[3] / detections[:, 5]
        self._height_ratios = np.minimum(self._height_ratios, np.reciprocal(self._height_ratios))
        """ratio of previous width to detection width
        """
        self.width_ratios = prev_location[2] / detections[:, 4]
        self.width_ratios = np.minimum(self.width_ratios, np.reciprocal(self.width_ratios))
        """indices of detections that might be possible matches for the target's predicted location
        """
        indices = np.flatnonzero(np.logical_and(
            self._distances < self._params.threshold_dist,
            self._height_ratios > self._params.threshold_ratio,
            self.width_ratios > self._params.threshold_ratio,
        ))

        if self._params.vis:
            frame_disp = np.copy(frame)
            draw_box(frame_disp, predicted_location, color='red')
            draw_box(frame_disp, prev_location, color='green')
            n_det = detections.shape[0]
            for _det_id in range(n_det):
                if _det_id in indices:
                    color = 'blue'
                else:
                    continue
                    # color = 'black'
                draw_box(frame_disp, detections[_det_id, 2:6], color=color, _id=_det_id)

            cv2.putText(frame_disp, 'frame {:d}'.format(frame_id), (5, 15),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, col_bgr['green'], 1, cv2.LINE_AA)

            frame_disp = resize_ar(frame_disp, 1280, 720)

            # cv2.imshow('_getMatchingDetections', frame_disp)
            self._pause = annotate_and_show('_getMatchingDetections', frame_disp,
                                            f'frame {self._frame_id} target {self._target_id}', self._pause)

        return indices

    def _get_features(self, frame, frame_id, detections, indices):
        """
        Track each one of the templates into patches extracted from around each one of the filtered detections
        and use the results to produce summarized features for each one of the detections

        :type indices: np.ndarray
        :rtype: np.ndarray
        """
        if indices is None:
            indices = np.array(list(range(detections.shape[0])), dtype=np.int32)

        n_det = indices.size

        features = np.zeros((n_det, self._n_features), dtype=np.float32)
        self._model_heuristics_flags = np.zeros((n_det, 1), dtype=np.bool)

        _detections = detections[indices, 2:6].reshape((n_det, 4))

        track_res = self._tracker.track(frame, frame_id, _detections, indices.size,
                                        heuristics=self._track_heuristics)  # type: TrackerBase.Result

        if track_res is None:
            return None

        self._track_res = track_res

        self._locations_multi = self._track_res.locations

        self._gt_labels = self._get_labels(self._locations_multi, self._track_res, indices)

        self._best_tracked_templates = {}

        _labels = None

        for det_id in range(n_det):
            if not self._params.summarize_templates:
                features[det_id, :self._n_track_features] = self._track_res.features[det_id, ...].flatten()
                continue

            det_data = detections[indices[det_id], :].reshape((1, 10))
            tracked_locations = self._locations_multi[det_id, :, :]  # type: np.ndarray

            self._templates.apply_heuristics(self._frame_id, det_data, tracked_locations,
                                             track_id=det_id, check_ratio=False,
                                             track_res=self._track_res,
                                             )
            if self._params.vis:
                frame_disp = np.copy(frame)
                draw_box(frame_disp, det_data[0, 2:6], color='black')
                for i in range(tracked_locations.shape[0]):
                    draw_box(frame_disp, tracked_locations[i, :], color='red')
                annotate_and_show('frame_disp', frame_disp, f'det {det_id}')

            tracking_scores, success_ids = self._get_scores(self._track_res, det_id, self._gt_labels[det_id])
            best_tracked_template = np.argmax(tracking_scores)

            """debugging"""
            # _labels, _probabilities = self._model.predict(tracking_result.get_features(det_id).squeeze())
            # tracking_scores2 = _probabilities[:, 0]            #
            # tracking_scores_cmb = np.stack((tracking_scores, tracking_scores2), axis=0)            #
            # best_tracked_template2 = np.argmax(tracking_scores2)

            self._best_tracked_templates[indices[det_id]] = best_tracked_template

            if self._params.heuristic_features:
                """always 0"""
                _det_id = self._templates.max_iou_det_idx[best_tracked_template]
                self._det_location[:] = det_data[_det_id, 2:6].reshape((1, 4))
                if self._templates.max_iou[best_tracked_template] > self._params.iou_det_box:
                    """weighted average of locations of the best tracked template and its maximally overlapping 
                    detection
                    """
                    self._det_location[:] = np.average(np.concatenate(
                        (self._det_location,
                         self._locations_multi[det_id, best_tracked_template, :].reshape((1, 4))),
                        axis=0), axis=0,
                        weights=(self._params.weight_association, self._params.weight_tracking))

                self._templates.get_similarity(self._similarity, frame, self._det_location, self._similarity_type)

            # success_ids = None
            # if self._params.model_heuristics:

            if success_ids.size == 0:
                """none of the templates tracked this detection
                """
                if self._params.model_heuristics:
                    """supposed to represent invalid features - filling with zeros only works when each feature 
                    goes from 0 to 1 and 0 represents failure
                    hangover from original code with LK features, left for now till a correspondence for 
                    Siamese can be found
                    """
                    features[det_id, :].fill(0)
                else:
                    """assume that the best tracked template was successful anyway as a (poorly designed) hack to get 
                    features for each of the detections even if they are garbage"""
                    success_ids = np.array([best_tracked_template, ])

            if success_ids.size > 0:
                """annoying model_heuristics are disabled or at least one template tracked successfully
                """

                """tracker specific features"""
                features[det_id, :self._n_track_features] = self._track_res.get_summarized_features(
                    success_ids, det_id)

                if self._params.heuristic_features:
                    """annoying arbitrary ad-hoc heuristical hand-crafted features"""
                    features[det_id, self._n_track_features:] = [
                        np.mean(self._templates.max_iou[success_ids]),
                        np.mean(self._similarity[success_ids]),
                        np.mean(self._templates.ratios[success_ids]),
                        self._templates.max_iou_det_scores[0] / self._max_score,
                        self._height_ratios[indices[det_id]],
                        np.exp(-self._distances[indices[det_id]])
                    ]
                self._model_heuristics_flags[det_id] = True

        return features

    def reset_streak(self):
        self.streak = 0

    def _vis_train(self, frame, annotation, detections, indices, labels, frame_label=''):
        """

        :param np.ndarray frame:
        :param np.ndarray annotation:
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
        draw_box(frame_vis, annotation, color='green', thickness=2)
        for i in range(indices.size):
            if labels[i] == -1:
                color = 'red'
            else:
                color = 'blue'

            draw_box(frame_vis, detections[indices[i], 2:6],
                     _id='{}:{}'.format(indices[i], self._best_tracked_templates[indices[i]]),
                     color=color, thickness=2)

        if frame_label:
            cv2.putText(frame_vis, frame_label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col_bgr['green'], 1, cv2.LINE_AA)

        templates, stacked_frames = self._tracker.get_stacked_roi(
            self._templates.anchor_id, self._templates.frame_ids, self._templates.frames, self._templates.locations,
            show=0, grid_size=[-1, 2])

        if len(templates.shape) == 2:
            templates = cv2.cvtColor(templates, cv2.COLOR_GRAY2BGR)

        templates = resize_ar(templates, width=frame_vis.shape[1])
        stacked_img = stack_images((templates, frame_vis), grid_size=[2, 1])

        stacked_img = resize_ar(stacked_img, height=2000)
        stacked_frames = resize_ar(stacked_frames, height=2000)

        annotate_and_show('vis_train stacked_img', stacked_img, 'lost:_vis_train', pause=-1)
        self._pause = annotate_and_show('vis_train stacked_frames', stacked_frames,
                                        f'frame {self._frame_id} target {self._target_id}', pause=self._pause)

        return stacked_img

    def _vis_assoc(self, frame, predicted_location, detections, indices, labels, frame_label=''):
        """

        :param np.ndarray frame:
        :param np.ndarray annotation:
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
        draw_box(frame_vis, predicted_location, color='green', thickness=2)
        for i in range(indices.size):
            if labels[i] == -1:
                color = 'red'
            else:
                color = 'blue'

            draw_box(frame_vis, detections[indices[i], 2:6],
                     _id='{}:{}'.format(indices[i], self._best_tracked_templates[indices[i]]),
                     color=color, thickness=2)

        if frame_label:
            cv2.putText(frame_vis, frame_label, (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col_bgr['green'], 1, cv2.LINE_AA)

        templates, stacked_frames = self._tracker.get_stacked_roi(
            self._templates.anchor_id, self._templates.frame_ids, self._templates.frames, self._templates.locations,
            show=0, grid_size=[-1, 2])

        if len(templates.shape) == 2:
            templates = cv2.cvtColor(templates, cv2.COLOR_GRAY2BGR)

        templates = resize_ar(templates, width=frame_vis.shape[1])
        stacked_img = stack_images((templates, frame_vis), grid_size=[2, 1])

        stacked_img = resize_ar(stacked_img, height=2000)
        # stacked_frames = resizeAR(stacked_frames, height=2000)

        self._pause = annotate_and_show('vis_train stacked_img', stacked_img,
                                        f'frame {self._frame_id} target {self._target_id}', pause=self._pause)

        return stacked_img

    def save(self, save_dir, summary_dir):
        """
        :type save_dir: str
        :type summary_dir: str
        :rtype: None
        """
        if self._model is None:
            self._logger.warning('No model to save')
            return True
        else:
            # curr_dir = '{:s}/lost'.format(root_dir)

            os.makedirs(save_dir, exist_ok=True)

            self._model.save(save_dir)
            # pickle.dump(self._train_features, open('{:s}/features.bin'.format(curr_dir), "wb"))
            # pickle.dump(self._train_labels, open('{:s}/labels.bin'.format(curr_dir), "wb"))

        """apparently always 1"""
        # pickle.dump(self._max_score, open('{:s}/max_score.bin'.format(save_dir), "wb"))

        os.makedirs(summary_dir, exist_ok=True)

        stats_file = '{}/lost.csv'.format(summary_dir)
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

        write_df(self.train_stats, stats_file, 'lost', time_stamp)

    # def write_state_info(self, files, root_dir, write_to_bin, write_roi,
    #                      fp_fmt='%.4f', fp_dtype=np.float32):
    #     """
    #     :type files: list[(str, Type(np.dtype), tuple)]
    #     :type root_dir: str
    #     :type write_to_bin: bool
    #     :type write_roi: bool
    #     :type fp_fmt: str
    #     :type fp_dtype: Type(np.dtype)
    #     :rtype: None
    #     """
    #     log_dir = '{:s}/lost'.format(root_dir)
    #
    #     entries = [
    #         (self._distances, 'det_distances', fp_dtype, fp_fmt),
    #         (self._height_ratios, 'det_ratios', fp_dtype, fp_fmt),
    #     ]
    #
    #     # if not self._test_mode:
    #     #     self._train_features_trunc = self._train_features[:, self._trunc_idx]
    #     #     entries.append((self._train_features_trunc, 'train_features', fp_dtype, fp_fmt))
    #     #     entries.append((self._train_labels, 'train_labels', fp_dtype, fp_fmt))
    #     # if self.labels is not None:
    #     #     entries.append((self.labels, 'labels', fp_dtype, fp_fmt))
    #     # if self.probabilities is not None:
    #     #     entries.append((self.probabilities, 'probabilities', fp_dtype, fp_fmt))
    #
    #     if self._features is not None:
    #         self._features_trunc = self._features[:, self._trunc_idx]
    #         entries.append((self._features_trunc, 'features', fp_dtype, fp_fmt))
    #         # entries.append((self._similarity, 'similarity', fp_dtype, fp_fmt))
    #
    #     if write_roi and self._roi is not None:
    #         n_det = self._roi.shape[0]
    #         roi_size = self._roi.shape[1] * self._roi.shape[2]
    #         roi_flattened = np.empty((n_det, roi_size), dtype=np.uint8)
    #         for i in range(n_det):
    #             roi_flattened[i, :] = self._roi[i, :, :].flatten(order='F')
    #         entries.append((roi_flattened, 'roi', np.uint8, '%d'))
    #     write_to_files(log_dir, write_to_bin, entries)
    #     # files.extend(['lost/{:s}'.format(entry[1]) for entry in entries])
    #     files.extend([('lost/{:s}'.format(entry[1]), entry[2], entry[0].shape) for entry in entries])
