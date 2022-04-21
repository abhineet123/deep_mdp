import numpy as np
import cv2
import functools
from numpy import linalg
from contextlib import contextmanager

import paramparse

try:
    import pyWarp
except BaseException as e:
    print('pyWarp unavailable')
    pyWarp = None

try:
    import pyDist
except BaseException as e:
    print('pyDist unavailable')
    pyDist = None

from trackers.lk import LK
from trackers.siamese import Siamese
from trackers.pyt import PYT
from trackers.dummy import IdentityTracker, GTTracker
from trackers.tracker_base import TrackerBase

from utilities import BaseParams, PY_DIST, CVConstants, compute_overlap, compute_overlaps_multi, draw_box, \
    stack_images, get_unique_ids, ids_to_member_names


class Templates:
    """
    :type _params: Templates.Params
    :type _logger: logging.RootLogger | CustomLogger
    :type tracker: Siamese | LK | IdentityTracker | GTTracker
    :type _rgb_input: int | bool
    """

    class Params(BaseParams):
        """

        :ivar tracker: {
            'LK': ('0', 'lk'),
            'Siamese': ('1', 'siam', 'siamese'),
            'PYT': ('2', 'pyt', 'pytracking'),
            'Identity': ('3', 'id_tracker', 'none', 'idt'),
            'GT': ('4', 'gt', 'gtt', 'gt_tracker'),
        }

        :ivar roi: 'ROI settings for the tracker; only provided for convenience in setting from cfg files; ',
        :ivar count: 'number of templates',
        :ivar init_shift: 'proportional shifts used to generate initial template locations'
                      ' about the main target location',
        :ivar min_ratio: 'min allowed height ratio',
        :ivar min_velocity_norm: 'min allowed velocity norm in LK',
        :ivar max_velocity_frames: 'max frames used to compute mean velocity',
        :ivar pattern_shape: 'Patch shape for template appearance model - (n_rows, n_cols)',

        :ivar pattern_interp_type_id: 'ID of the interpolation type used while resizing images before extracting the '
                          'patches that represent the template appearance - indexes into '
                          'Utilities.CVConstants.interp_types',

        :ivar min_keyframe_gap: 'minimum gap between successive keyframes from where templates are extracted',

        :ivar vis: 'visualize results',
        :ivar lk: 'LK.Params',
        :ivar siamese: 'Siamese.Params',
        :ivar pause_for_debug: 'pause execution for debugging'


        """

        def __init__(self):
            self.tracker = 'lk'
            self.roi = TrackerBase.Params.ROI()
            self.count = 10

            # self.max_detections = 10  # max number of detections per frame

            self.init_shift = (
                (-0.01, -0.01),
                (-0.01, 0.01),
                (0.01, -0.01),
                (0.01, 0.01)
            )
            self.min_ratio = 0.9
            self.min_velocity_norm = 0.2
            self.max_velocity_frames = 3
            self.pattern_shape = (24, 12)

            self.update_location = 0

            self.sub_pix_method = 1

            self.min_keyframe_gap = 1

            self.pattern_interp_type_id = 1
            self.velocity_heuristics = 0
            self.ratio_heuristics = 1

            self.pause_for_debug = 0
            self.vis = 0
            self.verbose = 0

            self.lk = LK.Params()
            self.siamese = Siamese.Params()
            self.pyt = PYT.Params()
            self.idt = IdentityTracker.Params()
            self.gtt = GTTracker.Params()

    def __init__(self, params, rgb_input, heuristics, logger, parent=None):
        """
        :type params: Templates.Params
        :type logger: logging.RootLogger | CustomLogger
        :type rgb_input: int | bool
        :type heuristics: int | bool
        :type parent: Templates | None
        :rtype: None
        """
        self._params = params
        self._logger = logger
        self._rgb_input = rgb_input
        self._heuristics = heuristics
        self.ratio_heuristics = self._params.ratio_heuristics
        self._target_id = -1

        if parent is not None:
            tracker = parent.tracker
            self._members_to_spawn = parent._members_to_spawn
            for _member in self._members_to_spawn:
                setattr(self, _member, getattr(parent, _member))
        else:
            spawn_ids = []
            spawn_ids_gen = get_unique_ids(spawn_ids)

            self.patch_interp_type = next(spawn_ids_gen)
            self.n_init_shift = next(spawn_ids_gen)
            self._test_mode = next(spawn_ids_gen)
            self._tracker_info = next(spawn_ids_gen)
            self._members_to_spawn = next(spawn_ids_gen)
            self._members_to_copy = next(spawn_ids_gen)

            self._members_to_spawn = ids_to_member_names(self, spawn_ids)

            copy_ids = []
            copy_ids_gen = get_unique_ids(copy_ids, spawn_ids)

            self.max_iou_det_idx = next(copy_ids_gen)
            self.locations = next(copy_ids_gen)
            self.max_iou = next(copy_ids_gen)
            self.ratios = next(copy_ids_gen)
            self.max_iou_det_scores = next(copy_ids_gen)

            self._members_to_copy = ids_to_member_names(self, copy_ids)

            tracker = None

            self.patch_interp_type = CVConstants.interp_types[self._params.pattern_interp_type_id]
            # self.pattern_n_pix = self.params.pattern_shape[0] * self.params.pattern_shape[1]
            self.n_init_shift = min(len(self._params.init_shift), self._params.count - 1)
            self._test_mode = 0

            self._logger.info(f'count: {self._params.count}')
            tracker_type = str(self._params.tracker)
            tracker_types = paramparse.obj_from_docs(self._params, 'tracker')
            try:
                tracker_name = [k for k in tracker_types if tracker_type in tracker_types[k]][0]
            except IndexError:
                raise AssertionError('Invalid tracker_type: {}'.format(tracker_type))
            tracker_info_dict = {
                'LK': (LK, self._params.lk),
                'Siamese': (Siamese, self._params.siamese),
                'PYT': (PYT, self._params.pyt),
                'Identity': (IdentityTracker, self._params.idt),
                'GT': (GTTracker, self._params.gtt),
            }
            self._tracker_info = tracker_info_dict[tracker_name]
            self._logger.info(f'Using {tracker_name} tracker')

            if not self.ratio_heuristics:
                self._logger.info('ratio heuristics are disabled')

        """no. of templates"""
        self.count = self._params.count

        """only needed for efficiency :: pass along the standard BB location within the cropped patches along with
        number and size of patches to be tracked so that any pre-computations can be performed to reduce online work
        """
        # pre_cmpt_params = (self.std_box, self.count, self.roi_shape)
        update_location = self._params.update_location

        tracker_type, tracker_params = self._tracker_info  # type: type(TrackerBase), TrackerBase.Params
        tracker_params.roi = self._params.roi
        common_params = dict(
            params=tracker_params,
            rgb_input=self._rgb_input,
            update_location=update_location,
            n_templates=self.count,
            parent=tracker,
            logger=self._logger,
            policy_name=None
        )

        self.tracker = tracker_type(**common_params)

        # self.n_init_samples = self._tracker.n_init_samples

        """id of the main template"""
        self.anchor_id = 0

        """id of the template to be replaced by a new one in a keyframe"""
        self.replace_id = 0

        """locations of templates from current and past frames representing the target appearance over time"""
        self.locations = np.zeros((self._params.count, 4), dtype=np.float32)
        # self.transform = np.zeros((1, 4))
        self.locations_mat = np.zeros((self._params.count, 4), dtype=np.float32)
        """corresonding centroids"""
        self.centers = np.zeros((self._params.count, 2), dtype=np.float32)
        """frames from where the templates have been extracted"""
        self.frames = [None] * self._params.count
        """IDs of the frames from where the templates have been extracted"""
        self.frame_ids = np.zeros((self._params.count, 1), dtype=np.int32)
        """IDs of the latest frame from where a template has been extracted"""
        self.prev_frame_id = 0
        """tracking_scores of the extracted templates"""
        self.tracking_scores = np.zeros((self._params.count, 1), dtype=np.int32)
        """max overlap of each template with all detections in the frame where it was added"""
        # self.iou_when_added = np.ones((self._params.count, 1), dtype=np.float32)
        """max overlaps of each template with all the detections in each frame"""
        self.max_iou = np.zeros((self._params.count, 1), dtype=np.float32)

        """scores of the maximally overlapping detections"""
        self.max_iou_det_scores = np.zeros((self._params.count, 1), dtype=np.float32)
        """corresponding indices"""
        self.max_iou_det_idx = np.zeros((self._params.count, 1), dtype=np.int32)

        """angles between the speed vectors"""
        # self.angles = np.zeros((self.params.count, 1), dtype=np.float32)
        if self.ratio_heuristics:
            """ratios of bounding box heights between consecutive frames"""
            self.ratios = np.zeros((self._params.count,), dtype=np.float32)
        else:
            self.ratios = None

        if self._heuristics:
            """resized and centered patch corresponding to each template location for computing similarity"""
            if self._rgb_input:
                self._pattern_shape = (
                    self._params.count, self._params.pattern_shape[0], self._params.pattern_shape[1], 3)
            else:
                self._pattern_shape = (self._params.count, self._params.pattern_shape[0], self._params.pattern_shape[1])
            self._patterns = np.zeros(self._pattern_shape, dtype=np.float32)
        else:
            self._patterns = self._pattern_shape = None

        """flags indicating tracking success"""
        # self.flags = np.ones((self.params.count, 1), dtype=np.int32)

        """tracking features for all templates"""
        # self.features = np.zeros((self.params.count, self.n_features))

        """velocity of each template when tracked in the current frame"""
        # self.velocities = np.zeros((self._params.count, 1), dtype=np.float32)
        """mean inter frame velocity over all templates"""
        # self.mean_velocity = np.zeros((1, 2), dtype=np.float32)

        self.init_shift = np.array(self._params.init_shift, dtype=np.float32)

        self._pause = 0

    def train_mode(self):
        self._test_mode = 0

    def test_mode(self):
        self._test_mode = 1

    def initialize(self, target_id, frame_id, frame, location):
        """
        :type frame_id: int
        :type frame: np.ndarray
        :type location: np.ndarray
        :rtype: None
        """
        self._target_id = target_id
        self.tracker.set_id(self._target_id)

        if not self._test_mode:
            # self.iou_when_added.fill(1)
            self.max_iou.fill(0)
            # self.flags.fill(1)
            # self.features.fill(0)
            self.max_iou_det_scores.fill(0)
            self.max_iou_det_idx.fill(0)
            # self.angles.fill(0)

            if self.ratio_heuristics:
                self.ratios.fill(0)

        # print 'Templates :: initialize :: self.frame_size: ', self.frame_size

        self.locations[:] = np.tile(location, (self._params.count, 1))

        # print 'n_shifted_templates: ', n_shifted_templates

        # location_shift = np.multiply(self.locations[1:n_shifted_templates + 1, 2:],
        #                              self.init_shift[:n_shifted_templates, :])

        # location_shift = np.empty((n_shifted_templates, 2), dtype=np.float32)
        # for i in range(n_shifted_templates):
        #     location_shift[i, 0] = self.locations[0, 2]*self.init_shift[i, 0]
        #     location_shift[i, 1] = self.locations[0, 3]*self.init_shift[i, 1]

        self.locations[1:self.n_init_shift + 1, :2] += np.multiply(
            self.locations[1:self.n_init_shift + 1, 2:],
            self.init_shift[:self.n_init_shift, :])

        # temporary workaround for the insidious floating point error issue
        # self.locations[:] = np.around(self.locations, 2)

        self.locations_mat[:, :2] = self.locations[:, :2]
        self.locations_mat[:, 2:] = self.locations[:, :2] + self.locations[:, 2:] - 1

        self.frames = [frame, ] * self._params.count
        self.frame_ids.fill(frame_id)

        self.prev_frame_id = frame_id

        self.tracking_scores.fill(1)
        self.anchor_id = 0

        # frame_disp = np.copy(frame)
        # for i in range(self.params.count):
        # utils.drawBox(frame_disp, self.locations[i, :])
        # cv2.imshow('Templates::init::Frame', frame_disp)
        # if cv2.waitKey(0)==27:
        # exit()

        self.tracker.initialize(frame, self.locations)

        self.centers[:] = self.locations[:, :2] + (self.locations[:, 2:] - 1) / 2.0

        if self._heuristics:
            for patch_id in range(self._params.count):
                self.extract_pattern(self._patterns[patch_id, :], frame, self.locations[patch_id, :])

        if self._params.pause_for_debug:
            self._logger.debug('paused')

    def update(self, frame, frame_id, curr_det_data, location, tracking_scores, change_anchor):
        """
        Use the detections and tracked object location in the current frame to update the templates
        by replacing the worst or the least important template with the one extracted from the new location
         and optionally also set it to be the new anchor template

        :type frame: np.ndarray
        :type frame_id: int
        :type curr_det_data:
        :type location: np.ndarray
        :type tracking_scores: np.ndarray
        :type change_anchor: int
        """
        keyframe_gap = frame_id - self.prev_frame_id
        if keyframe_gap < self._params.min_keyframe_gap:
            if self._params.verbose:
                self._logger.debug(f'Skipping template updating in frame {frame_id} '
                                   f'due to insufficient keyframe gap {keyframe_gap} / {self._params.min_keyframe_gap}')
            return

        location = location.squeeze()
        # location_size = location[:2] + location[2:] - 1

        if len(tracking_scores) == self.count:
            """
            template that was tracked worst with either the predicted location (in tracked) or the 
            associated detection (in lost) is the one to be replaced with a new one extracted from this location
            """
            curr_tracking_score = tracking_scores[self.anchor_id]
            _tracking_scores = np.copy(tracking_scores)
            if not change_anchor:
                """not changing the anchor so template to be replaced must exclude it - dubious heuristic at best
                """
                _tracking_scores[self.anchor_id] = np.inf
            self.replace_id = np.argmin(_tracking_scores).item()
        elif len(tracking_scores) == 1:
            """continuous tracking mode or no feature summary
            
            replace template with least tracking_score in the frame where it was added
            """
            curr_tracking_score = tracking_scores[0]
            self.replace_id = np.argmin(self.tracking_scores).item()
        else:
            raise AssertionError(f'Invalid tracking_scores: {tracking_scores}')

        if change_anchor:
            self.anchor_id = self.replace_id

        # if self._params.pause_for_debug:
        #     self._logger.debug('paused')

        self.frames[self.replace_id] = frame
        self.frame_ids[self.replace_id] = frame_id
        self.tracking_scores[self.replace_id] = curr_tracking_score
        self.locations[self.replace_id, :] = location
        self.centers[self.replace_id, :] = location[:2] + (location[2:] - 1) / 2.0

        if self._heuristics:
            self.extract_pattern(self._patterns[self.replace_id, :], frame, location)

        self.tracker.update(self.replace_id, frame, location)

        if self._params.vis:
            self.tracker.get_stacked_roi(self.anchor_id, self.frame_ids, self.frames, self.locations)

        # scaling_factors = 1.0 / self.transform[1]
        # if curr_det_data.shape[0]:
        #     iou = np.empty((curr_det_data.shape[0], 1))
        #     compute_overlap(iou, None, None, location.reshape((1, 4)), curr_det_data[:, 2:6],
        #                     logger=self._logger, debug=self._params.pause_for_debug)
        # max_iou_idx = np.argmax(iou)
        # self.iou_when_added[self.replace_id] = iou[max_iou_idx]
        # self.indices[self.max_fb_idx] = max_iou_idx
        # self.scores[self.max_fb_idx] = curr_det_data[max_iou_idx, 6]
        # else:
        #     self.iou_when_added[self.replace_id] = 0
        # self.indices[self.max_fb_idx] = 0
        # self.scores[self.max_fb_idx] = -1

        self.prev_frame_id = frame_id

        # if self._params.pause_for_debug:
        #     self._logger.debug('paused')

    # def transform(self, out_locations, factor, patch_id=0):
    #     """
    #     convert tracked location of ROI into frame of reference of the input image
    #     :param out_locations: array to store result of transform
    #     :type out_locations: np.ndarray
    #     :param factor: factor by which to tansform as returned by extractROI
    #     :type factor: np.ndarray
    #     :param patch_id: optional ID if one of the multi track results are to be used
    #     :type patch_id: int | None
    #     :return: None
    #     """
    #     # self.logger.debug('transform')
    #     in_locations = self._tracker._locations[patch_id, :, :].squeeze()
    #
    #     out_locations[:, :2] = (in_locations[:, :2] + factor[:2]) * factor[2:]
    #     out_locations[:, 2:] = (in_locations[:, 2:] - 1) * factor[2:] + 1

    def apply_heuristics(self, frame_id, curr_det_data, locations, track_res,
                         check_ratio, track_id):
        """
        compare tracking result with detections to extract annoying heuristical features and
        compute the overall target location

        :type frame_id: int
        :param np.ndarray curr_det_data: only used for heuristical comparisons with tracking results
        :type locations: np.ndarray
        :type track_res: TrackerBase.Result
        :type track_id: int
        :type check_ratio: bool | int
        :rtype: None
        """

        """templates that satisfy certain coarse measures of tracking success - all BB values finite and positive"""
        valid_idx_bool = np.logical_and(np.isfinite(locations).all(axis=1), (locations[:, 2:] > 0).all(axis=1))

        if track_res.heuristics:
            tracker_status = track_res.get_status_m(track_id=track_id)
            valid_idx_bool = np.logical_and(
                valid_idx_bool,
                tracker_status
            )
        if self.ratio_heuristics:
            """Compare shapes and sizes of tracked boxes with the template boxes to find any
            sudden and drastic changes which would indicate tracking failure
            """
            self.ratios.fill(0)
            valid_idx = np.flatnonzero(valid_idx_bool)
            self.ratios[valid_idx] = locations[valid_idx, 3] / self.locations[valid_idx, 3]

            """ratio of the box with the lesser height to the one with the greater height
             so it is always between 0 and 1"""
            self.ratios[valid_idx] = np.minimum(self.ratios[valid_idx], np.reciprocal(self.ratios[valid_idx]))

            if check_ratio:
                """for some annoying reason, ratio is not checked during association"""
                temp = self.ratios >= self._params.min_ratio
                valid_idx_bool = np.logical_and(
                    valid_idx_bool, temp)

        valid_idx = np.flatnonzero(valid_idx_bool)
        invalid_idx = np.flatnonzero(np.logical_not(valid_idx_bool))

        if track_res.heuristics:
            track_res.set_status(valid_idx, invalid_idx, track_id)

        if valid_idx.size == 0:
            # locations.fill(np.NaN)
            self.max_iou.fill(0)
            self.max_iou_det_idx.fill(0)
            self.max_iou_det_scores.fill(0)
            # self.angles.fill(-1)
            return

        # locations[invalid_idx, :] = np.NaN
        self.max_iou[invalid_idx, :] = 0
        self.max_iou_det_idx[invalid_idx, :] = 0
        self.max_iou_det_scores[invalid_idx, :] = 0
        # self.angles[invalid_idx, :] = -1

        """
        annoying heuristics based on comparisons between tracking results and detections needed mainly to eliminate 
        tracked targets with no associated detections
        """

        if curr_det_data.shape[0] == 0:
            """no detections"""
            self.max_iou[valid_idx, :] = 0
            self.max_iou_det_idx[valid_idx, :] = 0
            self.max_iou_det_scores[valid_idx, :] = -1
        elif curr_det_data.shape[0] == 1:
            """single detection"""
            iou = np.empty((valid_idx.size, 1))
            compute_overlap(iou, None, None, curr_det_data[0, 2:6].reshape((1, 4)),
                            locations[valid_idx, :], self._logger)
            self.max_iou_det_idx[valid_idx, :] = 0
            self.max_iou[valid_idx, :] = iou
            self.max_iou_det_scores[valid_idx, :] = curr_det_data[0, 6]
        else:
            """get detection with maximum overlap with the tracked location of each template"""
            iou = np.empty((curr_det_data.shape[0], valid_idx.size))
            # self.logger.debug(f'curr_det_data: {curr_det_data}')
            compute_overlaps_multi(iou, None, None, curr_det_data[:, 2:6],
                                   locations[valid_idx, :], self._logger)
            self.max_iou_det_idx[valid_idx, :] = np.argmax(iou, axis=0).transpose().reshape((valid_idx.size, 1))
            for i in range(valid_idx.size):
                idx = valid_idx[i]
                self.max_iou[idx, :] = iou[self.max_iou_det_idx[idx], i]
            self.max_iou_det_scores[valid_idx, :] = curr_det_data[self.max_iou_det_idx[valid_idx], 6]

        if frame_id in self.frame_ids:
            return

        """
        Amazingly annoying velocity based heuristics along with some angles that are not even used
        """
        # if self._params.velocity_heuristics:
        #     self._compute_mean_velocity()
        #     norm_mean_velocity = np.linalg.norm(self.mean_velocity)
        #     if norm_mean_velocity <= self._params.min_velocity_norm:
        #         self.angles[valid_idx, :] = 1
        #         return
        #     centers = (locations[valid_idx, :2] + (locations[valid_idx, 2:] - 1) / 2.0).reshape((-1, 2))
        #     velocities = (centers - self.centers[valid_idx, :].reshape((-1, 2))) / \
        #                  (frame_id - self.frame_ids[valid_idx].reshape((-1, 1)))
        #     norm_velocities = np.linalg.norm(velocities, axis=1).reshape((-1, 1))
        #
        #     vel_valid_idx_bool = norm_velocities > self._params.min_velocity_norm
        #     vel_valid_idx = np.flatnonzero(vel_valid_idx_bool)
        #     if vel_valid_idx.size == 0:
        #         self.angles[valid_idx, :] = 1
        #     else:
        #         a1 = velocities[vel_valid_idx, :].reshape((-1, 2))
        #         a2 = a1 * self.mean_velocity
        #         a = np.sum(a2, axis=1).reshape((-1, 1))
        #         b = norm_mean_velocity * norm_velocities[vel_valid_idx, :].reshape((-1, 1))
        #         self.angles[valid_idx[vel_valid_idx], :] = (a / b).reshape((-1, 1))
        #
        #         vel_invalid_idx = np.flatnonzero(np.logical_not(vel_valid_idx_bool))
        #         self.angles[valid_idx[vel_invalid_idx], :] = 1

        # if self._params.pause_for_debug:
        #     self._logger.debug('paused')

    def get_similarity(self, similarity, frame, location, similarity_type):
        """
        Compute a similarity metric for all template patches with respect to the given location
        Apparently this is updated even when the target has been decided to be in the lost state
        also, this thing is only used as one of the heuristic features in the lost state

        :param np.ndarray similarity:
        :param np.ndarray frame:
        :param np.ndarray location:
        :param int similarity_type:
        :return:
        """

        if np.isfinite(location).all():
            pattern_shape = (1,) + self._pattern_shape[1:]
            pattern = np.zeros(pattern_shape, dtype=np.float32).squeeze()
            self.extract_pattern(pattern, frame, location)
            if similarity_type == PY_DIST:
                similarity[:] = pyDist.get(pattern, self._patterns, 1)
            else:
                for i in range(self.count):
                    similarity[i] = cv2.matchTemplate(
                        self._patterns[i, ...].squeeze(), pattern, method=similarity_type)
        else:
            similarity.fill(0)

    def extract_pattern(self, pattern, frame, location):

        # if len(frame.shape) == 3:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        location = location.squeeze()
        if np.equal(np.mod(location, 1), 0).all():
            smin = np.maximum(0, location[0:2]).reshape((2,)).astype(np.int32)
            smax = np.minimum(np.array(frame.shape[:2])[::-1],
                              location[0:2] + location[2:4]).reshape((2,)).astype(np.int32)

            patch = frame[smin[1]:smax[1], smin[0]:smax[0], ...].astype(np.float32)
        else:

            center = location[0:2] + (location[2:4] - 1) / 2.0 - 1

            if self._params.sub_pix_method == 0:
                H = np.array([
                    [1, 0, -center[0]],
                    [0, 1, -center[1]],
                    [0, 0, 1]
                ], dtype=np.float32)
                H_inv = linalg.inv(H)
                half_w = location[2] / 2.0
                half_h = location[3] / 2.0
                box = np.array([-half_w, half_w, -half_h, half_h], dtype=np.float32)
                patch = pyWarp.get(frame, H_inv, box).astype(np.float32)
            else:
                patch = cv2.getRectSubPix(frame, (int(location[2]), int(location[3])), tuple(center))
                if patch is not None:
                    patch = patch.astype(np.float32)

        if patch is not None:
            if self._params.vis == 2:
                patch_uint8 = self.round(patch).astype(np.uint8)
                pt1 = (int(location[0]), int(location[1]))
                pt2 = (int(location[0] + location[2]),
                       int(location[1] + location[3]))
                frame_disp = np.copy(frame)
                cv2.rectangle(frame_disp, pt1, pt2, (0, 255, 0), thickness=2)
                cv2.imshow('patch_uint8', patch_uint8)
                cv2.imshow('frame_disp', frame_disp)

            resized_patch = cv2.resize(patch, dsize=self._params.pattern_shape[::-1],
                                   interpolation=self.patch_interp_type)
            pattern[:] = (resized_patch - resized_patch.mean())
        else:
            pattern[:] = 0

    def copy(self):
        """
        :rtype: dict
        """
        obj_dict = {}
        for _member in self._members_to_copy:
            obj_dict[_member] = np.copy(getattr(self, _member))
        obj_dict['tracker'] = self.tracker.copy()
        return obj_dict

    def restore(self, obj_dict, deep_copy=False):
        """
        :type obj_dict: dict
        :type deep_copy: bool
        """

        if deep_copy:
            for _member in self._members_to_copy:
                setattr(self, _member, np.copy(obj_dict[_member]))
        else:
            for _member in self._members_to_copy:
                setattr(self, _member, obj_dict[_member])

        self.tracker.restore(obj_dict['tracker'], deep_copy)

    @contextmanager
    def immutable(self, enabled=True):
        if enabled:

            """original code copied templates state matrices before getting learning features
            for some unknown annoying reason
            """
            templates_copy = self.copy()

            yield None

            """really annoying and insidious part of the original code where the tracker is simply
            not updated in this particular case for some completely unknown reason
            """
            self.restore(templates_copy)

        else:
            yield None

    # def _compute_mean_velocity(self):
    #     unique_frame_ids = np.unique(self.frame_ids)
    #     n_ids = unique_frame_ids.size
    #     if n_ids > self._params.max_velocity_frames:
    #         unique_frame_ids = unique_frame_ids[-self._params.max_velocity_frames:]
    #         n_ids = self._params.max_velocity_frames
    #     centers = np.zeros((n_ids, 2))
    #     for i in range(n_ids):
    #         idx = np.flatnonzero(self.frame_ids == unique_frame_ids[i])
    #         if idx.size > 1:
    #             centers[i, :] = np.mean(
    #                 self.locations[idx, 0:2] + (self.locations[idx, 2:4] - 1) / 2.0, axis=0)
    #         else:
    #             centers[i, :] = self.locations[idx, 0:2] + (self.locations[idx, 2:4] - 1) / 2.0
    #     if n_ids > 1:
    #         velocities = (centers[1:, :] - centers[:-1, :]) / (unique_frame_ids[1:] - unique_frame_ids[:-1]).reshape(
    #             (n_ids - 1, 1))
    #         self.mean_velocity[:] = np.mean(velocities, axis=0)
    #     else:
    #         self.mean_velocity[:] = 0

    # def write_state_info(self, files, root_dir, write_to_bin, write_roi=True, write_patterns=False,
    #                      fp_fmt='%.4f', fp_dtype=np.float32, include_tracker=1):
    #     """
    #     :type files: list[(str, Type(np.dtype), tuple)]
    #     :type root_dir: str
    #     :type write_to_bin: bool | int
    #     :type write_roi: bool | int
    #     :type write_patterns: bool | int
    #     :type fp_fmt: str
    #     :type fp_dtype: Type(np.dtype)
    #     :type include_tracker: int
    #     :rtype: None
    #     """
    #
    #     if include_tracker:
    #         self._tracker.write_state_info(files, root_dir, write_to_bin, fp_fmt=fp_fmt, fp_dtype=fp_dtype)
    #
    #     log_dir = '{:s}/templates'.format(root_dir)
    #     entries = [
    #         (self.max_iou_det_idx, 'indices', np.uint32, '%d'),
    #         (self.locations, 'locations', fp_dtype, fp_fmt),
    #         (self.max_iou, 'overlaps', fp_dtype, fp_fmt),
    #         (self.ratios, 'ratios', fp_dtype, fp_fmt),
    #         # (self.angles, 'angles', fp_dtype, fp_fmt),
    #         (self.iou_when_added, 'bb_overlaps', fp_dtype, fp_fmt),
    #         (self.max_iou_det_scores, 'scores', fp_dtype, fp_fmt),
    #         # (self.features, 'features', fp_dtype, fp_fmt)
    #     ]
    #     if write_roi:
    #         n_roi = self.roi.shape[0]
    #         roi_size = self.roi.shape[1] * self.roi.shape[2]
    #         roi_flattened = np.empty((roi_size, n_roi), dtype=np.uint8)
    #         for i in range(n_roi):
    #             roi_flattened[:, i] = self.roi[i, :, :].flatten(order='F')
    #         entries.append((roi_flattened, 'roi', np.uint8, '%d'))
    #
    #     if write_patterns:
    #         entries.append((self._patterns.transpose(), 'patterns', fp_dtype, fp_fmt))
    #
    #     write_to_files(log_dir, write_to_bin, entries)
    #     # files.extend(['templates/{:s}'.format(entry[1]) for entry in entries])
    #     files.extend([('templates/{:s}'.format(entry[1]), entry[2], entry[0].shape) for entry in entries])
