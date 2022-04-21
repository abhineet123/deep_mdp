import numpy as np
import scipy.spatial
import functools
import cv2

from trackers.tracker_base import TrackerBase
from utilities import stack_images_1D, TrackingStatus, draw_pts, draw_box, stack_images, \
    annotate_and_show


class LK(TrackerBase):
    """
    :type params: LostParams
    :type _n_templates: int
    :type logger: logging.RootLogger
    :type n_features: int
    :type _n_pts: int
    :type _template_bbox: np.ndarray
    :type _lk_out: list[np.ndarray]
    :type lk_out: list[list[np.ndarray]] | None
    :type similarity_type: int
    :type _std_box_center: np.ndarray
    :type _template_pts: np.ndarray
    :type features: np.ndarray
    :type locations: np.ndarray
    :type shifts: np.ndarray
    :type flags: np.ndarray
    :type features: np.ndarray
    :type locations: np.ndarray
    :type shifts: np.ndarray
    :type flags: np.ndarray
    """

    class Params(TrackerBase.Params):
        """
        :ivar margin_box: '[width height] in pixels of the border around the object bounding box within which '
                      'optical flow is also computed',
        :ivar grid_res: 'resolution of the grid of points where LK optical flow is computed; '
                    'e.g. a resolution of 10 means that the object patch is sampled by a 10x10 grid of'
                    ' equally spaced points where the optical flow is computed',
        :ivar max_iters: 'maximum no. of iterations of the optical flow process per frame',
        :ivar eps: 'threshold of change in LK estimate that is used for terminating the iterative process',
        :ivar level_track: 'no. of pyramidal levels to use',
        :ivar lk_win_size: 'size of the neighborhood around each point whose pixel values are used for '
                       'computing the optical flow estimate',
        :ivar ncc_win_size: 'size of sub patches around each point whose similarities (NCC) are computed',
        :ivar fb_thresh: 'forward-backward error threshold',
        :ivar fb_norm_factor: 'normalization factor for computing features from optical flow forward-backward error',
        :ivar cv_wrapper: 'use the OpenCV python wrapper for LK and NCC computation instead '
                      'of the custom implementation; this is usually slower than the custom version',
        :ivar stacked: 'use stacked implementation of LK - here all patches are stacked onto a single large image '
                   'and pairwise LK is computed in a single call to the OpenCV LK function; this is usually faster'
                   ' especially when there are many objects / detections in the scene',
        :ivar gpu: 'use GPU implementation of LK; this is only supported if a modern GPU with CUDA support '
               'is available and OpenCV is compiled with CUDA support; ',
        :ivar show_points: 'show the optical flow points for each tracked patch',

        :ivar verbose: 'print detailed information'

        """

        def __init__(self):
            TrackerBase.Params.__init__(self)

            self.margin_box = (5, 2)  # [width height] of the margin in computing flow
            self.grid_res = 10

            self.max_iters = 20
            self.eps = 0.03
            self.pre_pyr = 0
            self.level_track = 1
            self.lk_win_size = 4
            self.ncc_win_size = 10

            self.fb_thresh = 10
            self.fb_norm_factor = 30

            self.cv_wrapper = 1
            self.stacked = 0
            self.gpu = 0
            self.show_points = 0
            self.verbose = 1
            self.vis = 0

            self.pause_for_debug = 0

    def __init__(self, **kwargs):
        """
        :type params: LK.Params
        :type logger: logging.RootLogger
        :type parent: LK  | TrackerBase | None
        :rtype: None
        """
        # assert pre_cmpt_params is not None, "pre_cmpt_params must be provided"

        TrackerBase.__init__(self, 'lk', **kwargs)

        assert self._params.roi.enable, "Tracking in ROI free mode is not supported yet"

        if self._parent is None:
            self._n_features = next(self._spawn_ids_gen)
            self._n_pts = next(self._spawn_ids_gen)
            self._std_box_center = next(self._spawn_ids_gen)
            self._template_pts = next(self._spawn_ids_gen)
            self._left_half_idx = next(self._spawn_ids_gen)
            self._right_half_idx = next(self._spawn_ids_gen)
            self._top_half_idx = next(self._spawn_ids_gen)
            self._bottom_half_idx = next(self._spawn_ids_gen)
            self._start_ids = next(self._spawn_ids_gen)
            self._end_ids = next(self._spawn_ids_gen)
            self._x_offsets = next(self._spawn_ids_gen)

            self.locations = next(self._copy_ids_gen)
            self._templates = next(self._copy_ids_gen)
            self._features = next(self._copy_ids_gen)
            self._shifts = next(self._copy_ids_gen)
            self._status = next(self._copy_ids_gen)
            self._lk_out = next(self._copy_ids_gen)

            self._register()
        else:
            self._spawn()

        """needed for intellisense"""
        self._params = kwargs["params"]  # type: LK.Params

        # self._create_object = functools.partial(LK, **kwargs)

        self._pause = 1
        # self.lk_in = lk

        # choose tracker implementation
        if self._params.stacked:
            self._track_roi = self.__track_stacked
        else:
            self._track_roi = self.__track

        if self._params.cv_wrapper:
            pts_dtype = np.float32
            self._lk = self._get
        else:
            """has to be 64 bit due to some annoying PyArrayObject crap"""
            pts_dtype = np.float64
            if self._params.gpu:
                try:
                    import pyLKGPU as pyLK
                except ImportError as e:
                    raise ImportError('pyLKGPU unavailable: {}'.format(e))

                pyLK.initialize(self._params.level_track, self._params.lk_win_size,
                                self._params.max_iters, self._params.ncc_win_size, self._params.show_points)
                self._lk = pyLK.get
                # self._logger.info('Using pyLKGPU')
            else:
                try:
                    import pyLK
                except BaseException as e:
                    raise ImportError('pyLK unavailable: {}'.format(e))
                pyLK.initialize(self._params.level_track, self._params.lk_win_size, self._params.max_iters,
                                self._params.eps, self._params.ncc_win_size, self._params.show_points)
                self._lk = pyLK.get
                # self._logger.info('Using pyLK')

        if self._parent is None:

            if self._params.use_gt:
                self._logger.warning('getting object locations from GT when available')

            self._n_features = 6

            if self._params.feature_type == 0:
                self.feature_shape = (self._n_features,)
            elif self._params.feature_type != - 1:
                raise AssertionError(f'invalid feature_type: {self._params.feature_type}')

            if self._params.cv_wrapper:
                self._logger.info('Using cv wrapper')

            self._n_pts = self._params.grid_res ** 2

            if self._params.verbose:
                if self._params.stacked:
                    self._logger.info('Using stacked version')

            self._std_box_center = self._std_box[0, 0:2] + self._std_box[0, 2:4] / 2.0

            x = np.linspace(self._std_box[0, 0] + self._params.margin_box[0],
                            self._std_box[0, 0] + self._std_box[0, 2] - 1 - self._params.margin_box[0],
                            self._params.grid_res, dtype=pts_dtype)
            y = np.linspace(self._std_box[0, 1] + self._params.margin_box[1],
                            self._std_box[0, 1] + self._std_box[0, 3] - 1 - self._params.margin_box[1],
                            self._params.grid_res, dtype=pts_dtype)
            xx, yy = np.meshgrid(x, y)
            self._template_pts = np.concatenate(
                (xx.flatten(order='F').reshape((self._n_pts, 1)),
                 yy.flatten(order='F').reshape((self._n_pts, 1))),
                axis=1)

            self._left_half_idx = np.flatnonzero(np.less(self._template_pts[:, 0], self._std_box_center[0]))
            self._right_half_idx = np.flatnonzero(
                np.greater_equal(self._template_pts[:, 0], self._std_box_center[0]))
            self._top_half_idx = np.flatnonzero(np.less(self._template_pts[:, 1], self._std_box_center[1]))
            self._bottom_half_idx = np.flatnonzero(
                np.greater_equal(self._template_pts[:, 1], self._std_box_center[1]))

            self._start_ids = [0] * self._n_templates
            self._end_ids = [0] * self._n_templates
            self._x_offsets = [0] * self._n_templates
            for i in range(self._n_templates):
                self._start_ids[i] = i * self._n_pts
                self._end_ids[i] = self._start_ids[i] + self._n_pts
                self._x_offsets[i] = i * self._roi_shape[1]

        self.locations = None

        self._templates = [None] * self._n_templates

        self._features = None
        self._shifts = None
        self._status = None
        self._lk_out = None

        self.__features = None
        self.__status = None

    def _initialize_roi(self, template_id, roi):
        """

        :param int template_id:
        :param np.ndarray roi:
        :return:
        """
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self._templates[template_id] = roi

    def _update_roi(self, template_id, roi):
        """

        :param int template_id:
        :param np.ndarray roi:
        :return:
        """
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        self._templates[template_id] = roi

    def get_init_samples(self):
        assert not self._patch_as_features, "init samples are only supported with heuristic features"

        heuristic_features = np.concatenate(
            (np.ones((1, self._n_features), dtype=np.float32),
             np.zeros((1, self._n_features), dtype=np.float32)),
            axis=0)
        labels = np.array((1, -1), dtype=np.float32)

        return heuristic_features, labels

    def __track(self, patches):
        """
        track each template into each patch in the given set of of patches - highly parallelizable process

        :param patches:
        :param heuristics:
        :return:
        :rtype LK.Result
        """

        n_patches_2 = patches.shape[0]

        self.locations = np.zeros((n_patches_2, self._n_templates, 4))

        self._features = np.zeros((n_patches_2, self._n_templates, self._n_features))
        self._shifts = np.zeros((n_patches_2, self._n_templates, 2))
        self._status = np.zeros((n_patches_2, self._n_templates, 1), dtype=np.uint8)
        self._lk_out = [[None] * self._n_templates] * n_patches_2

        all_disp_images = []

        patches_list = []

        for patch_id in range(n_patches_2):
            patch = np.squeeze(patches[patch_id, ...])
            if len(patch.shape) == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            patches_list.append(patch)

            for template_id in range(self._n_templates):
                # compute optical flow

                template = self._templates[template_id]  # type: np.ndarray

                # template = patches_1[template_id, :, :].squeeze()
                temp = self._lk(template, patch, self._template_pts, self._template_pts)

                self._lk_out[patch_id][template_id] = temp
                self._process(self._lk_out[patch_id][template_id], self.locations[patch_id, template_id, :],
                              self._features[patch_id, template_id, :],
                              self._shifts[patch_id, template_id, :],
                              self._status[patch_id, template_id, :])
                if self._params.vis:
                    patch_disp = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                    template_disp = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                    location = self.locations[patch_id, template_id, :].squeeze()
                    status = self._status[patch_id, template_id, :].item()
                    if np.any(np.isnan(location)):
                        self._logger.info('skipping drawing invalid location: {}'.format(location))
                    else:
                        if status == TrackingStatus.failure:
                            col = 'red'
                        else:
                            col = 'green'
                        draw_box(patch_disp, location, color=col)

                    cmb_img = stack_images_1D([template_disp, patch_disp])
                    all_disp_images.append(cmb_img)

        if self._params.vis:
            self._pause = annotate_and_show('lk', all_disp_images, pause=self._pause)

        self.__features = np.copy(self._features)
        self.__status = np.copy(self._status)

        is_roi = 1
        if self._params.use_gt:
            gt_location, locations = self.locations_from_gt(n_patches_2)
            if gt_location is not None:
                self.locations = locations
                is_roi = 0

        track_res = LK.Result(self.locations, self._features, self._status, self._params.fb_norm_factor, is_roi)
        return track_res

    def __track_stacked(self, patches):
        """
         track each patch in the template set into each patch in the provided set
         highly parallelizable process

        :param patches:
        :return:
        :rtype LK.Result
        """
        n_patches = patches.shape[0]

        # patches_1_list =  self._templates
        # patches_1_list = [np.squeeze(patches_1[i, :, :]) for i in range(n_patches_1)]
        templates_stacked = stack_images_1D(self._templates, stack_order=0)
        templates_stacked_list = [templates_stacked] * n_patches
        templates_stacked_2 = stack_images_1D(templates_stacked_list, stack_order=1)

        patches_list = [np.squeeze(patches[i, ...]) for i in range(n_patches)]
        if len(patches_list[0].shape) == 3:
            patches_list = [cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) for patch in patches_list]
        patches_stacked = stack_images_1D(patches_list, stack_order=1)
        patches_stacked_list = [patches_stacked] * self._n_templates
        patches_stacked_2 = stack_images_1D(patches_stacked_list, stack_order=0)

        n_pairs = n_patches * self._n_templates
        std_pts_stacked = np.empty((self._n_pts * n_pairs, 2), dtype=np.float64)

        y_offsets = [0] * n_patches
        start_ids = [0] * n_pairs
        end_ids = [0] * n_pairs
        y_offset = 0
        idx = 0
        start_id = 0
        for p2 in range(n_patches):
            y_offsets[p2] = y_offset
            for p1 in range(self._n_templates):
                start_ids[idx] = start_id
                end_ids[idx] = start_ids[idx] + self._n_pts
                std_pts_stacked[start_ids[idx]:end_ids[idx], :] = \
                    self._template_pts + [self._x_offsets[p1], y_offsets[p2]]
                start_id = end_ids[idx]
                idx += 1
            y_offset += self._roi_shape[0]

        lk_out_stacked = self._lk(templates_stacked_2, patches_stacked_2, std_pts_stacked, std_pts_stacked)

        self._features = np.zeros((n_patches, self._n_templates, self._n_features))
        self.locations = np.zeros((n_patches, self._n_templates, 4))
        self._shifts = np.zeros((n_patches, self._n_templates, 2))
        self._status = np.zeros((n_patches, self._n_templates, 1), dtype=np.uint8)
        self._lk_out = [[None] * self._n_templates] * n_patches

        idx = 0
        for p2 in range(n_patches):
            for p1 in range(self._n_templates):
                lk_out = lk_out_stacked[start_ids[idx]:end_ids[idx], :]
                lk_out[:, :2] -= [self._x_offsets[p1], y_offsets[p2]]
                self._process(lk_out, self.locations[p2, p1, :], self._features[p2, p1, :],
                              self._shifts[p2, p1, :], self._status[p2, p1, :])
                self._lk_out[p2][p1] = lk_out
                idx += 1

        self.__features = np.copy(self._features)
        self.__status = np.copy(self._status)

        is_roi = 1
        if self._params.use_gt:
            gt_location, locations = self.locations_from_gt(n_patches)
            if gt_location is not None:
                self.locations = locations
                is_roi = 0

        track_res = LK.Result(self.locations, self._features, self._status, self._params.fb_norm_factor, is_roi)
        return track_res

    def _process(self, lk_out, location, features, shifts, status):

        valid_idx_bool = np.isfinite(lk_out[:, 0])
        valid_idx = np.flatnonzero(valid_idx_bool)

        if valid_idx.size == 0:
            status[:] = TrackingStatus.failure
            return

        features[0] = np.median(lk_out[valid_idx, 2])  # FB
        features[5] = np.median(lk_out[valid_idx, 3])  # NCC

        # compute bounding box location
        reliable_pts_idx = valid_idx[np.flatnonzero(np.logical_and(lk_out[valid_idx, 2] <= features[0],
                                                                   lk_out[valid_idx, 3] >= features[5]))]
        if reliable_pts_idx.size <= 1:
            # there must be at least two reliable points for the scaling factor to be computed from pairwise distances
            status[:] = TrackingStatus.failure
            return

        left_half_valid_idx = self._left_half_idx[np.flatnonzero(valid_idx_bool[self._left_half_idx])]
        right_half_valid_idx = self._right_half_idx[np.flatnonzero(valid_idx_bool[self._right_half_idx])]
        top_half_valid_idx = self._top_half_idx[np.flatnonzero(valid_idx_bool[self._top_half_idx])]
        bottom_half_valid_idx = self._bottom_half_idx[np.flatnonzero(valid_idx_bool[self._bottom_half_idx])]

        # region wise FB
        if left_half_valid_idx.size:
            features[1] = np.median(lk_out[left_half_valid_idx, 2])
        if right_half_valid_idx.size:
            features[2] = np.median(lk_out[right_half_valid_idx, 2])
        if top_half_valid_idx.size:
            features[3] = np.median(lk_out[top_half_valid_idx, 2])
        if bottom_half_valid_idx.size:
            features[4] = np.median(lk_out[bottom_half_valid_idx, 2])

        good_pts = self._template_pts[reliable_pts_idx, :]
        good_pts_fwd = lk_out[reliable_pts_idx, :2]

        pdist_1 = scipy.spatial.distance.pdist(good_pts, 'euclidean')
        pdist_2 = scipy.spatial.distance.pdist(good_pts_fwd, 'euclidean')

        s = np.median(pdist_2 / pdist_1)
        shifts[:] = 0.5 * (s - 1) * self._std_box[0, 2:4].reshape((1, 2))
        pts_diff = np.median(good_pts_fwd - good_pts, axis=0).reshape((1, 2))

        location[:2] = self._std_box[0, :2].reshape((1, 2)) + pts_diff - shifts
        location[2:] = self._std_box[0, 2:].reshape((1, 2)) + 2 * shifts

        """
        ul should be compared with 0 but using 1 to ensure consistency with the original code;
        similarly br should be >= patch_br to be outside but using > for consistency;
        only one of the many many annoying bugs in the original code
        """

        """compute tracking success indicator flag"""
        if (not np.isfinite(location).all()) or \
                np.less(location[:2], 1).any() or \
                np.greater(location[:2] + location[2:] - 1, self._roi_size).any():
            # unsuccessful
            status[:] = TrackingStatus.failure
        elif features[0] > self._params.fb_thresh:
            # unstable
            status[:] = TrackingStatus.unstable
        else:
            # successful
            status[:] = TrackingStatus.success

        if self._params.pause_for_debug:
            self._logger.debug('Done')

    def _get(self, patch_1, patch_2, pts_fwd, pts_bwd):

        pts_fwd = np.copy(pts_fwd).reshape((self._n_pts, 1, 2))
        pts_bwd = np.copy(pts_bwd).reshape((self._n_pts, 1, 2))

        # forward optical flow
        lk_out = np.zeros((self._n_pts, 4), dtype=np.float32)

        status_fwd = np.zeros((self._n_pts, 1), dtype=np.uint8)
        error_fwd = np.zeros((self._n_pts, 1), dtype=np.float32)

        # backward optical flow
        status_bwd = np.zeros((self._n_pts, 1), dtype=np.uint8)
        error_bwd = np.zeros((self._n_pts, 1), dtype=np.float32)

        # overall status - mark the success of both forward and backward OF
        status = np.zeros((self._n_pts, 1), dtype=np.bool)

        # feature_params = dict(maxCorners=100,
        #                       qualityLevel=0.3,
        #                       minDistance=7,
        #                       blockSize=7)
        # p0 = cv2.goodFeaturesToTrack(patch_1, mask=None, **feature_params)

        patch_1_pyr = patch_1.astype(np.uint8)
        patch_2_pyr = patch_2.astype(np.uint8)

        if self._params.pre_pyr and self._params.level_track > 0:
            """using pre computing pyramids in calcOpticalFlowPyrLK gives error due to a bug in opencv"""
            pyr_params = dict(
                winSize=(self._params.lk_win_size, self._params.lk_win_size),
                maxLevel=self._params.level_track,
                withDerivatives=1
            )
            _, patch_1_pyr = cv2.buildOpticalFlowPyramid(patch_1_pyr, **pyr_params)
            _, patch_2_pyr = cv2.buildOpticalFlowPyramid(patch_2_pyr, **pyr_params)

            # patch_1_pyr = tuple(patch_1_pyr)
            # patch_2_pyr = tuple(patch_2_pyr)

        lk_params = dict(
            winSize=(self._params.lk_win_size, self._params.lk_win_size),
            maxLevel=self._params.level_track,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, self._params.max_iters, self._params.eps),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)

        cv2.calcOpticalFlowPyrLK(patch_1_pyr, patch_2_pyr,
                                 self._template_pts.astype(np.float32),
                                 nextPts=pts_fwd, status=status_fwd, err=error_fwd, **lk_params)
        cv2.calcOpticalFlowPyrLK(patch_2_pyr, patch_1_pyr, pts_fwd,
                                 nextPts=pts_bwd, status=status_bwd, err=error_bwd, **lk_params)

        """turns out that the original mdp is just using status_bwd instead of the 'and' of the two - probably a bug"""
        # status[patch_id, :, :] = np.logical_and(status_fwd[patch_id, :, :],
        #                                              status_bwd[patch_id, :, :])
        status[:, :] = status_bwd[:, :]

        pts_fwd = pts_fwd.squeeze()
        pts_bwd = pts_bwd.squeeze()

        """compute features"""
        lk_out[:, 2] = np.linalg.norm(pts_fwd - pts_bwd.squeeze(), axis=1).reshape((self._n_pts,))
        for pt_id in range(self._n_pts):
            if status[pt_id]:
                sub_patch_1 = cv2.getRectSubPix(patch_1, (self._params.ncc_win_size, self._params.ncc_win_size),
                                                tuple(self._template_pts[pt_id, :].squeeze()))
                sub_patch_2 = cv2.getRectSubPix(patch_2, (self._params.ncc_win_size, self._params.ncc_win_size),
                                                tuple(pts_fwd[pt_id, :].squeeze()))
                result = cv2.matchTemplate(sub_patch_1, sub_patch_2, method=cv2.TM_CCOEFF_NORMED)
                lk_out[pt_id, :2] = pts_fwd[pt_id, :]
                lk_out[pt_id, 3] = result[0]
            else:
                lk_out[pt_id, :] = np.nan

        if self._params.show_points:
            _patch = np.copy(patch_1)
            draw_pts(_patch, pts_bwd)
            cv2.imshow('lk patch_1', _patch)

            _template = np.copy(patch_2)
            draw_pts(_template, pts_fwd)
            cv2.imshow('lk patch_2', _template)

            k = cv2.waitKey(1 - self._pause)
            if k == 27:
                cv2.destroyWindow('lk patch_1')
                cv2.destroyWindow('lk patch_2')
                exit()
            elif k == 32:
                self._pause = 1 - self._pause

        return lk_out

    def _vis(self, patch_1, patch_2):
        # np.savetxt('patch_1.txt', patch_1, delimiter='\t', fmt='%d')
        # np.savetxt('patch_2.txt', patch_2, delimiter='\t', fmt='%d')
        # np.savetxt('pts_1.txt', self.std_pts, delimiter='\t', fmt='%.4f')
        # np.savetxt('pts_2.txt', self.pts_fwd[patch_id, :, :].squeeze(), delimiter='\t', fmt='%.4f')

        img_list = np.empty((2, patch_1.shape[0], patch_1.shape[1]), dtype=np.uint8)
        img_list[0, ...] = patch_1
        img_list[1, ...] = patch_2
        stacked_patch = stack_images(img_list)
        cv2.imshow('stacked_patch', stacked_patch)
        if cv2.waitKey(0) == 27:
            exit(0)

    class Result(TrackerBase.Result):
        def __init__(self, locations, features, status, fb_norm_factor, is_roi):
            TrackerBase.Result.__init__(self, locations, features, status, None, None, is_roi)
            self._fb_norm_factor = fb_norm_factor
            self._heuristic_features = features
            self._n_heuristic_features = features.shape[-1]
            self.is_roi = is_roi

        def get_summarized_features(self, template_ids, track_id):
            # if template_ids is None:
            #     template_ids = self.get_success_ids(track_id)

            features = TrackerBase.Result.get_summarized_features(self, template_ids, track_id)
            if features is not None:
                return features

            features = np.zeros((1, self._n_heuristic_features))
            valid_features = self._heuristic_features[track_id, template_ids, :]

            # print 'k.shape: ', k.shape
            # print 'km.shape: ', km.shape
            # print 'self.distances: ', self.distances

            features[0, :5] = np.mean(
                np.exp(-valid_features[:, :5] / self._fb_norm_factor),
                axis=0)
            features[0, 5] = np.mean(valid_features[:, 5])

            return features

        def get_status_m(self, track_id):
            return np.isfinite(self._heuristic_features[track_id, :, :].reshape(
                (-1, self._n_heuristic_features))).all(axis=1)

        def set_status(self, valid_idx, invalid_idx, track_id=0):
            # self.flags[valid_idx, :] = flags[valid_idx].reshape((valid_idx.size, 1))
            # self.features[valid_idx, :] = features[valid_idx, :]

            if valid_idx.size == 0:
                self.status[track_id, :, :].fill(TrackingStatus.failure)
                self._heuristic_features[track_id, :, :5] = np.inf
                self._heuristic_features[track_id, :, 5] = 0
            else:
                self.status[track_id, invalid_idx, :] = TrackingStatus.failure
                self._heuristic_features[track_id, invalid_idx, :5] = np.inf
                self._heuristic_features[track_id, invalid_idx, 5] = 0

        def _get_best_template(self, template_ids, track_id):
            """
            Returns the ID of the best tracked patch
            :rtype: int
            """
            # object with the minimum FB error is the best one

            if template_ids is None:
                return np.argmin(self._heuristic_features[track_id, :, 0])

            best_template_id = np.argmin(self._heuristic_features[track_id, template_ids, 0])
            best_template = template_ids[best_template_id]
            return best_template

        def get_scores(self, track_id):
            """
            negative FB error represents tracking scores

            :param int track_id:
            :return:
            """

            return -self._heuristic_features[track_id, :, 0]

        # def get_worst_id(self, exclude_id=None, track_id=0):
        #     """
        #     Returns the ID of the worst tracked patch with an optional exclusion
        #
        #     :param exclude_id: optional ID of the patch to exclude
        #     :type exclude_id: int | None
        #     :type track_id: int
        #     :rtype: int
        #     """
        #     # object with the maximum FB error is the worst one
        #     if exclude_id is None:
        #         return np.argmax(self._heuristic_features[track_id, :, 0])
        #     else:
        #         median_fb_err = np.copy(self._heuristic_features[track_id, :, 0])
        #         median_fb_err[exclude_id] = -np.inf
        #         return np.argmax(median_fb_err)
