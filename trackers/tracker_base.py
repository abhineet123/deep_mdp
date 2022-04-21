import numpy as np
import copy
import functools
import cv2

from utilities import BaseParams, CustomLogger, TrackingStatus, get_patch, draw_box, CVConstants, stack_images, \
    get_unique_ids, ids_to_member_names


class TrackerBase:
    """
    :type _logger: logging.RootLogger
    :type _target_id: int
    :type _members_to_copy: dict
    :type _params: TrackerBase.Params
    """

    class Result:
        class ROI:
            def __init__(self, templates, patches, template_bbox, patch_bboxes):
                self.templates = templates
                self.patches = patches
                self.template_bbox = template_bbox
                self.patch_bboxes = patch_bboxes

                self.n_patches = len(self.patches)
                self.n_templates = len(self.templates)

                self._vis_image = None

                assert self.templates[0].shape == self.patches[0].shape, \
                    "shape mismatch between templates and patches"

            def get_vis_image(self):
                if self._vis_image is not None:
                    return self._vis_image

                col_stacked_img = None

                for _templ_id in range(self.n_templates):
                    template = np.copy(self.templates[_templ_id])

                    draw_box(template, self.template_bbox, color='red')

                    for _patch_id in range(self.n_patches):
                        patch = np.copy(self.patches[_patch_id])
                        draw_box(patch, self.patch_bboxes[_patch_id, _templ_id, ...], color='red')

                        row_stacked_img = stack_images((template, patch), grid_size=(1, 2))

                        if col_stacked_img is None:
                            col_stacked_img = row_stacked_img
                        else:
                            col_stacked_img = stack_images((col_stacked_img, row_stacked_img), grid_size=(2, 1))

                self._vis_image = col_stacked_img
                return self._vis_image

        def __init__(self, locations, features, status, conf, roi, is_roi):
            self.locations = locations

            self.features = features
            self.status = status
            self.conf = conf
            self._roi = roi

            self.is_roi = is_roi

            self.heuristics = 1

        def get_summarized_features(self, template_ids, track_id):
            assert template_ids.size > 0, "template_ids is empty"

            if template_ids.size == 1:
                best_template = template_ids.item()
            else:
                best_template = self._get_best_template(template_ids, track_id)

            _features = self.features[track_id, best_template, ...].reshape((1, -1))
            return _features

        def get_status(self, template_id=None, track_id=None):
            assert self.heuristics, "tracker heuristics are disabled"

            if template_id is None:
                if track_id is None:
                    return self.status
                return self.status[track_id, :]

            if track_id is None:
                return self.status[:, template_id]

            return self.status[track_id, template_id]

        def get_status_m(self, track_id):
            """separate function needed to ensure correspondence with original code that uses
            multiple extraordinarily annoying heuristics to get status from LK in tracked and lost states"""
            assert self.heuristics, "tracker heuristics are disabled"

            return self.status[track_id, :]

        def set_status(self, valid_idx, invalid_idx, track_id=0):
            if valid_idx.size == 0:
                self.status[track_id, ...].fill(TrackingStatus.failure)
            else:
                self.status[track_id, invalid_idx] = TrackingStatus.failure

        def get_scores(self, track_id):
            assert self.heuristics, "tracker heuristics are disabled"
            if track_id is None:
                return self.conf
            return self.conf[track_id, :]

        def get_success_ids(self, track_id):
            assert self.heuristics, "tracker heuristics are disabled"

            return np.flatnonzero(np.not_equal(self.status[track_id, ...], TrackingStatus.failure))

        def _get_best_template(self, template_ids, track_id):
            assert self.heuristics, "tracker heuristics are disabled"

            conf = self.conf[template_ids, track_id].squeeze()
            best_id = np.argmax(conf)
            best_template = template_ids[best_id]
            return best_template

    class Params(BaseParams):
        """
        :ivar context_ratio: ratio between the object size and amount of neighbouring background to be included
        while extracting raw image patch as features; only matters if feature_type=-1

        """

        class ROI:
            """
            :ivar std_box_size: 'size of the standard box that all templates are resized into - (width height)',
            :ivar enlarge_box_size: 'factors by which to enlarge the box to obtain the ROI around it - (width height)',
            :ivar roi_interp_type_id: 'ID of the interpolation type used while resizing images before extracting the '
                      'ROI within which tracking is performed - indexes into '
                      'Utilities.CVConstants.interp_types',            """

            def __init__(self):
                self.enable = 1
                self.std_box_size = (60, 45)
                self.enlarge_box_size = (5, 3)
                self.roi_interp_type_id = 1

        def __init__(self):
            self.roi = TrackerBase.Params.ROI()
            self.feature_type = 0
            self.feature_shape = (64, 64)
            self.feature_rgb = 0
            self.context_ratio = 0.
            self.use_gt = 0.
            self.vis = 0
            self.verbose = 0

    def __init__(self, name, params, n_templates, update_location, rgb_input, logger, parent, policy_name):
        """

        :param TrackerBase.Params params:
        :param int update_location:
        :param logging.RootLogger logger:
        """

        self._parent = parent

        if self._parent is None:
            self._spawn_ids = []
            self._spawn_ids_gen = get_unique_ids(self._spawn_ids)

            self._enlarge_box = next(self._spawn_ids_gen)
            self._std_box_size = next(self._spawn_ids_gen)
            self._roi_offset = next(self._spawn_ids_gen)
            self._roi_size = next(self._spawn_ids_gen)
            self._roi_shape = next(self._spawn_ids_gen)
            self._std_box = next(self._spawn_ids_gen)
            self._roi_interp_type = next(self._spawn_ids_gen)
            self._to_gs = next(self._spawn_ids_gen)
            self._patch_as_features = next(self._spawn_ids_gen)

            self.feature_shape = next(self._spawn_ids_gen)
            self.n_features = next(self._spawn_ids_gen)

            self._members_to_spawn = next(self._spawn_ids_gen)
            self._members_to_copy = next(self._spawn_ids_gen)

            self._copy_ids = []
            self._copy_ids_gen = get_unique_ids(self._copy_ids, self._spawn_ids)
            self._roi = next(self._copy_ids_gen)

            self._register(init=1)
        else:
            self._spawn()

        self._target_id = -1

        self.name = name

        self._params = params
        self._n_templates = n_templates
        self._rgb_input = rgb_input
        self._policy_name = policy_name

        header_names = [name, ]
        if policy_name:
            header_names.insert(0, policy_name)

        self._logger = CustomLogger(logger, names=header_names)
        self._update_location = update_location

        if self._parent is None:
            if self._params.feature_type == - 1:
                self._to_gs = 1
                h, w = self._params.feature_shape
                self._logger.info('Using raw image box patches resized to {} x {}  as features'.format(h, w))
                if self._params.context_ratio > 0:
                    self._logger.info('using a ratio of {:.2f} to include neighbouring background '
                                      'in the patch for context'.format(self._params.context_ratio))
                if self._params.feature_rgb:
                    assert self._rgb_input, "rgb patch features can only be used with rgb input"

                    self.feature_shape = (6, h, w)
                    self._logger.info('Using RGB patches')
                    self._to_gs = 0
                else:
                    self.feature_shape = (2, h, w)

                self.n_features = np.prod(self.feature_shape)
                self._patch_as_features = 1
            else:
                self._patch_as_features = 0

            if not self._params.roi.enable:
                self._logger.info('Using ROI free tracking mode')
            else:
                self._enlarge_box = np.array(self._params.roi.enlarge_box_size, dtype=np.int32).reshape((1, 2))
                self._std_box_size = np.array(self._params.roi.std_box_size, dtype=np.int32).reshape((1, 2))

                # size of the border added around the scaled locations to get the ROI to be cropped
                # from the scaled images - same for all boxes
                self._roi_offset = (0.5 * (self._enlarge_box - 1) * self._std_box_size)

                # size of cropped ROI patch - same for all cropped patches
                self._roi_size = (self._std_box_size + 2 * self._roi_offset).astype(np.int32)

                # rearrange as (n_rows, n_cols, <n_channels>) for direct use as shape parameter for numpy arrays
                if self._rgb_input:
                    self._roi_shape = (self._roi_size[0, 1], self._roi_size[0, 0], 3)
                else:
                    self._roi_shape = (self._roi_size[0, 1], self._roi_size[0, 0])

                """location of the template bounding boxes within the cropped patches - same of all boxes
                and depends only on the ROI size and offset
                """
                self._std_box = np.concatenate((self._roi_offset, self._std_box_size), axis=1).reshape((1, 4))
                self._roi_interp_type = CVConstants.interp_types[self._params.roi.roi_interp_type_id]

        self._roi = None

        # if self._params.roi.enable:
        #     """cropped patches corresponding to a region of interest around each of the templates"""
        #     if self._rgb_input:
        #         self._roi = np.zeros((self._n_templates, self._roi_shape[0], self._roi_shape[1], 3), dtype=np.uint8)
        #     else:
        #         self._roi = np.zeros((self._n_templates, self._roi_shape[0], self._roi_shape[1]), dtype=np.uint8)

        self._round = np.vectorize(round)
        self._pause = 1
        self._heuristics = 1
        self._template_patches = {}
        self._annotations = None
        self._traj_idx_by_frame = None
        self._frame_id = None

    def update_frame(self, frame, frame_id):
        pass

    def _spawn(self):
        self._members_to_spawn = self._parent._members_to_spawn
        for _member in self._members_to_spawn:
            setattr(self, _member, getattr(self._parent, _member))

    def _register(self, init=0, reset=1):
        members_to_spawn = ids_to_member_names(self, self._spawn_ids)
        members_to_copy = ids_to_member_names(self, self._copy_ids)

        if init:
            self._members_to_spawn = []
            self._members_to_copy = []

        self._members_to_spawn += members_to_spawn
        self._members_to_copy += members_to_copy

        if reset:
            self._spawn_ids.clear()
            self._copy_ids.clear()

    def initialize(self, frame, bboxes):
        assert bboxes.shape[0] == self._n_templates, "initialization location counts does not match n_templates"

        if not self._params.roi.enable:
            for i in range(self._n_templates):
                self._initialize(i, frame, bboxes[i, ...].squeeze())
        else:
            if not self._get_roi(frame, bboxes, self._n_templates):
                raise SystemError('Template ROI extraction failed')

            for i in range(self._n_templates):
                self._initialize_roi(i, self._roi[i, ...].squeeze())

        if self._patch_as_features:
            _, h, w = self.feature_shape
            for template_id in range(self._n_templates):
                _, self._template_patches[template_id] = get_patch(frame, bboxes[template_id, ...], to_gs=self._to_gs,
                                                                   out_size=(h, w),
                                                                   context_ratio=self._params.context_ratio)

    def _initialize(self, template_id, frame, bbox):
        """

        :param int template_id:
        :param np.ndarray frame:
        :param np.ndarray bbox:
        :return:
        """
        raise NotImplementedError()

    def _initialize_roi(self, template_id, roi):
        """

        :param int template_id:
        :param np.ndarray roi:
        :return:
        """
        raise NotImplementedError()

    def update(self, template_id, frame, bbox):
        if not self._params.roi.enable:
            self._update(template_id, frame, bbox)
        else:
            if not self._get_roi(frame, bbox.reshape((1, 4)), 1):
                raise SystemError('Template ROI extraction failed')

            self._update_roi(template_id, self._roi.squeeze())

        if self._patch_as_features:
            _, h, w = self.feature_shape
            _, self._template_patches[template_id] = get_patch(frame, bbox, to_gs=self._to_gs, out_size=(h, w),
                                                               context_ratio=self._params.context_ratio)

    def set_gt(self, annotations, traj_idx_by_frame):
        self._annotations = annotations
        self._traj_idx_by_frame = traj_idx_by_frame

    def _update(self, template_id, frame, bbox):
        raise NotImplementedError()

    def locations_from_gt(self, n_patches_2):
        assert self._annotations is not None and self._traj_idx_by_frame is not None, \
            "valid annotations must be provided to get locations from GT"

        gt_location = locations = None
        try:
            self._curr_ann_idx = self._traj_idx_by_frame[self._frame_id]
        except:
            pass
        else:
            gt_location = self._annotations.data[self._curr_ann_idx[0], 2:6].copy()
            exp_locations = np.expand_dims(np.expand_dims(gt_location, axis=0), axis=0)
            locations = np.tile(exp_locations, (n_patches_2, self._n_templates, 1))

        return gt_location, locations

    def _update_roi(self, template_id, roi):
        raise NotImplementedError()

    @staticmethod
    def from_roi(in_locations, out_locations, _n_templates, _transform):
        _transform = np.expand_dims(_transform, axis=1)
        __transform = np.tile(_transform, (1, _n_templates, 1))
        out_locations[..., :2] = (in_locations[..., :2] + __transform[..., :2]) * __transform[..., 2:]
        out_locations[..., 2:] = (in_locations[..., 2:] - 1) * __transform[..., 2:] + 1

    @staticmethod
    def to_roi(in_locations, out_locations, _n_templates, _transform):
        _transform = np.expand_dims(_transform, axis=1)
        __transform = np.tile(_transform, (1, _n_templates, 1))
        out_locations[..., :2] = in_locations[..., :2] / __transform[..., 2:] - __transform[..., :2]
        out_locations[..., 2:] = ((in_locations[..., 2:] - 1) / __transform[..., 2:]) + 1

    def track(self, frame, frame_id, locations, n_objs, heuristics):
        """

        :param np.ndarray frame:
        :param int frame_id:
        :param np.ndarray | None locations:
        :param int n_objs:
        :param int heuristics:
        :return:
        """
        self._heuristics = heuristics
        self._frame_id = frame_id

        if not self._params.roi.enable:
            track_res = self._track(frame, frame_id, locations)  # type: TrackerBase.Result
        else:
            if not self._get_roi(frame, locations, n_objs):
                return None

            track_res = self._track_roi(self._roi)  # type: TrackerBase.Result

            assert track_res.locations.shape[1] == self._n_templates, \
                "mismatch between in_locations shape and n_templates"

            if track_res.is_roi:
                in_locations = np.copy(track_res.locations)
                self.from_roi(in_locations, track_res.locations,  self._n_templates,  self._transform)

        if track_res is None:
            return None

        if self._patch_as_features:
            track_res.features = self.get_features(frame, None, track_res.locations, n_objs)

        track_res.heuristics = self._heuristics
        return track_res

    def _track(self, frame, frame_id, locations):
        raise NotImplementedError()

    def _track_roi(self, patches):
        raise NotImplementedError()

    def get_features(self, frame, templ_id, bboxes, n_objs):
        if not self._patch_as_features:
            return self._get_features(frame, templ_id, bboxes, n_objs)

        _, h, w = self.feature_shape

        if templ_id is not None:
            templ_ids = [templ_id, ]
            n_templates = 1
        else:
            templ_ids = list(range(self._n_templates))
            n_templates = self._n_templates

        assert bboxes.shape == (n_objs, n_templates, 4), "invalid shape of bboxes"

        features = np.zeros((n_objs, n_templates, self.n_features), dtype=np.float32)
        for __id, _templ_id in enumerate(templ_ids):
            template = self._template_patches[_templ_id]

            for _patch_id in range(n_objs):
                _, patch = get_patch(frame, bboxes[_patch_id, __id, ...],
                                     to_gs=self._to_gs, out_size=(h, w), context_ratio=self._params.context_ratio)

                stacked_img = np.stack((template, patch), axis=0).astype(np.float32)
                features[_patch_id, __id, ...] = stacked_img.flatten() / 255.0

        return features

    def _get_features(self, frame, templ_id, bboxes, n_objs):
        raise NotImplementedError()

    def set_id(self, target_id):
        self._target_id = target_id

    def get_init_samples(self):
        raise NotImplementedError()

    def set_region(self, template_id, frame, bbox):
        raise NotImplementedError()

    def extract_roi(self, roi, frame, locations, n_roi, transform=None):
        """
        :type roi: np.ndarray
        :type frame: np.ndarray
        :type locations: np.ndarray
        :type n_roi: int
        :type transform: None | np.ndarray
        :rtype: None
        """

        assert self._params.roi.enable, "ROI mode is disabled"

        # if len(roi.shape)==2:
        #     roi=np.expand_dims(roi, axis=0)

        frame_size = np.array((frame.shape[1], frame.shape[0]))
        # ratio between the new and old sizes of the boxes
        scaling_factors = np.array(self._params.roi.std_box_size) / locations[:, 2:]
        # UL of the scaled boxes within the scaled images
        scaled_ul = self._round(locations[:, :2] * scaling_factors).astype(np.int32)
        # sizes of the scaled images
        scaled_frame_sizes = self._round(frame_size * scaling_factors).astype(np.int32)
        # UL of the cropped ROI within the scaled images
        roi_ul = scaled_ul - self._roi_offset

        # location of ROI within the scaled image after adjusting for the image extents
        smin = np.maximum(0, np.minimum(scaled_frame_sizes, roi_ul - 1)).astype(np.int32).reshape((n_roi, 2))
        smax = np.maximum(0, np.minimum(scaled_frame_sizes, roi_ul + self._roi_size - 1)).astype(np.int32).reshape(
            (n_roi, 2))

        # location of ROI within the cropped image - the ROI might be partially outside the scaled image
        # in which case the cropped image will be partially empty or zero
        cmin = (smin - roi_ul + 1).astype(np.int32).reshape((n_roi, 2))
        cmax = (smax - roi_ul + 1).astype(np.int32).reshape((n_roi, 2))

        if self._params.vis:
            frame_disp = np.copy(frame)
            for roi_id in range(n_roi):
                roi_pt1 = tuple((smin[roi_id, :] / scaling_factors[roi_id]).astype(np.int32))
                roi_pt2 = tuple((smax[roi_id, :] / scaling_factors[roi_id]).astype(np.int32))

                location = locations[roi_id, :].squeeze()

                pt1 = (int(location[0]), int(location[1]))
                pt2 = (int(location[0] + location[2]),
                       int(location[1] + location[3]))

                cv2.rectangle(frame_disp, roi_pt1, roi_pt2, (0, 0, 255), thickness=2)
                cv2.rectangle(frame_disp, pt1, pt2, (0, 255, 0), thickness=2)

            cv2.imshow('extractROI', frame_disp)
            k = cv2.waitKey(1 - self._pause)
            if k == 32:
                self._pause = 1 - self._pause

        roi.fill(0)
        # if n_roi > 1:
        for roi_id in range(n_roi):
            cmin_x, cmin_y = cmin[roi_id, :]
            cmax_x, cmax_y = cmax[roi_id, :]
            if cmin_x >= cmax_x or cmin_y >= cmax_y:
                orig_box = locations[roi_id, :].squeeze()
                min_x, min_y, w, h = orig_box.squeeze()
                max_x, max_y = min_x + w, min_y + h
                self._logger.error('Invalid target roi bbox: {} with orig_box: {}, {}'.format(
                    [cmin_x, cmin_y, cmax_x, cmax_y], orig_box, [min_x, min_y, max_x, max_y]))
                return False
            smin_x, smin_y = smin[roi_id, :]
            smax_x, smax_y = smax[roi_id, :]
            if smin_x >= smax_x or smin_y >= smax_y:
                self._logger.error('Invalid source roi bbox: {}'.format([smin_x, smin_y, smax_x, smax_y]))
                return False

            scaled_frame = cv2.resize(frame, tuple(scaled_frame_sizes[roi_id, :]),
                                      interpolation=self._roi_interp_type)
            roi[roi_id, cmin[roi_id, 1]:cmax[roi_id, 1], cmin[roi_id, 0]:cmax[roi_id, 0], ...] = \
                scaled_frame[smin[roi_id, 1]:smax[roi_id, 1], smin[roi_id, 0]:smax[roi_id, 0], ...]

        if transform is not None:
            # additive and multiplicative factors to convert from coordinate locations from the frame of reference
            # of the ROI to that of the original frame
            transform[:] = np.concatenate((roi_ul, 1.0 / scaling_factors), axis=1)

        return True

    def _get_roi(self, frame, bboxes, n_bboxes):
        self._transform = np.zeros((n_bboxes, 4))
        self._roi = np.zeros((n_bboxes,) + self._roi_shape, dtype=np.uint8)

        if not self.extract_roi(
                self._roi, frame, bboxes, n_bboxes, self._transform):
            self._logger.error('ROI extraction failed')
            return False

        if n_bboxes == 1:
            self._transform = self._transform.reshape((1, -1))

        return True

    def get_stacked_roi(self, anchor_id, frame_ids, frames, locations, show=1, grid_size=None, ):

        assert self._params.roi.enable, "ROI mode is disabled"

        roi_disp = np.copy(self._roi)
        frames_disp = []

        for i in range(self._n_templates):
            if i == anchor_id:
                color = 'green'
            else:
                color = 'cyan'
            draw_box(roi_disp[i, ...], self._std_box,
                     _id=f'{i}:{frame_ids[i].item()}',
                     color=color, thickness=2)
            frame_disp = np.copy(frames[i])
            draw_box(frame_disp, locations[i, ...].squeeze(),
                     _id=f'{i}:{frame_ids[i].item()}',
                     color=color, thickness=2)
            frames_disp.append(frame_disp)

        stacked_roi = stack_images(roi_disp, grid_size=grid_size)
        stacked_frames_disp = stack_images(frames_disp, grid_size=grid_size)
        if show:
            cv2.imshow('get_stacked_roi', stacked_roi)
            k = cv2.waitKey(1 - self._pause)
            if k == 27:
                cv2.destroyWindow('get_stacked_roi')
                exit()
            elif k == 32:
                self._pause = 1 - self._pause
        return stacked_roi, stacked_frames_disp

    def copy(self):
        """
        :rtype: dict
        """
        obj_dict = {}

        for _attr in self._members_to_copy:
            self_attr = getattr(self, _attr)
            try:
                obj_dict[_attr] = copy.deepcopy(self_attr)
            except TypeError as err:
                raise TypeError('Deep copying of attribute {} failed:'.format(_attr)) from err
        return obj_dict

    def restore(self, obj_dict, deep_copy=False):
        """
        :type obj_dict: dict
        :type deep_copy: bool
        """
        for _attr in self._members_to_copy:
            obj_attr = obj_dict[_attr]
            if deep_copy:
                setattr(self, _attr, copy.deepcopy(obj_attr))
            else:
                setattr(self, _attr, obj_attr)

    def write_state_info(self, files, root_dir, write_to_bin,
                         fp_fmt='%.4f', fp_dtype=np.float32):
        raise NotImplementedError()
