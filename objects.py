import numpy as np
import os
import sys
import time
import ast

from paramparse import MultiPath

from utilities import compute_overlaps_multi, CrossOverlaps, SelfOverlaps, CustomLogger, MDPStates, BaseParams, \
    add_suffix


class Objects:
    """
    set of labeled objects - abstracts out the common components of detections and annotations


    :ivar _params:
    :type _params: Objects.Params
    :ivar _logger:
    :type _logger: CustomLogger
    """

    class Params(BaseParams):
        """

        :ivar path: 'path of the text file in MOT format from where the objects data is to be read;'
                        'if this is empty, then a default path is constructed from the sequence and dataset names'

        :ivar fix_frame_ids: 'convert the frame IDs in the annotations and detections from 1-based '
                                 '(default MOT challenge format) to 0-based that is needed for internal'
                                 ' processing convenience',

        :ivar sort_by_frame_ids: 'sort data by frame IDs'

        :ivar ignored_regions: '1: read ignored_regions from annotations; '
                                '2: discard the regions after reading'

        :ivar clamp_scores:
        1: set all the score/confidence values to be between 0 and the 1 to deal with annoying
        entries with invalid confidence values outside this range
        2: set all confidence values to 1 to be able to use some MOT results as detections
        where the confidence values are all -1
        :type clamp_scores: int
        """

        def __init__(self, name):
            self.path = ''
            self.src_dir = MultiPath(name)
            self.fix_frame_ids = 1
            self.sort_by_frame_ids = 0
            self.ignore_ioa_thresh = 0.5
            self.allow_missing = 0

            self.percent_scores = 0
            self.clamp_scores = 0

    def __init__(self, obj_type, params, logger):
        """
        :param str obj_type:
        :param Objects.Params  params:
        :param CustomLogger logger:
        :rtype: None
        """
        self._type = obj_type
        self._params = params
        self._logger = CustomLogger(logger, names=[self._type.lower(), ])
        self.path = self._params.path

        # custom_log_header = {'custom_module': '{}'.format(self._type.lower())}
        # try:
        #     custom_log_header.update(self._logger.info.keywords['extra'])
        # except:
        #     pass
        # self._log_info = functools.partial(self._logger.info, extra=custom_log_header)
        # self._log_warn = functools.partial(self._logger.warning, extra=custom_log_header)
        # self._log_error = functools.partial(self._logger.error, extra=custom_log_header)

        self.ignored_regions = None
        self.data = None
        self.count = 0
        self.orig_n_frames = 0
        self.start_frame_id = 0
        self.end_frame_id = 0
        self.n_frames = 0
        self.idx = None
        # iou and index of the max iuo detection with each annotation or
        # the max iuo annotation with each detection in each frame
        self.max_cross_iou = None
        self.max_cross_iou_idx = None
        # future extensions
        self.features = None

        self._resize_factor = 1

    def initialize(self, orig_n_frames, start_frame_id=-1, end_frame_id=-1):
        """
        :type orig_n_frames: int
        :type start_frame_id: int
        :type end_frame_id: int
        :rtype: None
        """
        self.orig_n_frames = orig_n_frames
        self.start_frame_id = start_frame_id if start_frame_id >= 0 else 0
        self.end_frame_id = end_frame_id if end_frame_id > 0 else self.orig_n_frames - 1
        self.n_frames = self.end_frame_id - self.start_frame_id + 1

    def _sanity_check(self):
        """
        sanity checks
        """

        """frame IDs"""
        if (self.data[:, 0] < 0).any():
            raise IOError('Negative frame IDs found in data')
        """scores"""
        invalid_score_ids = np.where(np.logical_or(self.data[:, 6] < 0, self.data[:, 6] > 1))[0]
        if invalid_score_ids.size > 0:
            raise IOError('Invalid scores outside the range [0, 1] found in data')
        return True

    def _remove_ignored(self, ignored_regions):
        ioa_1 = np.empty((self.data.shape[0], ignored_regions.shape[0]))
        compute_overlaps_multi(None, ioa_1, None, self.data[:, 2:6], ignored_regions)
        valid_idx = np.flatnonzero(np.apply_along_axis(
            lambda x: np.all(np.less_equal(x, self._params.ignore_ioa_thresh)),
            axis=1, arr=ioa_1))
        n_invalid = self.data.shape[0] - valid_idx.size
        if n_invalid > 0:
            self._logger.info(f'Removing {n_invalid} {self._type} having IOA > {self._params.ignore_ioa_thresh} '
                              f'with {ignored_regions.shape[0]} ignored regions')
            self.data = self.data[valid_idx, ...]

    def _curtail(self):
        if self.start_frame_id > 0 or self.end_frame_id < self.orig_n_frames - 1:
            self._logger.info('Curtailing data to frames {:d} - {:d}'.format(self.start_frame_id,
                                                                             self.end_frame_id))
            valid_idx = np.logical_and(self.data[:, 0] >= self.start_frame_id,
                                       self.data[:, 0] <= self.end_frame_id)
            self.data = self.data[valid_idx, :]
            if self.start_frame_id > 0:
                self.data[:, 0] -= self.start_frame_id

    def _process(self, resize_factor):

        self._resize_factor = resize_factor

        if self._params.percent_scores:
            self._logger.info('Converting scores from percent to ratios')
            self.data[:, 6] /= 100.0
        if self._params.clamp_scores == 2:
            self._logger.info('Setting all scores to 1')
            self.data[:, 6] = 1.0
        elif self._params.clamp_scores:
            self._logger.info('Clamping out of range scores to be between 0 and 1')
            self.data[:, 6] = np.maximum(self.data[:, 6], 0)
            self.data[:, 6] = np.minimum(self.data[:, 6], 1)

        if self._params.sort_by_frame_ids:
            # sort by frame ID
            self._logger.info('sorting by frame IDs')
            self.data = self.data[self.data[:, 0].argsort(kind='mergesort')]

        if self._params.fix_frame_ids:
            # convert frame IDs from 1-based to 0-based
            self._logger.info('converting frame IDs to 0-based')
            self.data[:, 0] -= 1

        if resize_factor != 1:
            self._logger.info('resizing by factor: {}'.format(resize_factor))
            self.data[:, 2:6] *= resize_factor

    def _read(self, frame_size):
        """

        :param frame_size: [image_width, image_height]
        :return:
        """
        assert frame_size is not None and frame_size[0] > 0 and frame_size[1] > 0, \
            'invalid frame_size: {}'.format(frame_size)

        if not self._params.path:
            raise IOError('Data file path is not provided')

        if not os.path.isfile(self._params.path):
            msg = 'Data file does not exist: {:s}'.format(self._params.path)
            if self._params.allow_missing:
                self._logger.error(msg)
                return False
            else:
                raise IOError(msg)

        self.path = self._params.path
        self._logger.info('Reading from {:s}'.format(self._params.path))
        self.data = np.loadtxt(self._params.path, delimiter=',', ndmin=2)

        """get rid of annoying zero confidence meaningless garbage boxes in MOT"""
        valid_conf = self.data[:, 6] != 0
        valid_conf_idx = np.flatnonzero(valid_conf)

        self.data = self.data[valid_conf, :]

        """check spatial extents and clip to image boundaries and get rid of even more annoying 
        unspeakably foul boxes in mot17"""
        xmin, ymin, w, h = self.data[:, 2], self.data[:, 3], self.data[:, 4], self.data[:, 5]
        xmax, ymax = xmin + w, ymin + h

        xmin = np.clip(xmin, 0, frame_size[0])
        xmax = np.clip(xmax, 0, frame_size[0])
        ymin = np.clip(ymin, 0, frame_size[1])
        ymax = np.clip(ymax, 0, frame_size[1])

        valid_boxes = np.flatnonzero(np.logical_and(xmax > xmin, ymax > ymin))

        # assert invalid_boxes.size == 0, "invalid_boxes found in indices {}:\n{}".format(
        #     invalid_boxes, self.data[invalid_boxes, :])

        self.data[:, 2] = xmin
        self.data[:, 3] = ymin

        self.data[:, 4] = xmax - xmin
        self.data[:, 5] = ymax - ymin

        self.data = self.data[valid_boxes, :]

        return True

    def _build_index(self):
        """
        :rtype: None
        """
        # assumes that data is sorted by frame ID

        # end_ids = np.zeros((self.n_frames,), dtype=np.uint32)

        """locations where consecutive frame IDs are not equal
        these are decremented by 1 when compared to the original data vector
        since the vectors being compared are one less in size"""
        end_ids = np.flatnonzero(np.not_equal(self.data[1:, 0], self.data[:-1, 0]))

        # self.logger.debug('data_change_ids.shape: %(1)s', {'1': data_change_ids.shape})
        # self.logger.debug('data_change_ids: %(1)s', {'1': data_change_ids})

        """frame ids whose detections/annotations ended at the indices preceding before these;
        these too are located at one location before the one where the frame ID changed"""

        # frame_ids = self.data[data_change_ids, 0]
        # end_ids[frame_ids] = 1 + data_change_ids

        # last trajectory ends at the last entry

        # end_ids[data[-1]] = n_data

        self.idx = [None] * self.n_frames
        start_id = 0

        # for i in range(index_size):
        #     if not end_ids[i]:
        #         continue
        #     self.idx[i] = np.arange(start_id, end_ids[i])
        #     start_id = end_ids[i]

        for i in range(end_ids.size):
            frame_id = int(self.data[start_id, 0])
            end_id = end_ids[i] + 1
            self.idx[frame_id] = np.arange(start_id, end_id)
            start_id = end_id
        frame_id = int(self.data[start_id, 0])
        self.idx[frame_id] = np.arange(start_id, self.count)

        # self.logger.debug('paused')

    def _build_index_slow(self):
        """
        :rtype: None
        """
        # must be stable sorting to ensure that the indices for each object are sorted by frame ID
        frame_sort_idx = np.argsort(self.data[:, 0], kind='mergesort')
        sorted_frame_ids = self.data[frame_sort_idx, 0]
        end_ids = np.flatnonzero(
            np.not_equal(sorted_frame_ids[1:], sorted_frame_ids[:-1]))
        self.idx = [None] * self.n_frames
        start_id = 0
        for i in range(end_ids.size):
            frame_id = int(self.data[frame_sort_idx[start_id], 0])
            end_id = end_ids[i] + 1
            self.idx[frame_id] = frame_sort_idx[start_id:end_id]
            start_id = end_id
        frame_id = int(self.data[frame_sort_idx[start_id], 0])
        self.idx[frame_id] = frame_sort_idx[start_id:self.count]


class Detections(Objects):
    """
    :ivar _params:
    :type _params: Detections.Params
    """

    class Params(Objects.Params):
        def __init__(self):
            Objects.Params.__init__(self, 'Detections')
            self.score_thresh = 0

    def __init__(self, params, logger):
        """
        :type params: Detections.Params
        :type logger: CustomLogger
        :rtype: None
        """
        Objects.__init__(self, 'Detections', params, logger)
        self._params = params
        self.labels = None

    def read(self, frame_size, ignored_regions, resize_factor):
        """
        :param frame_size: [image_width, image_height]
        :type resize_factor: float
        :type ignored_regions: np.ndarray | None
        :rtype: bool
        """
        if not self._read(frame_size):
            return False

        self._process(resize_factor)

        if not self._sanity_check():
            return False

        n_data_cols = self.data.shape[1]

        if n_data_cols != 10:
            if n_data_cols == 6 or n_data_cols == 7 or n_data_cols == 9:
                """the extra -1s at the end of each line are missing and can be filled up"""
                diff = 10 - self.data.shape[1]
                self._logger.info(
                    'Data file has only {:d} values in each line so padding it with {:d} -1s'.format(
                        self.data.shape[1], diff))
                self.data = np.concatenate((self.data, np.tile([-1] * diff, (self.data.shape[0], 1))), axis=1)
            else:
                raise AssertionError('Data file has incorrect data dimensionality: {:d}'.format(
                    self.data.shape[1]))

        """curtail data to subsequence"""
        self._curtail()

        if ignored_regions is not None and ignored_regions.size > 0:
            self._remove_ignored(ignored_regions)

        if self._params.score_thresh > 0:
            self._logger.info('Removing detections with scores < {}'.format(self._params.score_thresh))
            valid_idx = self.data[:, 6] >= self._params.score_thresh
            self.data = self.data[valid_idx, :]

            min_score = np.amin(self.data[:, 6])
            max_score = np.amax(self.data[:, 6])
            self._logger.info('min, max scores: {}, {}'.format(min_score, max_score))

        self.count = self.data.shape[0]
        assert self.count > 0, 'No objects found'

        self._logger.info('count: {:d}'.format(self.count))

        if self._params.sort_by_frame_ids:
            self._build_index()
        else:
            self._build_index_slow()

        return True


class Annotations(Objects):
    """
    :type _params: Annotations.Params
    :type n_traj: int
    :type traj_idx: list[np.ndarray]
    :type traj_idx_by_frame: list[dict]
    """

    class Params(Objects.Params):
        """
        :ivar read_ignored_regions: read ignored regions from annotations
        :ivar read_occlusion_status: read occlusion status from annotations
        :ivar overlap_occ: minimum IOU between two annotations for them to be considered occluded;
        only used if read_occlusion_status is 0
        :ivar read_tra: 'read track data in CTC format needed for cell tracking',

        """

        def __init__(self, name='Annotations'):
            Objects.Params.__init__(self, name)
            self.data_dim = 10
            self.read_ignored_regions = 1
            self.read_occlusion_status = 0
            self.occlusion_heuristics = 0

            self.overlap_occ = 0.7

            self.remove_unknown_cols = 0
            self.read_tra = 0

    def __init__(self, params, logger, obj_type='Annotations'):
        """
        :type params: Annotations.Params
        :type logger: CustomLogger
        :rtype: None
        """
        Objects.__init__(self, obj_type, params, logger)
        self._params = params

        """no. of trajectories"""
        self.n_traj = 0

        """indices into data of all annotations in each trajectory"""
        self.traj_idx = None
        self.traj_idx_by_frame = None
        """dict mapping object IDs to trajectory IDs within traj_idx"""
        self.obj_to_traj = None
        """index of occurrence of each frame ID within each trajectory"""
        self.ann_idx = None

        self.max_ioa = None
        # self.occluded = None
        # self.occlusion_ratio = None
        self.occlusion_metadata = None
        self.scores = None
        self.area_inside_frame = None
        self.obj_sort_idx = None
        self.sorted_obj_ids = None
        self.areas = None
        """x, y coordinates of the bottom right corner of the bounding box"""
        self.br = None
        """intersection-over-union of all annotations in each frame with all other annotations in the same frame"""
        self.self_iou = None
        """intersection-over-union of all annotations in each frame with all detections in the same frame"""
        self.cross_iou = None

        self.cross_overlaps = None
        self._self_overlaps = None

    def _build_trajectory_index(self):
        # print('Annotations :: Building trajectory index...')

        # self.annotations.unique_ids, self.annotations.unique_ids_map = np.unique(
        # self.annotations.data[:, 1], return_inverse=True)

        # must be stable sorting to ensure that the indices for each object are sorted by frame ID
        self.obj_sort_idx = np.argsort(self.data[:, 1], kind='mergesort')
        self.sorted_obj_ids = self.data[self.obj_sort_idx, 1]
        end_ids = list(np.flatnonzero(
            np.not_equal(self.sorted_obj_ids[1:], self.sorted_obj_ids[:-1])))
        end_ids.append(self.count - 1)

        self.n_traj = len(end_ids)

        self.traj_idx = [None] * self.n_traj
        self.traj_idx_by_frame = [None] * self.n_traj
        self.obj_to_traj = {}
        self.traj_to_obj = {}

        start_id = 0
        for traj_id in range(self.n_traj):
            end_id = end_ids[traj_id] + 1
            traj_idx = self.obj_sort_idx[start_id:end_id]
            obj_id = int(self.data[traj_idx[0], 1])
            traj_frame_ids = self.data[traj_idx, 0].astype(np.int32)
            traj_obj_ids = self.data[traj_idx, 1].astype(np.int32)

            """sanity checks"""
            _, frame_counts = np.unique(traj_frame_ids, return_counts=True)
            unique = np.unique(traj_obj_ids, return_counts=False)

            """only one instance of each distinct object in each frame"""
            assert np.all(frame_counts == 1), f"duplicate frame IDs found for object {obj_id}:\n{traj_frame_ids}"
            """all object IDs are identical"""
            assert len(unique) == 1, f"duplicate object IDs found in trajectory {traj_id}:\n{traj_obj_ids}"
            """only one instance of each distinct object ID"""
            assert obj_id not in self.obj_to_traj, f"Duplicate object ID {obj_id} found " \
                f"for trajectories {self.obj_to_traj[obj_id]} and {traj_id}"

            # start_frame_id = int(np.amin(traj_frame_ids))
            # end_frame_id = int(np.amax(traj_frame_ids))

            self.traj_idx[traj_id] = traj_idx
            self.traj_idx_by_frame[traj_id] = {}
            for j, frame_id in enumerate(traj_frame_ids):
                frame_traj_idx = int(np.flatnonzero(np.equal(self.idx[frame_id], traj_idx[j])).item())
                assert traj_idx[j] == self.idx[frame_id][frame_traj_idx], "invalid frame_traj_idx"

                self.traj_idx_by_frame[traj_id][frame_id] = (traj_idx[j], frame_traj_idx)

            self.obj_to_traj[obj_id] = traj_id
            self.traj_to_obj[traj_id] = obj_id

            start_id = end_id

        self._logger.info('n_trajectories: {:d}'.format(self.n_traj))

    def get_mot_compatible_file(self):

        # return self.path

        suffix = ''

        if self.start_frame_id > 0 or self.end_frame_id < self.orig_n_frames - 1:
            suffix = '{}_{}_'.format(self.start_frame_id, self.end_frame_id)

        if self._resize_factor != 1:
            suffix += 'resize_{}_'.format(self._resize_factor)

        suffix += 'mot_compat'
        save_path = add_suffix(self.path, suffix)

        if os.path.exists(save_path):
            orig_mtime = os.path.getmtime(self.path)
            save_mtime = os.path.getmtime(save_path)

            if save_mtime > orig_mtime:
                """mot compatible annotations file was created after the source so doesn't need recreating"""
                # print('\nskipping creation of {} with save_mtime: {} > orig_mtime: {}\n'.format(
                #     save_path, save_mtime, orig_mtime
                # ))
                return save_path

            # print('save_mtime: {} < orig_mtime: {}\n'.format(
            #     save_mtime, orig_mtime
            # ))

        valid_data = np.copy(self.data[:, :10])

        """convert to 1-based frame IDss"""
        valid_data[:, 0] += 1

        self._logger.info('Saving mot_compatible annotations to: {}'.format(save_path))
        save_fmt = '%d,%d,%f,%f,%f,%f,%f,%d,%d,%d'
        np.savetxt(save_path, valid_data, fmt=save_fmt, delimiter=',', newline='\n')

        return save_path

    def read(self, frame_size, resize_factor):
        """

        :param frame_size: [image_width, image_height]
        :param float resize_factor:
        :rtype: bool
        """
        if not self._read(frame_size):
            return False

        if self._params.read_ignored_regions:
            ignored_regions_idx = np.logical_and(self.data[:, 0] == -1, self.data[:, 1] == -1)
            _ignored_regions_idx = np.flatnonzero(ignored_regions_idx)
            self.ignored_regions = self.data[_ignored_regions_idx, 2:6]
            _valid_idx = np.flatnonzero(np.logical_not(ignored_regions_idx))
            self.data = self.data[_valid_idx, ...]
            n_ignored_regions = _ignored_regions_idx.size
            if n_ignored_regions:
                self._logger.info('Found {} ignored_regions'.format(n_ignored_regions))
            # if self._params.read_ignored_regions == 2:
            #     self._logger.info('Discarding ignored_regions')
            #     self.ignored_regions = None

        self._process(resize_factor)

        if not self._sanity_check():
            return False

        if self._params.remove_unknown_cols:
            self._logger.info('Removing the amazingly annoying unexplained columns 8 and 9 from data')
            self.data = np.delete(self.data, (7, 8), 1)

        data_dim = self._params.data_dim
        assert self.data.shape[1] >= data_dim, \
            'Data file has incorrect data dimensionality: {:d}. Expected at least: {:d}'.format(
                self.data.shape[1], data_dim)

        if data_dim < 10:
            diff = 10 - data_dim
            self._logger.info(
                'Data file has only {:d} values in each line so padding it with {:d} -1s'.format(
                    self._params.data_dim, diff))
            concat_arr = [
                self.data[:, :data_dim],
                np.tile([-1] * diff, (self.data.shape[0], 1)),
                self.data[:, data_dim:]
            ]
            # if self.data.shape[1] > data_dim:
            #     concat_arr.append(self.data[:, data_dim:])
            self.data = np.concatenate(concat_arr, axis=1)

        if self._params.read_occlusion_status:
            assert self.data.shape[1] == 11, 'occlusion ratio column is unavailable'

            occlusion_ratio = self.data[:, 10]

            occluded = occlusion_ratio > self._params.overlap_occ
            meta_file_path = self.path.replace('.txt', '.meta')
            self._logger.info(f'Reading occlusion data from {meta_file_path}')
            with open(meta_file_path, 'r') as fid:
                self.occlusion_metadata = ast.literal_eval(fid.read())

            self.data = np.concatenate((
                self.data,
                occluded.reshape((-1, 1))
            ), axis=1)

        if self.ignored_regions is not None and self.ignored_regions.size > 0:
            self._remove_ignored(self.ignored_regions)

        """curtail data to subsequence"""
        self._curtail()

        self.count = self.data.shape[0]
        assert self.count > 0, 'No objects found'

        self._logger.info('count: {:d}'.format(self.count))

        # print('Building frame index...'.format(self.type))
        if self._params.sort_by_frame_ids:
            self._build_index()
        else:
            self._build_index_slow()

        """obtain the indices contained in each of the trajectories"""
        self._build_trajectory_index()

        return True

    def get_features(self, detections, n_frames, frame_size):
        """
        :type detections: Detections
        :type n_frames: int
        :type frame_size: tuple(int, int)
        :rtype: bool
        """

        """
        Compute self overlaps between annotations
        """
        # self.logger.info('Computing self overlaps between annotations')
        self._self_overlaps = SelfOverlaps()
        self._self_overlaps.compute(self.data[:, 2:6], self.idx, n_frames)

        self.max_ioa = self._self_overlaps.max_ioa
        self.areas = self._self_overlaps.areas
        self.br = self._self_overlaps.br
        self.self_iou = self._self_overlaps.iou

        if not self._params.read_occlusion_status:
            assert self.data.shape[1] == 10, 'unexpected number of columns in data'

            occluded = np.zeros((self.count,))
            occlusion_ratio = self.max_ioa

            if self._params.occlusion_heuristics:
                occluded[self.max_ioa > self._params.overlap_occ] = 1
            else:
                self._logger.warning('occlusion heuristics are disabled')

            self.data = np.concatenate((
                self.data,
                occlusion_ratio.reshape((-1, 1)),
                occluded.reshape((-1, 1))
            ), axis=1)

        """'Compute cross overlaps between detections and annotations"""
        # self.logger.info('Computing cross overlaps between detections and annotations')
        self.cross_overlaps = CrossOverlaps()
        self.cross_overlaps.compute(detections.data[:, 2:6], self.data[:, 2:6],
                                    detections.idx, self.idx, self.n_frames)

        # self.cross_iou = self.cross_overlaps.iou
        self.max_cross_iou = self.cross_overlaps.max_iou_2
        self.max_cross_iou_idx = self.cross_overlaps.max_iou_2_idx

        """annotations for which there are no corresponding detections"""
        no_det_idx = np.flatnonzero(self.max_cross_iou_idx == -1)

        self.scores = detections.data[self.max_cross_iou_idx, 6]
        self.scores[no_det_idx] = 0
        """
        compute intersection with the frame to determine the fraction of bounding box lying inside the frame extents
        """
        # n = self.count
        max_iou_det_data = detections.data[self.max_cross_iou_idx, :]
        ul_inter = np.maximum(np.array((1, 1)), max_iou_det_data[:, 2:4])  # n x 2
        br = max_iou_det_data[:, 2:4] + max_iou_det_data[:, 4:6] - 1
        br_inter = np.minimum(np.array(frame_size), br)  # n x 2
        size_inter = br_inter - ul_inter + 1  # n x 2
        size_inter[size_inter < 0] = 0
        area_inter = np.multiply(size_inter[:, 0], size_inter[:, 1])  # n x 1
        areas = np.multiply(max_iou_det_data[:, 4], max_iou_det_data[:, 5])
        self.area_inside_frame = np.divide(area_inter, areas)  # n x 1

        self.area_inside_frame[no_det_idx] = 0

        return True

    def get_mot_metrics(self, track_res, seq_name, dist_type=0):
        """
        :param TrackingResults track_res: tracking result
        :type dist_type: int
        :rtype: pandas.DataFrame, str
        """

        summary = strsummary = None
        import evaluation.motmetrics as mm

        assert self.n_frames == track_res.n_frames, 'MOT data to be compared must have the same number of frames'

        self._logger.info('Accumulating MOT data...')
        start_t = time.time()
        acc = mm.MOTAccumulator(auto_id=True)

        if dist_type == -1:
            return summary, strsummary, acc
        elif dist_type == 0:
            dist_func = mm.distances.iou_matrix
            self._logger.info('Using intersection over union (IoU) distance')
        else:
            dist_func = mm.distances.norm2squared_matrix
            self._logger.info('Using squared Euclidean distance')

        # cross_overlaps = utils.CrossOverlaps()
        # cross_overlaps.compute(self.data[:, 2:6], obj.data[:, 2:6], self.idx, obj.idx, self.n_frames)
        print_diff = int(self.n_frames / 10)
        states = track_res.states
        is_valid = track_res.is_valid

        # states = None
        # is_valid = None

        for frame_id in range(self.n_frames):
            idx1 = self.idx[frame_id]
            idx2 = track_res.idx[frame_id]
            if idx1 is not None:
                bbs_1 = self.data[idx1, 2:6]
                ids_1 = self.data[idx1, 1]
            else:
                bbs_1 = []
                ids_1 = []

            if idx2 is not None:
                if states is not None:
                    _states = states[idx2]
                    tracked_idx2 = idx2[_states == MDPStates.tracked]
                    idx2 = tracked_idx2

                if is_valid is not None:
                    _is_valid = is_valid[idx2].astype(np.bool)
                    valid_idx2 = idx2[_is_valid]
                    idx2 = valid_idx2

                bbs_2 = track_res.data[idx2, 2:6]
                ids_2 = track_res.data[idx2, 1]
            else:
                bbs_2 = []
                ids_2 = []

            # dist = cross_overlaps.iou[frame_id]
            dist = dist_func(bbs_1, bbs_2)
            acc.update(ids_1, ids_2, dist)
            if print_diff > 0 and (frame_id + 1) % print_diff == 0:
                # print('Done {:d}/{:d} frames'.format(frame_id + 1, self.n_frames))
                # sys.stdout.write("\033[F")
                sys.stdout.write('\rProcessed {:d}/{:d} frames'.format(
                    frame_id + 1, self.n_frames))
                sys.stdout.flush()
        sys.stdout.write('\rProcessed {:d}/{:d} frames\n'.format(self.n_frames, self.n_frames))
        sys.stdout.flush()
        end_t = time.time()
        fps = self.n_frames / (end_t - start_t)
        self._logger.info('FPS: {:.3f}'.format(fps))

        self._logger.info('Computing MOT metrics...')
        start_t = time.time()
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name=seq_name)
        end_t = time.time()
        fps = self.n_frames / (end_t - start_t)
        self._logger.info('FPS: {:.3f}'.format(fps))

        summary = summary.rename(columns=mm.io.motchallenge_metric_names)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters
        )
        return summary, strsummary, acc


class TrackingResults(Annotations):
    """
    :type _params: TrackingResults.Params
    """

    class Params(Objects.Params):
        """
        :ivar allow_debug: 'read debugging data from tracking results including MDP state and trajectory validity'


        """

        def __init__(self):
            Objects.Params.__init__(self, 'Tracking_Results')
            self.allow_debug = 2

    def __init__(self, params, logger, obj_type='Tracking_Results'):
        Annotations.__init__(self, params, logger, obj_type)
        """
        :type params: TrackingResults.Params
        :type logger: CustomLogger
        :rtype: None
        """
        self._params = params

        self.states = None
        self.is_valid = None

    def get_mot_compatible_file(self, check_states=1):

        # if self.is_valid is None and self.states is None:
        #     return self.path

        suffix = ''

        if self.start_frame_id > 0 or self.end_frame_id < self.orig_n_frames - 1:
            suffix = '{}_{}_'.format(self.start_frame_id, self.end_frame_id)

        if self._resize_factor != 1:
            suffix += 'resize_{}_'.format(self._resize_factor)

        suffix += 'mot_compat'
        save_path = add_suffix(self.path, suffix)

        if os.path.exists(save_path):
            # return save_path

            orig_mtime = os.path.getmtime(self.path)
            save_mtime = os.path.getmtime(save_path)

            if save_mtime > orig_mtime:
                """mot compatible file was created after the source so doesn't need recreating"""
                # print('\nskipping creation of {} with save_mtime: {} > orig_mtime: {}\n'.format(
                #     save_path, save_mtime, orig_mtime
                # ))
                return save_path

            print('save_mtime: {} < orig_mtime: {}\n'.format(
                save_mtime, orig_mtime
            ))

        valid_idx = np.ones((self.data.shape[0],), dtype=np.bool)

        if self.is_valid is not None:
            valid_idx = (self.is_valid != 0)

        if check_states and self.states is not None:
            valid_idx = np.logical_and(valid_idx, self.states == MDPStates.tracked)

        valid_data = self.data[valid_idx, :]

        """convert to 1-based frame IDs"""
        valid_data[:, 0] += 1

        self._logger.info('Saving mot_compatible results with {} / {} boxes to: {}'.format(
            valid_data.shape[0], self.data.shape[0], save_path))
        save_fmt = '%d,%d,%f,%f,%f,%f,%f,%d,%d,%d'
        np.savetxt(save_path, valid_data, fmt=save_fmt, delimiter=',', newline='\n')

        return save_path

    def read(self, frame_size, resize_factor):
        """
        :param frame_size: [image_width, image_height]
        :param float resize_factor:

        :rtype: bool
        """
        if not self._read(frame_size):
            return False

        self._process(resize_factor)

        if not self._sanity_check():
            return False

        n_data_cols = self.data.shape[1]

        """curtail data to subsequence"""
        self._curtail()

        if n_data_cols != 10:
            if n_data_cols == 12:
                if self._params.allow_debug == 2:
                    self.is_valid = self.data[:, 11]
                    self.states = self.data[:, 10]
                    self.data = self.data[:, :10]
                else:
                    raise IOError('Data has 12 columns which is invalid when invalid trajectories are not allowed')
            elif n_data_cols == 11:
                if self._params.allow_debug:
                    self.states = self.data[:, 10]
                    self.data = self.data[:, :10]
                else:
                    raise IOError('Data has 11 columns which is invalid when reading states is not allowed')
            else:
                raise AssertionError('Data file has incorrect data dimensionality: {:d}'.format(
                    self.data.shape[1]))

        self.count = self.data.shape[0]
        assert self.count > 0, 'No objects found'

        self._logger.info('count: {:d}'.format(self.count))

        if self._params.sort_by_frame_ids:
            self._build_index()
        else:
            self._build_index_slow()

        self._build_trajectory_index()

        return True
