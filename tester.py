import sys
import os
import shutil
import logging
import time
import math
import cProfile, pstats, io
from pprint import pformat
from datetime import datetime

import pandas as pd
import numpy as np

import paramparse

np.seterr(all='raise')

from target import Target
from input import Input
from policy_base import PolicyBase
# from lost import Lost
# from tracked import Tracked
from visualizer import Visualizer, ObjTypes
from data import Data
from utilities import MDPStates, DebugParams, compute_overlaps_multi, compute_self_overlaps, \
    mot_metrics_to_file, combined_motmetrics, CustomLogger, \
    AnnotationStatus, print_df, write_df, linux_path, draw_box, annotate_and_show, \
    list_to_str, BaseParams, add_suffix, find_associations, combine_hota_devkit_results, \
    hota_metrics_to_file, copy_rgb, interpolate_missing_objects, associate_gt_to_detections

from torch.utils.tensorboard import SummaryWriter

try:
    import pyHungarian
except ImportError as e:
    print('pyHungarian unavailable: {}'.format(e))

# temporary hack for debugging
from history import History


class Tester:
    """
    :type trained_target: Target
    :type _params: Tester.Params
    :type _logger: logging.RootLogger | CustomLogger
    :type __logger: CustomLogger
    :type input: Input
    :type visualizer: Visualizer
    :type targets: list[Target]
    :type n_live: int
    :type target_sort_idx: np.ndarray
    :type _results: np.ndarray
    :type states: np.ndarray
    :type locations: np.ndarray
    :type tracked_streaks: np.ndarray
    :type vis: bool
    """

    class Params(BaseParams):
        """
        :ivar overlap_sup: 'IOA threshold used to decide if two targets  are in the same image region and if'
                       'one of them therefore needs to be suppressed',
        :ivar overlap_suppress1: 'IOA threshold for non-maximum suppression or future extension',
        :ivar overlap_suppress2: 'IOA threshold for non-maximum suppression or future extension',
        :ivar read_all_frames: 'read all frames in the input sequence at once to avoid repeated disk accesses',
        :ivar enable_nms: 'enable non-maximum suppression of detections',
        :ivar nms_overlap_thresh: 'three element tuple of floats that specifies the iou, ioa_1 and ioa_2 thresholds'
                      'for performing non-maximum suppression; only matters if enable_nms is 1 ',
        :ivar target_sort_sep: 'tracked streak size to use to separate the two sets of targets while sorting them',
        :ivar max_inactive_targets: 'maximum no. of inactive targets allowed to accumulate in the record'
                        ' before being removed',
        :ivar filter_iou_thresh: 'IOU threshold between detection and target location for deciding which detections'
                     ' to filter out',
        :ivar filter_ioa_thresh: 'IOU threshold between detection and target location for deciding which detections'
                     ' to filter out',
        :ivar check_out_of_scene: 'check if target has gone out of scene after lost state decision making - this is '
                      'performed irrespective of whether it transitions to the tracked state and is one'
                      ' of the annoying ad-hoc heuristics used in the original code',
        :ivar check_next_frame: 'check if the predicted location of the target in the next frame is out of scene after'
                    ' lost state decision making - this is only performed if  the target remains in the '
                    'lost state and is one of the annoying ad-hoc heuristics used in the original code',
        :ivar min_trajectory_len: 'Minimum number of frames for which a particular target must be in the tracked '
                      'state for it to be considered a valid trajectory and be written to the final results',
        :ivar next_frame_exit_threshold: 'minimum fraction of the predicted location of the object in the next frame '
                             'that must lie inside the frame for it not to be removed',
        :ivar tracked_as_detection: 'Use tracked location as an additional dummy detection for association whenever '
                        'transitioning to lost from tracked state;
                        actual value indicates the maximum number of consecutive associations with these
                        dummy detections that can be allowed before discarding the target',
        :ivar hungarian: 'Use the Hungarian algorithm for performing association between lost targets and detections',
        :ivar devkit: 'use official MOT devkit for evaluation; '
                    'only works if a supported MATLAB version is installed '
                    'including its python interface and devkit module has been compiled',
        :ivar vis: 'Enable visualization of tracking results',
        :ivar profile: 'Enable code profiling',
        :ivar max_active_targets: 'maximum number of active targets allowed before testing is terminated '
                      'assuming bad model or bug',
        :ivar train_active: 'train_active policy using detections and annotations for each sequence '
                'before testing on it; '
                'hack to enable lost and tracked policies to be evaluated independently of active',
        :ivar ignore_missing_ann: 'disable annotation dependent functionality if annotations not found '
                      'and continue testing instead of raising an error; should be on for datasets '
                      'where test sets do not have labels',
        :ivar save_debug_info: '>= 1: save lost targets in output txt; '
                   '2: also save trajectories whose length < min_trajectory_len with a '
                   'corresponding validity marker in an extra column',
        :ivar filtering_method: 0: iterative mutual maximum overlap between detections and targets,
                                1: annoying old heuristics based filtering with iou and ioa comparisons
                                between individual pairs of detections and targets

        :ivar override: 'Override parameters in the trained target with those specified at runtime; '
                    'the learned model in the target might depend on several of these parameters '
                    'and might not function well or otherwise behave in an unpredictable manner'
                    ' if these are changed;',
        :ivar target: 'placeholder Target parameters; only used if override is enabled',
        :type target: Target.Params

        :ivar verbose: 'Enable printing of general diagnostic messages',
        :ivar input: 'Input parameters',
        :ivar visualizer: 'Visualizer parameters',
        :ivar debug: 'Debugging parameters'

        :ivar res_from_gt: generate the best possible tracking results that can be obtained using the ground truth by
        associating them together using the method specified by ann_assoc_method and ann_iou_threshold
        res_from_gt = 2 enables interpolation of missing results corresponding to false negative detections

        :ivar ann_assoc_method: 0: associate each detection to the max overlapping annotation
                                1: same as ann_assoc_method=0 except that the unique GT association constraint is also
                                applied, i.e. and annotation is associated with a detection only if has not already
                                been associated with some other detection
                                2: iterative mutual maximum overlap between detections and annotations,
                                3: associate using Hungarian algorithm

        :ivar ann_iou_threshold: minimum overlap between the max-overlap annotation and initial location for this
        target to correspond to that annotation otherwise it corresponds to a
            false positive,


        """

        def __init__(self):
            self.overlap_sup = 0.7  # min iou between targets to suppress one of them

            self.overlap_suppress1 = 0.5  # overlap for suppressing detections with tracked objects
            self.overlap_suppress2 = 0.5  # overlap for suppressing detections with tracked objects

            self.enable_nms = 0  # enable non-maximum suppression of detections
            """ iou, ioa_1 and ioa_2 thresholds for performing non-maximum suppression
            """
            self.nms_overlap_thresh = (0.6, 0.95, 0.95)
            """
            tracked streak size to use to separate the two sets of targets while sorting them
            """
            self.target_sort_sep = 10

            # overlap thresholds for deciding which detections to filter out
            self.filter_iou_thresh = 0.5
            self.filter_ioa_thresh = 0.5

            self.hungarian = 0

            """target level heuristics"""
            self.lost_heuristics = 1
            self.tracked_heuristics = 1
            self.reconnect_lost = 1

            """global context heuristics"""
            self.resolve_conflicts = 1
            self.sort_targets = 1
            self.filter_detections = 1
            self.filtering_method = 1

            # maximum allowed lost to trained ratio before least likely lost targets are removed
            self.max_lost_ratio = 0
            # maximum allowed targets before overall failure is assumed
            self.max_targets = 0

            # maximum no. of inactive targets allowed to accumulate in the record before being removed
            self.max_inactive_targets = 0
            self.min_trajectory_len = 5

            self.tracked_as_detection = 0

            self.ignore_missing_ann = 0

            self.ann_iou_threshold = 0.25
            self.ann_assoc_method = 0

            # self.ann_iou_threshold = 0.5
            # self.ann_assoc_method = 1

            self.print_stats = 0
            self.save_debug_info = 2
            self.hota = 1
            self.devkit = 0
            self.load_from_alternative = 0
            self.save_mot = 1

            self.accumulative_eval_path = 'log/mot_metrics_accumulative.log'

            self.input = Input.Params()
            self.target = Target.Params()
            self.visualizer = Visualizer.Params()

            """
            Debugging
            """
            """pass annotations to the target for debugging"""
            self.use_annotations = 1
            """train active state for each tested sequence"""
            self.train_active = 0

            """override params in trained target"""
            # self.override = 0

            self.res_from_gt = 0

            self.vis = 0
            self.profile = 0
            self.verbose = 0
            self.verbose_depth = 0
            self.debug = DebugParams()

    def __init__(self, trained_target, params, logger, args_in=()):
        """
        :type trained_target: Target | None
        :type params: Tester.Params
        :type logger: CustomLogger
        :rtype: None
        """

        self._params = params
        self._logger = logger
        self._args_in = args_in

        if self._args_in:
            self._target_args_in = [k.replace('tester.target.', '') for k in self._args_in
                                    if k.startswith('--tester.target.')]

        self.__logger = logger
        self._trained_target = trained_target

        # self.logging_handler = None

        # convert_to_gs = self.params.input.convert_to_gs
        # tracker = self.params.target.templates.tracker
        # if not convert_to_gs and tracker in utils.Trackers['LK']:
        #     self.logger.warning('LK only supports grayscale images')
        #     self.params.input.convert_to_gs = 1

        self.rgb_input = not self._params.input.convert_to_gs
        self.input = Input(self._params.input, self._logger)

        self.targets = []
        self.failed_targets = []
        self.n_live = 0
        self.n_total_targets = 0
        self.target_sort_idx = None
        self._results = None
        self.ids = np.empty((0,), dtype=np.uint32)
        self.states = np.empty((0,), dtype=np.uint32)
        self.locations = np.empty((0, 4), dtype=np.float64)
        self.tracked_streaks = np.empty((0,), dtype=np.uint32)
        self.deleted_targets = []

        self.next_predicted_location = np.zeros((1, 4))

        self.vis = np.array(self._params.visualizer.mode).any() and \
                   (self._params.visualizer.save or self._params.visualizer.show) and \
                   self._params.vis
        self.visualizer = Visualizer(self._params.visualizer, self._logger)

        self.tracked_as_detection = self._params.tracked_as_detection
        if self.tracked_as_detection < 0:
            self.tracked_as_detection = self._params.min_trajectory_len - 1

        self.is_initialized = False
        self.pause_for_debug = 0
        self.load_fname = None
        self.profiler = cProfile.Profile()

        if self._params.res_from_gt:
            self._logger.debug('Getting tracking results from GT')
            if self._params.res_from_gt == 2:
                self._logger.debug('interpolation of missing objects is enabled')
            else:
                self._logger.debug('interpolation of missing objects is disabled')
        else:
            self._logger.info('using min_trajectory_len: {}'.format(self._params.min_trajectory_len))

            if self.tracked_as_detection:
                self._logger.info('using tracked_as_detection: {}'.format(self.tracked_as_detection))
            if self._params.profile:
                self._logger.info('Profiling is enabled')

            if self._params.train_active:
                self._logger.warning('Active policy training is enabled')

            # self.read_img = 1
            # if self.params.load and not self.visualize:
            #     self.read_img = 0
            #
            # self.read_det = 1
            # if self.params.load and not self.params.visualizer.mode[1]:
            #     self.read_det = 0
            #
            # self.read_ann = 0
            # if self.params.evaluate or self.params.visualizer.mode[2]:
            #     self.read_ann = 1

            """target level heuristics"""
            if not self._params.lost_heuristics:
                self._logger.warning('lost_heuristics are disabled')
            if not self._params.tracked_heuristics:
                self._logger.warning('tracked_heuristics are disabled')
            if not self._params.reconnect_lost:
                self._logger.warning('reconnecting of recently lost targets is disabled')

            """global context heuristics"""
            if not self._params.sort_targets:
                self._logger.warning('target sorting is disabled')
            if not self._params.filter_detections:
                self._logger.warning('detection filtering is disabled')
            if not self._params.resolve_conflicts:
                self._logger.warning('target conflict resolution is disabled')

            if self._params.max_lost_ratio > 0:
                self._logger.warning('Limiting lost targets to {}% of tracked targets'.format(
                    self._params.max_lost_ratio * 100))

            if self._params.hungarian:
                self._logger.info('Using Hungarian algorithm for association')

            if self._trained_target is not None:
                if self._target_args_in:
                    # if self._params.verbose == 2:
                    self._logger.warning('\n\nOverriding following trained target parameters:\n{}\n'.format(
                        pformat(self._target_args_in)))
                    paramparse.process(self._trained_target.params, self._target_args_in, prog='target', usage=None)
                    # self.trained_target.params = self.params.target

                self._params.target = self._trained_target.params

                # temporary hack for debugging
                self._trained_target.params.set_verbosity(self._params.verbose, self._params.verbose_depth)
                self._trained_target.params.history = History.Params()

                # amazingly annoying heuristic from the original code
                self._trained_target.params.tracked.exit_threshold = 0.70

                self._trained_target.test_mode()

                # if self._params.mode != SaveModes.none:
                #     assert self._params.use_annotations, "use_annotations must be enabled to generate test samples"

        self.lost_seq_ids = []
        self.lost_ids = []
        self.lost_targets = []

        self.tracked_ids = []
        self.tracked_targets = []
        self._start_end_frame_ids = {}

        self._n_lost = 0
        self._n_tracked = 0

        self._results_raw = None
        self._results = None
        self._gt_results = None
        self.annotations = None
        self._detections = None
        self._ann_to_targets = {}
        self._targets_to_ann = {}

        self._tb_path = None

        self._acc_dict = {}
        self._stats_dict = {}
        self._percent_stats_dict = {}

    def initialize(self, data=None, load=False, evaluate=False, tb_path=None, frame_size=None,
                   save_fname_templ=None, logger=None):
        """
        :type data: Data | None
        :type load: bool | int
        :type evaluate: bool | int
        :type frame_size: tuple | None
        :type save_fname_templ: str | None
        :type logger: CustomLogger
        :rtype: bool
        """
        if logger is not None:
            self._logger = logger

        self.targets = []
        self.n_live = 0
        self.n_total_targets = 0
        self.target_sort_idx = None
        self._results_raw = None
        self._results = None
        self.states = np.empty((0,), dtype=np.uint32)
        self.locations = np.empty((0, 4), dtype=np.float64)
        self.tracked_streaks = np.empty((0,), dtype=np.uint32)
        self.deleted_targets = []

        self.annotations = None
        self._detections = None
        self._ann_to_targets = {}
        self._targets_to_ann = {}
        self._tb_path = tb_path

        self._start_end_frame_ids = {}

        self.next_predicted_location = np.zeros((1, 4))

        ignored_regions = None
        if data is not None:
            # self._logger = CustomLogger(self.__logger, names=(data.seq_name,), key='custom_header')

            """initialize input pipeline"""
            read_img = 1
            self.input.params.batch_mode = 0
            if not self.input.initialize(data, read_img, logger=self._logger):
                raise IOError('Input pipeline could not be initialized')
                # return False

            """read annotations"""
            read_ann = evaluate or self._params.visualizer.mode[2] or \
                       self._params.use_annotations or \
                       self._params.train_active

            if read_ann:
                if not self.input.read_annotations():
                    if self._params.ignore_missing_ann:
                        self._logger.warning('Annotations not found so disabling corresponding functionality')
                        _mode = list(self._params.visualizer.mode)
                        _mode[2] = 0
                        self._params.visualizer.mode = tuple(_mode)
                        self._params.use_annotations = self._params.train_active = 0
                        ignored_regions = None
                    else:
                        raise IOError('Annotations could not be read')
                else:
                    self.annotations = self.input.annotations
                    ignored_regions = self.annotations.ignored_regions

            """read detections"""
            read_det = not load or self._params.visualizer.mode[1] or \
                       self._params.use_annotations or \
                       self._params.train_active

            if read_det:
                if not self.input.read_detections():
                    raise IOError('Detections could not be read')
                else:
                    self._detections = self.input.detections

            if save_fname_templ is None:
                save_fname_templ = '{:s}_{:d}_{:d}'.format(
                    data.seq_name, data.start_frame_id + 1,
                                   data.end_frame_id + 1)
            frame_size = self.input.frame_size
        else:
            if save_fname_templ is None:
                save_fname_templ = 'server'
            if frame_size is None:
                raise IOError('Frame size must be provided in the server mode')
                # return False

        if self._params.use_annotations or self._params.train_active:
            self.annotations.get_features(self.input.detections, self.input.n_frames, self.input.frame_size)

        # initialize visualizer
        if self.vis:
            if self._tb_path is not None:
                save_dir = os.path.dirname(self._tb_path)
                if self._params.visualizer.save:
                    vis_save_dir = os.path.join(save_dir, "videos")
                    self._params.visualizer.save_dir = vis_save_dir

            if not self.visualizer.initialize(save_fname_templ, frame_size, ignored_regions=ignored_regions):
                raise IOError('Visualizer could not be initialized')

        if self._trained_target is not None:
            self._trained_target.reset_stats()
            self._trained_target.reset_test_samples()

        self.is_initialized = True

        return True

    def run(self):
        """
        :type tb_path: str
        :rtype: bool
        """

        if not self.is_initialized:
            raise AssertionError('Tester has not been initialized')

        if self._params.use_annotations:
            annotations = self.input.annotations
            self._logger.info('Annotations are enabled')
        else:
            annotations = None

        if self._params.train_active:
            self.input.detections.max_cross_iou = self.input.annotations.cross_overlaps.max_iou_1
            self.input.detections.max_cross_iou_idx = self.input.annotations.cross_overlaps.max_iou_1_idx
            self._trained_target.active.train(self.input.frame_size, self.input.detections)

        if self._params.profile:
            self.profiler.enable()

        tee_frame_gap = int(self.input.n_frames / 5)
        status_msg = None
        tb_writer = None

        if self._tb_path is not None:
            self._logger.info(f'Saving tensorboard summary to: {self._tb_path}')
            tb_writer = SummaryWriter(log_dir=self._tb_path)
            self._trained_target.set_tb_writer(tb_writer)

        avg_fps = 0
        fps_count = 0
        for frame_id in range(self.input.n_frames):
            # verbosity and debugging
            if self._params.verbose:
                print('\n\nframe {:d}, targets {:d}'.format(frame_id + 1, self.n_live))

            """get current frame"""
            if self.input.params.batch_mode:
                curr_frame = self.input.all_frames[frame_id]
            else:
                """first frame was read during pipeline initialization"""
                if frame_id > 0 and not self.input.update():
                    raise IOError('Input image {:d} could not be read'.format(frame_id))
                curr_frame = self.input.curr_frame

            """get current detections"""
            det_ids = self.input.detections.idx[frame_id]
            if det_ids is not None:
                curr_det_data = self.input.detections.data[det_ids, :]
                if self._params.enable_nms:
                    curr_det_data = self._non_maximum_suppression(curr_det_data)
                n_detections = curr_det_data.shape[0]
            else:
                curr_det_data = np.empty(shape=(0, 0))
                n_detections = 0

            track_start_t = time.time()

            """process current frame"""
            if not self.update(frame_id, curr_frame, curr_det_data, n_detections,
                               annotations=annotations, msg=status_msg):
                return False

            end_t = time.time()
            try:
                fps = 1.0 / (end_t - track_start_t)

                fps_count += 1

                avg_fps += (fps - avg_fps) / fps_count
            except ZeroDivisionError:
                fps = np.inf


            """collect diagnostics information"""
            self.lost_seq_ids = []
            self.lost_ids = []
            self.lost_targets = []
            self.tracked_ids = []
            self.tracked_targets = []

            for i, t in enumerate(self.targets):
                if t.state == MDPStates.lost:
                    self.lost_seq_ids.append(i)
                    self.lost_ids.append(t.id_)
                    self.lost_targets.append(t)
                elif t.state == MDPStates.tracked:
                    self.tracked_ids.append(t.id_)
                    self.tracked_targets.append(t)

            self._n_lost = len(self.lost_targets)
            self._n_tracked = len(self.tracked_targets)

            if tb_writer is not None:
                tb_writer.add_scalar(f'{self.input.seq_name}/n_lost', self._n_lost, frame_id)
                tb_writer.add_scalar(f'{self.input.seq_name}/n_tracked', self._n_tracked, frame_id)

                if annotations is not None:
                    curr_ann_idx = annotations.idx[frame_id]
                    if curr_ann_idx is None:
                        """no annotations in this frame"""
                        n_gt_objects = 0
                    else:
                        n_gt_objects = curr_ann_idx.size
                    tb_writer.add_scalar(f'{self.input.seq_name}/GT', n_gt_objects, frame_id)

            """apply heuristic criteria for removing lost targets"""
            if self._params.max_lost_ratio > 0:
                max_lost = math.ceil(self._params.max_lost_ratio * self._n_tracked)


                if self._n_lost > max_lost > 0:
                    lost_probs = [np.mean(t.lost.assoc_probabilities) if t.lost.assoc_probabilities else 1
                                  for t in self.lost_targets]
                    local_lost_ratios = [-t.history.lost_ratio for t in self.lost_targets]
                    local_lost_streak = [-t.lost.streak for t in self.lost_targets]
                    tracked_count = [t.history.n_tracked for t in self.lost_targets]
                    valid_count = [t.history.size for t in self.lost_targets]

                    lost_rm_criteria = np.stack((valid_count, tracked_count, local_lost_streak,
                                                 local_lost_ratios, lost_probs), axis=0)

                    targets_to_remove = self._n_lost - max_lost
                    # print()
                    # self._logger.info(f'removing {targets_to_remove} lost target(s) with least support')

                    sup_ids = np.lexsort(lost_rm_criteria)

                    for i in range(targets_to_remove):
                        sup_id = sup_ids[i]
                        sup_target = self.lost_targets[sup_id]
                        sup_target.set_state(MDPStates.inactive)
                        self.states[self.lost_seq_ids[sup_id]] = MDPStates.inactive
                    # print()

            status_msg = f'{self.input.seq_name} :: {frame_id + 1}/{self.input.n_frames} ' \
                f'fps: {fps:.3f} ({avg_fps:.3f}) ' \
                f'n_detections: {n_detections:4d} ' \
                f'tracked: {self._n_tracked:4d} ' \
                f'lost: {self._n_lost:4d} total: {self.n_total_targets:5d}'

            if not self.vis and (not self._params.tee_log or frame_id % tee_frame_gap == 0):
                if self._params.verbose == 2:
                    self._logger.info(status_msg)
                else:
                    sys.stdout.write('\r{}'.format(status_msg))
                    sys.stdout.flush()

            if self._params.max_targets > 0:
                assert self.n_live <= self._params.max_targets, \
                    f"\n{status_msg}\nn_active_targets: {self.n_live} " \
                        f"exceeds the max allowed: {self._params.max_targets}"
            # if self.vis:
            #     print()
            #     print(f'lost_targets: {pformat(self.lost_ids)}')
            #     print(f'tracked_targets: {pformat(self.tracked_ids)}')
            #     # print(f'sctive_targets: {pformat(self.sctive_targets)}')
            #     print()

            # for obj in gc.get_objects():
            #     try:
            #         if (torch.is_tensor(obj) and obj.device == torch.device('cuda')) or \
            #                 (hasattr(obj, 'data') and torch.is_tensor(obj.data) and obj.data.device == torch.device(
            #                     'cuda')):
            #             print(type(obj), obj.size())
            #             print()
            #     except:
            #         pass

        sys.stdout.write('\n\n')
        sys.stdout.flush()

        # process all remaining targets
        self._process_inactive_targets(np.arange(self.n_live, dtype=np.int32))
        sys.stdout.write('\n')

        if self.vis:
            self.visualizer.close()

        if self._params.profile:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())

        return True

    def _track(self, frame_id, frame, det_data, n_det):
        """
        called for each frame

        :type frame: np.ndarray
        :type frame_id: int
        :type det_data: np.ndarray
        :type n_det: int
        """

        if self._params.sort_targets:
            self.target_sort_idx = self._sort_targets()
        else:
            self.target_sort_idx = np.asarray(list(range(self.n_live)))

        self.deleted_targets = []
        inactive_idx = []
        n_inactive_targets = 0

        for i in range(self.n_live):
            associated_to_tracked = 0
            _id = self.target_sort_idx[i]
            target = self.targets[_id]

            if self._params.verbose == 3:
                self._logger.info('(before update){:d} :: Target {:d} state: {:s}'.format(
                    _id + 1, target.id_, target.state_str()))

            if target.state == MDPStates.inactive:
                inactive_idx.append(_id)
                n_inactive_targets += 1
                continue

            if target.state == MDPStates.lost:
                if self._params.hungarian:
                    # lost targets are processed separately
                    continue

                """remove detections corresponding to already processed targets"""
                filtered_det_data, _ = self._filter_detections(frame, det_data, self.target_sort_idx[:i])
            else:
                filtered_det_data = det_data

            """update target with current frame"""
            target.update(frame, frame_id, filtered_det_data,
                          associate=True, assoc_with_ann=False)

            if self._params.verbose == 3:
                self._logger.info('(after update){:d} :: Target {:d} state: {:s}'.format(
                    _id + 1, target.id_, target.state_str()))

            if target.state != MDPStates.inactive:
                if self._params.lost_heuristics and target.prev_state == MDPStates.lost:
                    target.lost_exit_heuristics(frame)

                if target.prev_state == MDPStates.tracked:
                    if self._params.tracked_heuristics:
                        target.tracked_exit_heuristics(frame)

                    if self._params.reconnect_lost and target.state == MDPStates.lost:
                        """
                        try to reconnect recently lost target
                        """
                        filtered_det_data, _ = self._filter_detections(frame, det_data, self.target_sort_idx[:i])
                        valid_tracked_location = 0

                        if self.tracked_as_detection:
                            tracked_location = target.tracked.location.squeeze()
                            if not np.isfinite(tracked_location).all() or np.any(np.isnan(tracked_location)):
                                self._logger.warning('ignoring invalid tracked_location: {}'.format(tracked_location))
                                valid_tracked_location = 0
                            else:
                                valid_tracked_location = 1
                                tracked_location_data = np.concatenate(
                                    ((frame_id, target.id_), tracked_location, (1, -1, -1, -1)), axis=0).reshape(
                                    (1, 10))
                                if filtered_det_data.size:
                                    filtered_det_data = np.concatenate((filtered_det_data, tracked_location_data),
                                                                       axis=0)
                                else:
                                    filtered_det_data = tracked_location_data

                        n_filtered_det = filtered_det_data.shape[0]
                        target.update(frame, frame_id, filtered_det_data,
                                      associate=True, assoc_with_ann=False)

                        if self._params.lost_heuristics:
                            target.lost_exit_heuristics(frame)

                        if self.tracked_as_detection and valid_tracked_location:
                            assoc_det_id = target.lost.assoc_det_id
                            if target.state == MDPStates.tracked and assoc_det_id == n_filtered_det - 1:
                                associated_to_tracked = 1
                                if self._params.verbose:
                                    print(f'target {target.id_} :: '
                                          f'assoc_to_tracked_counter: {target.assoc_to_tracked_counter} / '
                                          f'{self.tracked_as_detection}')

                if associated_to_tracked:
                    target.assoc_to_tracked_counter += 1
                else:
                    target.assoc_to_tracked_counter = 0

                if target.assoc_to_tracked_counter >= self.tracked_as_detection > 0:
                    target.state = MDPStates.inactive

            self._update_target_variables(_id)

            if target.state == MDPStates.inactive:
                inactive_idx.append(_id)
                n_inactive_targets += 1

        if self._params.hungarian:
            """associate detections with lost targets"""

            # targets not in the lost state
            idx = np.flatnonzero(self.states != MDPStates.lost)

            # get detections not associated with tracked or inactive targets
            filtered_det_data, filtered_idx = self._filter_detections(frame, det_data, idx)
            # targets in the lost state
            lost_idx = np.flatnonzero(self.states == MDPStates.lost)
            _n_lost = lost_idx.size
            _n_det = filtered_det_data.shape[0]
            if _n_lost > 0 and _n_det > 0:
                dist = np.zeros((_n_det, _n_lost))
                for i, _id in enumerate(lost_idx):
                    """partial target update  to compute the corresponding association scores / distances"""
                    self.targets[_id].update(frame, frame_id, filtered_det_data,
                                             associate=False, assoc_with_ann=False)
                    self.targets[_id].lost.get_distances(dist[:, i])
                assignments, cost = pyHungarian.get(dist)
                """associate using externally provided detection IDs"""
                for i in range(assignments.size):
                    _id = lost_idx[i]
                    """pyHungarian returns 1-based assignment indices"""
                    self.targets[_id].associate(assoc_det_id=int(assignments[i] - 1))
                    self._update_target_variables(_id)

        filtered_det_data, filtered_idx = self._filter_detections(frame, det_data)

        # print('det_data: ', det_data)
        # print('filtered_det_data: ', filtered_det_data)

        n_filtered_det = filtered_det_data.shape[0]
        if self._params.verbose:
            self._logger.info('Done {}/{} frames n_det: {} n_filtered_det: {} '
                              'active: {} lost: {} total: {}'.format(
                frame_id + 1, self.input.n_frames, n_det, n_filtered_det,
                self.n_live, self._n_lost, self.n_total_targets))

        return filtered_idx, n_inactive_targets, inactive_idx

    def update(self, frame_id, frame, det_data, n_det, gates=None,
               result=None, annotations=None, msg=None):
        """
        called for each frame

        :type frame: np.ndarray
        :type frame_id: int
        :type det_data: np.ndarray
        :type n_det: int
        :type gates: dict[int:Gate] | None
        :type result: list | None
        :rtype: bool
        """

        filtered_idx = n_inactive_targets = inactive_idx = None

        if not self._params.res_from_gt:
            filtered_idx, n_inactive_targets, inactive_idx = self._track(frame_id, frame, det_data, n_det)

        """try to create a new target for each filtered unassociated detection;
        also associate GT with detections to save corresponding results if enabled 
        or use it for decision statistics otherwise
        """
        self._add_targets(frame_id, frame, det_data, filtered_idx, annotations)

        if self._params.res_from_gt:
            return True

        if self._params.resolve_conflicts:
            self._resolve_target_conflicts(det_data, n_det)

        if result is not None:
            """for the tracking server"""
            for target in self.targets:
                if target.state != MDPStates.tracked:
                    continue
                result.append([target.id_, target.frame_id, target.location, target.max_iou_det_score])

        if n_inactive_targets > self._params.max_inactive_targets:
            self._process_inactive_targets(np.asarray(inactive_idx))

        # if self._params.save_raw:
        target_data_raw = np.empty((self.n_live, 11), dtype=np.float64)
        for i in range(self.n_live):
            target_data_raw[i, :] = self.targets[i].get_data(True)
            """1-based frame IDs"""
            target_data_raw[i, 0] += 1
        if self._results_raw is None:
            self._results_raw = target_data_raw
        else:
            self._results_raw = np.concatenate((self._results_raw, target_data_raw), axis=0)

        if self.vis and not self._visualize(frame_id, frame, det_data, gates, msg=msg):
            raise AssertionError('Visualizer update failed')
        return True

    def _add_targets(self, frame_id, frame, det_data, filtered_idx, annotations):

        n_det = det_data.shape[0]

        if n_det == 0:
            return

        if not self._params.res_from_gt:
            if not self._params.vis and filtered_idx.size == 0:
                return

        frame_disp = colors = None
        if self._params.vis == 2:
            frame_disp = copy_rgb(frame)
            colors = {
                'tp': 'forest_green',
                'fp_background': 'red',
                'fp_deleted': 'cyan',
                'fp_apart': 'magenta',
                'fp_concurrent': 'blue',
            }

        ann_obj_id, ann_traj_id, ann_status = associate_gt_to_detections(
            frame, frame_id, annotations, det_data, n_det,
            self._ann_to_targets, self._start_end_frame_ids,
            self._params.ann_assoc_method, self._params.ann_iou_threshold,
            self._params.vis, self._params.verbose, self._logger)

        if self._params.res_from_gt:
            for det_id in range(n_det):
                _ann_obj_id = ann_obj_id[det_id]
                _ann_traj_id = ann_traj_id[det_id]
                _ann_status = ann_status[det_id]

                if _ann_obj_id is None:
                    continue

                assert _ann_status != "fp_background", "valid object ID found for fp_background detection"

                curr_det_data = det_data[det_id, :]

                x, y, w, h = curr_det_data[2:6]

                target_data = [frame_id + 1, _ann_obj_id, x, y, w, h, 1, -1, -1, -1]

                if self._params.save_debug_info:
                    """add state column
                    """
                    target_data.append(MDPStates.tracked)

                    if self._params.save_debug_info == 2:
                        """add validity status column
                        """
                        target_data.append(1)

                target_data = np.asarray(target_data).reshape(1, len(target_data))

                if self._results is None:
                    self._results = target_data
                else:
                    self._results = np.concatenate((self._results, target_data), axis=0)

            return

        self.failed_targets = []

        all_sampled_boxes = []

        """all ground truth boxes in this frame"""
        if annotations is not None and annotations.idx[frame_id] is not None:
            gt_boxes = annotations.data[annotations.idx[frame_id], :][:, 2:6]
        else:
            gt_boxes = None

        for filter_id in range(filtered_idx.size):

            det_id = filtered_idx[filter_id]

            curr_det_data = det_data[det_id, :]

            _ann_obj_id = ann_obj_id[det_id]
            _ann_traj_id = ann_traj_id[det_id]
            _ann_status = ann_status[det_id]

            # ID is 1-based
            new_target = self._trained_target.spawn(
                self.n_total_targets + 1, frame_id, frame,
                curr_det_data, self._logger, annotations, _ann_traj_id, _ann_status)  # type: Target

            # if self.pause_for_debug:
            #     self._logger.debug('paused')

            if new_target is None:
                is_dotted = 1
                failed_targets_loc = np.copy(det_data[det_id, 2:6])

                self.failed_targets.append(failed_targets_loc)
                """target creation failed - transition to inactive"""
                if self._params.verbose:
                    self._logger.info('Target {:d} not added - transition to inactive'.format(det_id))
            else:
                # target creation succeeded
                is_dotted = 0
                self.targets.append(new_target)
                self._init_target_variables()
                self.n_live += 1
                self.n_total_targets += 1
                if _ann_obj_id is not None:
                    self._ann_to_targets[_ann_obj_id] = [new_target.id_, new_target]
                    self._targets_to_ann[new_target.id_] = _ann_obj_id

                if self._params.verbose:
                    print('Target {:d} with id {:d} added'.format(det_id, new_target.id_))

            if self._params.vis == 2 and _ann_status is not None:
                draw_box(frame_disp, curr_det_data[2:6], color=colors[_ann_status], is_dotted=is_dotted)

            # if _ann_status is not None:
            #     self.stats['active'][_ann_status] += self.targets[det_id].active.get_stats()
            #     self.stats['active']['combined'] += self.targets[det_id].active.get_stats()

            self._trained_target.active.add_synthetic_samples(frame, curr_det_data, gt_boxes,
                                                              ann_status, all_sampled_boxes)

        if frame_disp is not None:
            annotate_and_show('add_targets', frame_disp)

    def _init_target_variables(self):
        """
        :rtype: None
        """
        if self.n_live == 0:
            self.ids = np.array((self.targets[0].id_,))
            self.states = np.array((self.targets[0].state,))
            self.locations = self.targets[0].location.reshape((1, 4))
            self.tracked_streaks = np.array((self.targets[0].tracked.streak,))
        else:
            self.ids = np.concatenate((self.ids, (self.targets[-1].id_,)))
            self.states = np.concatenate((self.states, (self.targets[-1].state,)))
            self.locations = np.concatenate((self.locations, self.targets[-1].location.reshape((1, 4))), axis=0)
            self.tracked_streaks = np.concatenate((self.tracked_streaks, (self.targets[-1].tracked.streak,)))

    def _update_target_variables(self, target_id):
        """
        :type target_id: int
        :rtype: None
        """
        self.ids[target_id] = self.targets[target_id].id_
        self.states[target_id] = self.targets[target_id].state
        self.locations[target_id, :] = self.targets[target_id].location.reshape((1, 4))
        self.tracked_streaks[target_id] = self.targets[target_id].tracked.streak

    def _remove_target_variables(self, idx):
        """
        :type idx: np.ndarray
        :rtype: None
        """
        self.ids = np.delete(self.ids, idx)
        self.states = np.delete(self.states, idx)
        self.locations = np.delete(self.locations, idx, axis=0)
        self.tracked_streaks = np.delete(self.tracked_streaks, idx)

    def _sort_targets(self):
        """
        :rtype: np.ndarray | None
        """
        if self.n_live == 0:
            return None

        idx1 = np.flatnonzero(self.tracked_streaks > self._params.target_sort_sep)
        """stable sorting to maintain correspondence with the original code"""
        ind = np.argsort(self.states[idx1], kind='mergesort')
        idx1 = idx1[ind]

        idx2 = np.flatnonzero(self.tracked_streaks <= self._params.target_sort_sep)
        """stable sorting to maintain correspondence with the original code"""
        ind = np.argsort(self.states[idx2], kind='mergesort')
        idx2 = idx2[ind]

        # if self.pause_for_debug:
        #     self._logger.debug('paused')

        target_sort_idx = np.concatenate((idx1, idx2), axis=0)
        return target_sort_idx

    def _visualize(self, frame_id, frame, det_data, gates=None, msg=None):
        """
        :type frame_id: int
        :type frame: np.ndarray
        :type frame: np.ndarray
        :type det_data: np.ndarray
        :type gates: dict[int:Gate] | None
        :rtype: None
        """
        frame_data = {}
        if self._params.visualizer.mode[0]:
            tracked_data = np.empty((self.n_live, 11), dtype=np.float64)
            for i in range(self.n_live):
                tracked_data[i, :] = self.targets[i].get_data(True)
            frame_data[ObjTypes.tracking_result] = tracked_data
        else:
            frame_data[ObjTypes.tracking_result] = None

        if self._params.visualizer.mode[1]:
            frame_data[ObjTypes.detection] = det_data
        else:
            frame_data[ObjTypes.detection] = None

        if self._params.visualizer.mode[2]:
            ann_ids = self.annotations.idx[frame_id]
            if ann_ids is not None:
                curr_ann_data = self.input.annotations.data[ann_ids, :]
            else:
                curr_ann_data = np.empty(shape=(0, 0))
            frame_data[ObjTypes.annotation] = curr_ann_data
        else:
            frame_data[ObjTypes.annotation] = None
        return self.visualizer.update(frame_id, frame, frame_data, gates, self.deleted_targets,
                                      self.failed_targets, msg=msg)

    def _filter_detections(self, frame, det_data, target_ids=None):
        """
        Filter detections by getting rid of those that have high overlaps with the last known locations of one or more
        existing Targets; comparison is restricted to only specific targets if target_ids is provided

        :type det_data: np.ndarray
        :type target_ids: np.ndarray | None
        :rtype: np.ndarray
        """
        n_det = det_data.shape[0]
        all_idx = np.asarray(list(range(n_det)))

        if not self._params.filter_detections or self.n_live == 0 or n_det == 0:
            return det_data, all_idx

        if target_ids is None:
            states = self.states
            locations = self.locations.reshape((-1, 4))
        else:
            if target_ids.size == 0:
                return det_data, all_idx

            states = self.states[target_ids]
            locations = self.locations[target_ids, :].reshape((-1, 4))

        """Compute overlap between all detections and all tracked targets"""
        tracked_idx = np.flatnonzero(states == MDPStates.tracked)
        if tracked_idx.size == 0:
            return det_data, all_idx

        if self._params.filtering_method == 0:

            """
            iterative mutual maximum overlap between detections and targets
            """
            det_to_gt, gt_to_det, unassociated_dets, unassociated_gts = find_associations(
                frame, det_data[:, 2:6], locations[tracked_idx, :], self._params.filter_iou_thresh, use_hungarian=0)

            filtered_idx = unassociated_dets
        elif self._params.filtering_method == 1:
            """
            annoying old heuristics based filtering with iou and ioa comparisons 
            between individual pairs of detections and targets
            """
            iou = np.empty((det_data.shape[0], tracked_idx.size))
            ioa_1 = np.empty((det_data.shape[0], tracked_idx.size))
            compute_overlaps_multi(iou, ioa_1, None, det_data[:, 2:6], locations[tracked_idx, :], self._logger)
            max_iou = np.amax(iou, axis=1)
            max_ioa_1 = np.sum(ioa_1, axis=1)
            # sum_ioa_1 = np.sum(ioa_1, axis=1)
            filtered_idx = np.flatnonzero(np.logical_and(max_iou < self._params.filter_iou_thresh,
                                                         max_ioa_1 < self._params.filter_ioa_thresh))
        else:
            raise AssertionError('invalid filtering_method: {}'.format(self._params.filtering_method))

        filtered_det_data = det_data[filtered_idx, :]
        return filtered_det_data, filtered_idx

    def _resolve_target_conflicts(self, curr_det_data, n_detections):
        """
        Find all pairs of tracked targets with overlap > 0.7 and remove the one with shorter tracked streak assuming
        that they are both duplicate targets for the same object and the one that has been Tracked longer is more reliable
        If both targets have identical tracked streaks, then remove the one with smaller overlap with the maximally
        overlapping detection

        :type curr_det_data: np.ndarray
        :type n_detections: int
        :rtype: int
        """

        """only look amongst targets currently in tracked state"""
        tracked_idx = np.flatnonzero(self.states == MDPStates.tracked)
        n_tracked_idx = tracked_idx.size

        if n_tracked_idx == 0:
            return

        ioa = np.empty((n_tracked_idx, n_tracked_idx))
        compute_self_overlaps(None, ioa, self.locations[tracked_idx, :])
        sup_flags = np.zeros((n_tracked_idx,), dtype=np.bool)
        for i in range(n_tracked_idx):
            """discard already suppressed targets"""
            ioa[sup_flags, :] = 0

            """target having  maximal overlap with the current one"""
            max_ioa_idx = np.argmax(ioa[i, :])
            max_ioa = ioa[i, max_ioa_idx]

            if max_ioa <= self._params.overlap_sup:
                continue

            tracked_sreak_1 = self.tracked_streaks[tracked_idx[i]]
            tracked_sreak_2 = self.tracked_streaks[tracked_idx[max_ioa_idx]]

            """two targets with highly overlapping locations"""
            if tracked_sreak_1 > tracked_sreak_2:
                """current target tracked longer than the maximally overlapping one so discard the latter"""
                sup_id = max_ioa_idx
            elif tracked_sreak_2 > tracked_sreak_1:
                """current target tracked shorter than the maximally overlapping one so discard the former"""
                sup_id = i
            else:
                if n_detections > 0:
                    """suppress the target with smaller iou with the maximally overlapping detection"""
                    det_iou = np.empty((2, n_detections))
                    compute_overlaps_multi(det_iou, None, None, self.locations[(i, max_ioa_idx), :],
                                           curr_det_data[:, 2:6], self._logger)
                    max_det_iou = np.amax(det_iou, axis=1)
                    if max_det_iou[0] > max_det_iou[1]:
                        sup_id = max_ioa_idx
                    else:
                        sup_id = i
                else:
                    """nothing can be done"""
                    continue

            sup_target = self.targets[tracked_idx[sup_id]]
            if sup_target.n_frames <= 1:
                """suppressed target has just been added so likely a false positive"""
                sup_target.set_state(MDPStates.inactive)
                self.states[tracked_idx[sup_id]] = MDPStates.inactive
                # n_supp_inactive += 1
                if self._params.verbose:
                    print('target {:d} suppressed to inactive state'.format(self.targets[tracked_idx[sup_id]].id_))

            else:
                sup_target.set_state(MDPStates.lost)
                self.states[tracked_idx[sup_id]] = MDPStates.lost
                if self._params.verbose:
                    print('target {:d} suppressed to lost state'.format(self.targets[tracked_idx[sup_id]].id_))
            sup_flags[sup_id] = True

        # return n_supp_inactive

    def _non_maximum_suppression(self, det_data):
        """
        :type det_data: np.ndarray
        :rtype: np.ndarray
        """
        n_det = det_data.shape[0]
        if n_det == 0:
            return det_data
        x1 = det_data[:, 2].flatten()
        y1 = det_data[:, 3].flatten()
        x2 = (det_data[:, 2] + det_data[:, 4]).flatten()
        y2 = (det_data[:, 3] + det_data[:, 5]).flatten()
        s = det_data[:, 6].flatten()
        areas = np.multiply(det_data[:, 4], det_data[:, 5]).flatten()
        sort_idx = np.argsort(s, kind='mergesort')[::-1]
        n = sort_idx.size
        pick = np.ones((n,))
        for i in range(n):
            ii = sort_idx[i]
            for j in range(i):
                jj = sort_idx[j]
                if pick[jj]:
                    xx1 = max(x1[ii], x1[jj])
                    yy1 = max(y1[ii], y1[jj])
                    xx2 = min(x2[ii], x2[jj])
                    yy2 = min(y2[ii], y2[jj])
                    w = xx2 - xx1 + 1
                    h = yy2 - yy1 + 1
                    if w > 0 and h > 0:
                        area_inter = w * h
                        o = area_inter / (areas[ii] + areas[jj] - area_inter)
                        o1 = area_inter / areas[ii]
                        o2 = area_inter / areas[jj]
                        if o > self._params.nms_overlap_thresh[0] or \
                                o1 > self._params.nms_overlap_thresh[1] or \
                                o2 > self._params.nms_overlap_thresh[2]:
                            pick[ii] = 0
                            break
        pick = np.flatnonzero(pick == 1)
        nms_det_data = det_data[pick, :]
        return nms_det_data

    def _process_inactive_targets(self, inactive_idx):
        """
        :type inactive_idx: np.ndarray
        :rtype: None
        """
        for i in range(inactive_idx.size):
            history = self.targets[inactive_idx[i]].history

            tracked_idx = np.flatnonzero(np.equal(history.states, MDPStates.tracked))
            tracked_traj_len = tracked_idx.size
            traj_len = tracked_traj_len

            is_valid = 1
            if traj_len <= self._params.min_trajectory_len:
                """target trajectory is too short to be included in the final results"""
                if self._params.save_debug_info != 2:
                    continue
                is_valid = 0

            target_data = np.concatenate(
                (
                    (history.frame_ids[tracked_idx] + 1).reshape((tracked_traj_len, 1)),
                    history.ids[tracked_idx].reshape((tracked_traj_len, 1)),
                    history.locations[tracked_idx, :].reshape((tracked_traj_len, 4)),
                    history.scores[tracked_idx].reshape((tracked_traj_len, 1)),
                    np.tile((-1, -1, -1), (tracked_traj_len, 1))
                ),
                axis=1).reshape((tracked_traj_len, 10))

            if self._params.save_debug_info:
                """add state column
                """
                state_column = np.tile((MDPStates.tracked,), (tracked_traj_len, 1))

                target_data = np.concatenate(
                    (
                        target_data,
                        state_column
                    ),
                    axis=1)

                lost_idx = np.flatnonzero(np.equal(history.states, MDPStates.lost))
                lost_traj_len = lost_idx.size
                traj_len += lost_traj_len

                if lost_traj_len:
                    lost_data = np.concatenate(
                        (
                            (history.frame_ids[lost_idx] + 1).reshape((lost_traj_len, 1)),
                            history.ids[lost_idx].reshape((lost_traj_len, 1)),
                            history.locations[lost_idx, :].reshape((lost_traj_len, 4)),
                            history.scores[lost_idx].reshape((lost_traj_len, 1)),
                            np.tile((-1, -1, -1, MDPStates.lost), (lost_traj_len, 1))
                        ),
                        axis=1).reshape((lost_traj_len, 11))
                    target_data = np.concatenate((target_data, lost_data), axis=0)

            if self._params.save_debug_info == 2:
                """add validity status column
                """
                validity_status_column = np.tile((is_valid,), (traj_len, 1))
                target_data = np.concatenate(
                    (
                        target_data,
                        validity_status_column
                    ),
                    axis=1)

            if self._results is None:
                self._results = target_data
            else:
                self._results = np.concatenate((self._results, target_data), axis=0)

        # remove target and all corresponding variables
        # self.logger.debug('inactive_idx.shape: %(1)s', {'1': inactive_idx.shape})
        # self.logger.debug('inactive_idx: %(1)s', {'1': inactive_idx})

        if inactive_idx.size > 1:
            sorted_idx = sorted(list(inactive_idx.astype(np.int32).squeeze()), reverse=True)
        else:
            sorted_idx = inactive_idx

        for i in sorted_idx:
            _id = self.targets[i].id_
            self.deleted_targets.append(_id)
            if _id in self._targets_to_ann:
                _ann_obj_id = self._targets_to_ann[_id]
                self._ann_to_targets[_ann_obj_id][1] = None
                del self._targets_to_ann[_id]
            self._start_end_frame_ids[_id] = [self.targets[i].start_frame_id, self.targets[i].frame_id]

            # _ann_status = self.targets[i].ann_status
            # if _ann_status is not None:
            #     for _state in ('lost', 'tracked'):
            #         target_state = getattr(self.targets[i], _state)  # type: Policy
            #         self.stats[_state][_ann_status] += target_state.get_stats()
            #         self.stats[_state]['combined'] += target_state.get_stats()

            del self.targets[i]
        self._remove_target_variables(inactive_idx)
        self.n_live -= inactive_idx.size
        # self.n_total_targets -= inactive_idx.size

    # def get_stats(self):
    #     stats = {
    #         _state: getattr(getattr(self._trained_target, _state), 'stats').copy(deep=True)
    #         for _state in ('active', 'lost', 'tracked')
    #     }
    #     percent_stats = {
    #         _state: getattr(getattr(self._trained_target, _state), 'percent_stats').copy(deep=True)
    #         for _state in ('active', 'lost', 'tracked')
    #     }
    #     return stats, percent_stats

    def eval(self, load_fname, eval_path, eval_dist_type):
        """
        :type load_fname: str
        :type eval_path: str
        :type eval_dist_type: int
        :rtype: mm.MOTAccumulator | None
        """

        assert self.input.annotations is not None, "annotations have not been loaded"
        assert self.input.track_res is not None, "tracking results have not been loaded"

        stats_dir = linux_path(os.path.dirname(eval_path), 'stats')

        os.makedirs(stats_dir, exist_ok=True)

        check_states = 1
        if self._params.res_from_gt:
            check_states = 0

        seq_name = os.path.splitext(os.path.basename(load_fname))[0]
        if self._params.devkit or self._params.hota:
            gtfiles = [self.input.annotations.get_mot_compatible_file(), ]
            tsfiles = [self.input.track_res.get_mot_compatible_file(check_states=check_states), ]
            n_frames = [self.input.n_frames, ]
            sequences = [seq_name, ]

            datadir = os.path.dirname(self.input.annotations.path)
            benchmark_name = os.path.basename(os.path.dirname(datadir))

            acc = (gtfiles, tsfiles, n_frames, sequences, datadir, benchmark_name)

            self._logger.info('deferring evaluation for combined results')

            _eval = eval_str = None

            # from evaluation.devkit.MOT.evalMOT import MOT_evaluator
            # eval = MOT_evaluator()
            # _overall_Results, _results, _eval, eval_str = eval.run(*acc)
        else:

            _eval, eval_str, acc = self.input.annotations.get_mot_metrics(self.input.track_res,
                                                                          seq_name, eval_dist_type)

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        if self._params.tee_log:
            time_stamp = '{} :: {}'.format(time_stamp, self._params.tee_log)

        if self._params.use_annotations and self._trained_target is not None:
            self._get_stats(stats_dir, load_fname, time_stamp)

        if eval_str is not None:
            print('\n' + eval_str + '\n')
            if _eval is None:
                return None
            mot_metrics_to_file((eval_path,), _eval, load_fname, seq_name,
                                mode='a', time_stamp=time_stamp, devkit=self._params.devkit)

        self._acc_dict[self.input.seq_name] = acc

        return acc

    def accumulative_eval(self, load_dir, eval_path, _logger):
        """

        :param str load_dir:
        :param str eval_path:
        :param CustomLogger _logger:
        :return:
        """
        accumulative_eval_path = self._params.accumulative_eval_path

        eval_dir = os.path.dirname(eval_path)
        accumulative_eval_dir = os.path.dirname(accumulative_eval_path)

        devkit_accumulative_eval_path = add_suffix(accumulative_eval_path, 'devkit')
        devkit_eval_path = add_suffix(eval_path, 'devkit')

        hota_accumulative_eval_path = add_suffix(accumulative_eval_path, 'hota')
        hota_eval_path = add_suffix(eval_path, 'hota')

        # hota_mot_accumulative_eval_path = add_suffix(accumulative_eval_path, 'hota_mot')
        # hota_mot_eval_path = add_suffix(eval_path, 'hota_mot')

        # hota_mot_accumulative_eval_path = hota_accumulative_eval_path + 'm'
        # hota_mot_eval_path = hota_eval_path + 'm'

        accumulative_stats_dir = linux_path(accumulative_eval_dir, 'stats')
        stats_dir = linux_path(eval_dir, 'stats')

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        if self._params.tee_log:
            time_stamp = '{} :: {}'.format(time_stamp, self._params.tee_log)

        if self._params.use_annotations and len(self._stats_dict) > 1:
            self._get_accumulative_stats(stats_dir, accumulative_stats_dir, load_dir, time_stamp)

        if not self._acc_dict or len(self._acc_dict) == 0:
            return

        devkit_overall_results = devkit_results = devkit_summary = None
        hota_summaries = hota_res_dict = hota_devkit_df = None
        devkit_metric_names = hota_mot_metric_names = hota_metric_names = None

        if self._params.devkit or self._params.hota:
            gtfiles = []
            tsfiles = []

            sequences = []
            n_frames = []
            for _seq in self._acc_dict:
                _args = self._acc_dict[_seq]
                _gtfiles, _tsfiles, _n_frames, _sequences, _datadir, _benchmark_name, = _args
                gtfiles.append(_gtfiles[0])
                tsfiles.append(_tsfiles[0])
                sequences.append(_sequences[0])
                n_frames.append(_n_frames[0])

            datadir = os.path.dirname(self.input.annotations.path)
            benchmark_name = os.path.basename(os.path.dirname(datadir))

            if self._params.hota:
                from evaluation.hota_metrics.hota_evaluator import HOTA_evaluator
                hota_eval = HOTA_evaluator()

                output_dir = linux_path(eval_dir, 'hota_metrics')
                hota_res, hota_msg, hota_res_dict, hota_summaries = hota_eval.run(
                    gtfiles, tsfiles, datadir, sequences, benchmark_name, n_frames, output_dir)

                devkit_metric_names, hota_mot_metric_names, hota_metric_names, combined_metric_names = \
                    hota_eval.metric_names()

                # print()

            if self._params.devkit:
                from evaluation.devkit.Evaluator import MotEvaluator
                devkit_eval = MotEvaluator()
                devkit_overall_results, devkit_results, devkit_summary, devkit_strsummary = devkit_eval.run(
                    gtfiles, tsfiles, datadir, sequences, benchmark_name)

            if self._params.hota and self._params.devkit:
                devkit_cmb_results = {}
                for seq in devkit_results:
                    devkit_cmb_results[seq] = devkit_results[seq].__dict__

                devkit_cmb_results['COMBINED_SEQ'] = devkit_overall_results.__dict__
                hota_devkit_df = combine_hota_devkit_results(devkit_cmb_results, hota_res_dict,
                                                             devkit_metric_names, hota_mot_metric_names)

            for _seq in self._acc_dict:
                _args = self._acc_dict[_seq]
                _gtfiles, _tsfiles, _datadir, _sequences, _benchmark_name, _ = _args
                if self._params.devkit:
                    mot_metrics_to_file((devkit_eval_path,), devkit_summary, _tsfiles[0], _sequences[0], mode='a',
                                        time_stamp=time_stamp, verbose=0, devkit=self._params.devkit)
                if self._params.hota:
                    hota_combined_summary = hota_summaries['HOTA'][_gtfiles[0]]
                    metric_names = hota_metric_names
                    if not self._params.devkit:
                        hota_mot_summary = hota_res_dict[_gtfiles[0]]
                        hota_combined_summary.update(hota_mot_summary)
                        metric_names = combined_metric_names

                        # hota_metrics_to_file((hota_mot_eval_path,), hota_mot_summary, _tsfiles[0],
                        #                      hota_mot_metric_names, mode='a', time_stamp=time_stamp, verbose=0)

                    hota_metrics_to_file((hota_eval_path,), hota_combined_summary, _tsfiles[0],
                                         metric_names, mode='a', time_stamp=time_stamp, verbose=0)

        else:
            devkit_summary, devkit_strsummary = combined_motmetrics(self._acc_dict, _logger)

        if self._params.devkit:
            mot_metrics_to_file((devkit_eval_path, devkit_accumulative_eval_path), devkit_summary, load_dir, 'OVERALL',
                                time_stamp=time_stamp, devkit=self._params.devkit)
            if self._params.hota:
                devkit_hota_eval_path = add_suffix(eval_path, 'devkit_hota')
                # print(devkit_hota_eval_path)
                write_df(hota_devkit_df, devkit_hota_eval_path, title='{}\t{}'.format(time_stamp, load_dir),
                         index_label='sequence/type')

        if self._params.hota:
            hota_combined_summary = hota_summaries['HOTA']['COMBINED_SEQ']
            metric_names = hota_metric_names
            if not self._params.devkit:
                hota_mot_summary = hota_res_dict['COMBINED_SEQ']
                metric_names = combined_metric_names
                # hota_metrics_to_file((hota_mot_eval_path, hota_mot_accumulative_eval_path),
                #                      hota_mot_summary, load_dir,
                #                      hota_mot_metric_names, mode='a', time_stamp=time_stamp, verbose=0)
                hota_combined_summary.update(hota_mot_summary)

            hota_metrics_to_file((hota_eval_path, hota_accumulative_eval_path), hota_combined_summary,
                                 load_dir, metric_names, time_stamp=time_stamp)

    def _get_stats(self, stats_dir, load_fname, time_stamp):
        _stats_txt = '\n'
        _percent_stats_txt = '\n'
        stats_dict = {}
        percent_stats_dict = {}

        combined_stats = {k: [] for k in ['correct', 'total']}
        stats_summary_types = ['policy', 'correct', 'incorrect', 'total']
        fp_background_stats = {k: [] for k in stats_summary_types}
        tp_stats = {k: [] for k in stats_summary_types}

        enable_stats = 0
        for _state in ('active', 'lost', 'tracked'):
            policy = getattr(self._trained_target, _state)  # type: PolicyBase
            if policy._params.enable_stats:
                enable_stats = 1
                break
        if not enable_stats:
            self._logger.warning('stats are disabled for all policies')
            return

        for _state in ('active', 'lost', 'tracked'):
            # _stats_txt += f"{_state}\n"
            # _percent_stats_txt += f"{_state}\n"

            policy = getattr(self._trained_target, _state)  # type: PolicyBase
            stats = policy.stats
            percent_stats = policy.pc_stats
            status_percent_stats = pd.DataFrame(
                np.full((len(PolicyBase.Decision.types), len(AnnotationStatus.types[1:])), np.nan, dtype=np.float32),
                columns=AnnotationStatus.types[1:], index=PolicyBase.Decision.types)

            valid_rows = np.flatnonzero(stats.values[:, 0] > 0)
            arr1 = stats.values[valid_rows, 1:]
            arr2 = stats.values[valid_rows, 0].reshape((-1, 1))
            status_percent_stats.values[valid_rows, :] = (arr1 / arr2) * 100.0

            combined_stats['total'].append(stats['combined']['total'])
            combined_stats['correct'].append(percent_stats['combined']['correct'])
            fp_background_stats['policy'].append(percent_stats['fp_background']['correct'])
            tp_stats['policy'].append(percent_stats['tp']['correct'])

            for k in stats_summary_types[1:]:
                fp_background_stats[k].append(status_percent_stats['fp_background'][k])
                tp_stats[k].append(status_percent_stats['tp'][k])

            status_percent_stats.dropna(how='all', inplace=True)

            write_df(stats, linux_path(stats_dir, f'{_state}.log'), _state, load_fname)
            write_df(percent_stats, linux_path(stats_dir, f'{_state}_percent.log'), _state, load_fname)
            write_df(status_percent_stats, linux_path(stats_dir, f'{_state}_status_percent.log'), _state,
                     load_fname)
            stats.index.name = _state

            if self._params.print_stats:
                print()
                print_df(stats, _state)
                print_df(percent_stats)
                print_df(status_percent_stats)

            stats_dict[_state] = stats.copy(deep=True)
            percent_stats_dict[_state] = percent_stats.copy(deep=True)

        self._stats_dict[self.input.seq_name] = stats_dict
        self._percent_stats_dict[self.input.seq_name] = percent_stats_dict

        # print(_stats_txt)
        # print(_percent_stats_txt)

        # with open(eval_path, 'a') as fid:
        #     fid.write(_stats_txt)
        #     fid.write(_percent_stats_txt)

        self._stats_summary_to_file(combined_stats, fp_background_stats, tp_stats, load_fname, (stats_dir,),
                                    time_stamp=time_stamp)

        self._logger.info(f'Saved decision stats to: {stats_dir}')

    def _get_accumulative_stats(self, stats_dir, accumulative_stats_dir, load_dir, time_stamp):

        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(accumulative_stats_dir, exist_ok=True)

        from functools import reduce
        _stats_txt = '\n'
        _percent_stats_txt = '\n'

        combined_stats = {k: [] for k in ['correct', 'total']}
        stats_summary_types = ['policy', 'correct', 'incorrect', 'total']
        fp_background_stats = {k: [] for k in stats_summary_types}
        tp_stats = {k: [] for k in stats_summary_types}

        for _state in ('active', 'lost', 'tracked'):
            stats_list = [k[_state] for k in self._stats_dict.values()]
            stats = reduce(lambda x, y: x.add(y, fill_value=0), stats_list)
            # _stats_txt += f"{_state}\n"

            # stats_total = stats.loc['total']
            # percent_stats = stats.divide(stats_total, axis='columns')*100

            """all rows as percent of the first one """
            percent_stats = pd.DataFrame(
                np.full((len(PolicyBase.Decision.types[1:]), len(AnnotationStatus.types)), np.nan, dtype=np.float32),
                columns=AnnotationStatus.types, index=PolicyBase.Decision.types[1:])
            """all columns as percent of the first one """
            status_percent_stats = pd.DataFrame(
                np.full((len(PolicyBase.Decision.types), len(AnnotationStatus.types[1:])), np.nan, dtype=np.float32),
                columns=AnnotationStatus.types[1:], index=PolicyBase.Decision.types)

            valid_cols = np.flatnonzero(stats.values[0, :] > 0)
            percent_stats.values[:, valid_cols] = (stats.values[1:, valid_cols] /
                                                   stats.values[0, valid_cols].reshape((1, -1))) * 100.0

            valid_rows = np.flatnonzero(stats.values[:, 0] > 0)
            status_percent_stats.values[valid_rows, :] = (stats.values[valid_rows, 1:] /
                                                          stats.values[valid_rows, 0].reshape(
                                                              (-1, 1))) * 100.0

            """how many total decisions"""
            combined_stats['total'].append(stats['combined']['total'])

            """what percent of decisions for all, fp_background and tp targets are correct"""
            combined_stats['correct'].append(percent_stats['combined']['correct'])
            fp_background_stats['policy'].append(percent_stats['fp_background']['correct'])
            tp_stats['policy'].append(percent_stats['tp']['correct'])

            """what percent of all, correct and incorrect decisions are fp_background and tp"""
            for k in stats_summary_types[1:]:
                fp_background_stats[k].append(status_percent_stats['fp_background'][k])
                tp_stats[k].append(status_percent_stats['tp'][k])

            percent_stats.dropna(axis='columns', how='all', inplace=True)
            status_percent_stats.dropna(how='all', inplace=True)

            write_df(stats, linux_path(stats_dir, f'{_state}.log'), _state, load_dir)
            write_df(percent_stats, linux_path(stats_dir, f'{_state}_percent.log'), _state, load_dir)
            write_df(status_percent_stats, linux_path(stats_dir, f'{_state}_status_percent.log'), _state,
                     load_dir)

            write_df(stats, linux_path(accumulative_stats_dir, f'{_state}.log'), _state, load_dir)
            write_df(percent_stats, linux_path(accumulative_stats_dir, f'{_state}_percent.log'), _state,
                     load_dir)
            write_df(status_percent_stats,
                     linux_path(accumulative_stats_dir, f'{_state}_status_percent.log'), _state, load_dir)

            stats.index.name = _state
            if self._params.print_stats:
                print()
                print_df(stats, _state)
                print_df(percent_stats)
                print_df(status_percent_stats)

        self._stats_summary_to_file(combined_stats, fp_background_stats, tp_stats, load_dir,
                                    (stats_dir, accumulative_stats_dir), time_stamp=time_stamp)

        self._logger.info(f'Saved accumulative decision stats to folders:\n{stats_dir}\n{accumulative_stats_dir}')

    def _stats_summary_to_file(self, combined_stats, fp_background_stats, tp_stats, load_dir, out_dirs, time_stamp):
        """

        :param combined_stats:
        :param fp_background_stats:
        :param tp_stats:
        :param load_dir:
        :param list | tuple out_dirs:
        :return:
        """

        policy_summary_header = 'time_stamp\tfile\ttotal\t\t\t' \
                                'correct_out_of_total\t\t\t' \
                                'correct_out_of_fp_bkg\t\t\t' \
                                'correct_out_of_tp\t\t\t'
        policy_summary_str = list_to_str(combined_stats['total'] + combined_stats['correct'] +
                                         fp_background_stats['policy'] + tp_stats['policy'])
        status_summary_header = 'time_stamp\tfile\t' \
                                'fp_bkg_out_of_total\t\t\ttp_out_of_total\t\t\t' \
                                'fp_bkg_out_of_correct\t\t\ttp_out_of_correct\t\t\t' \
                                'fp_bkg_out_of_incorrect\t\t\ttp_out_of_incorrect\t\t\t'
        status_summary_str = list_to_str(fp_background_stats['total'] + tp_stats['total'] +
                                         fp_background_stats['correct'] + tp_stats['correct'] +
                                         fp_background_stats['incorrect'] + tp_stats['incorrect'])
        for _dir in out_dirs:

            policy_stats_path = linux_path(_dir, f'summary_policy.log')

            print(f'Writing policy_stats summary to: {policy_stats_path}')

            if not os.path.isfile(policy_stats_path):
                with open(policy_stats_path, 'w') as fid:
                    fid.write(policy_summary_header + '\n')
            with open(policy_stats_path, 'a') as fid:
                fid.write(time_stamp + '\t' + load_dir + '\t' + policy_summary_str + '\n')

            status_stats_path = linux_path(_dir, f'summary_status.log')
            print(f'Writing status_stats summary to: {status_stats_path}')

            if not os.path.isfile(status_stats_path):
                with open(status_stats_path, 'w') as fid:
                    fid.write(status_summary_header + '\n')
            with open(status_stats_path, 'a') as fid:
                fid.write(time_stamp + '\t' + load_dir + '\t' + status_summary_str + '\n')

    def load(self, load_path):
        """
        :type load_path: str
        :rtype: bool
        """

        if self._params.load_from_alternative and not os.path.exists(load_path):
            load_dir = os.path.dirname(load_path)
            load_parent_dir = os.path.dirname(load_dir)
            load_parent_root_dir = os.path.dirname(load_parent_dir)
            load_fname = os.path.basename(load_path)

            src_file_gen = [[(f, os.path.join(dirpath, f)) for f in filenames]
                            for (dirpath, dirnames, filenames) in os.walk(load_parent_root_dir, followlinks=True)]
            fname_to_path = dict([item for sublist in src_file_gen for item in sublist])

            try:
                src_path = fname_to_path[load_fname]
            except KeyError:
                raise IOError('No alternatives found for nonexistent results file:\n{}\nin\n{}'.format(
                    load_path, load_parent_root_dir
                ))
            else:
                print('\nfound alternative:\n{}\nfor nonexistent results file:\n{}'.format(src_path, load_path))
                if not os.path.exists(load_dir):
                    os.makedirs(load_dir, exist_ok=True)
                shutil.copy(src_path, load_path)

        if not self.input.read_tracking_results(load_path):
            raise IOError('Tracking results could not be loaded')

        self._logger.info('Tracking results loaded successfully')
        return True

    def save(self, save_path='', log_file=''):
        """
        :type save_path: str
        :type log_file: str
        :rtype: bool
        """

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        save_fname = os.path.basename(save_path)
        save_fname_no_ext, save_ext = os.path.splitext(save_fname)

        samples_save_dir = os.path.join(save_dir, save_fname_no_ext)

        check_states = 1

        if self._params.res_from_gt == 2:
            self._results = interpolate_missing_objects(self._results)
            check_states = 0

        if self._trained_target is not None:
            self._trained_target.save_test_samples(samples_save_dir)

        if log_file:
            if not os.path.isfile(log_file):
                self._logger.warning('nonexistent log file: {}'.format(log_file))
            else:
                dst_log_file = save_path.replace('.txt', '.log')
                self._logger.info('Moving log to {:s}'.format(dst_log_file))

                shutil.copy(log_file, dst_log_file)
                try:
                    os.remove(log_file)
                except PermissionError:
                    pass

        save_fmt = '%d,%d,%f,%f,%f,%f,%f,%d,%d,%d'
        if self._results_raw is not None:
            save_path_raw = save_path.replace('.txt', '.raw')
            save_fmt_raw = save_fmt + ',%d'
            self._logger.info('Saving raw tracking results to {:s}'.format(save_path_raw))
            np.savetxt(save_path_raw, self._results_raw, fmt=save_fmt_raw, delimiter=',', newline='\n')

        self._logger.info('Saving tracking results to {:s}'.format(save_path))

        if self._results is not None:
            results = self._results
        else:
            self._logger.warning('No tracking results so saving a single dummy box')
            """dummy results to allow evaluation to work"""
            results = np.zeros((1, 12))
            results[:, 0:2] = 1

            # results = np.zeros((self.input.n_frames, 12))
            # results[:, 0] = np.arange(1, self.input.n_frames)
            # results[:, 1] = np.arange(1, self.input.n_frames)

            """1 pixel box at (0, 0)"""
            # results[:, 4:6] = 1
            # results[:, 10] = MDPStates.tracked
            # results[:, 11] = 1

        """post-processing to clamp and remove invalid boxes"""
        xmin, ymin, w, h = results[:, 2], results[:, 3], results[:, 4], results[:, 5]
        xmax, ymax = xmin + w, ymin + h

        frame_size = self.input.frame_size

        xmin = np.clip(xmin, 0, frame_size[0])
        xmax = np.clip(xmax, 0, frame_size[0])
        ymin = np.clip(ymin, 0, frame_size[1])
        ymax = np.clip(ymax, 0, frame_size[1])

        valid_boxes = np.flatnonzero(np.logical_and(xmax > xmin, ymax > ymin))

        results[:, 2] = xmin
        results[:, 3] = ymin

        results[:, 4] = xmax - xmin
        results[:, 5] = ymax - ymin

        results = results[valid_boxes, ...]

        n_results_cols = results.shape[1]

        if n_results_cols > 10:
            """states"""
            save_fmt += ',%d'
            self._logger.info('state info is enabled')
            if n_results_cols > 11:
                """validity"""
                self._logger.info('validity info is enabled')
                save_fmt += ',%d'
        if self.input.start_frame_id > 0:
            results[:, 0] += self.input.start_frame_id

        np.savetxt(save_path, results, fmt=save_fmt, delimiter=',', newline='\n')

        if self._params.save_mot:
            if not self.input.read_tracking_results(save_path):
                raise IOError('Tracking results could not be loaded')
            self.input.track_res.get_mot_compatible_file(check_states=check_states)

        return True

    def close(self):
        # process remaining targets
        self._process_inactive_targets(np.arange(self.n_live, dtype=np.int32))
        if self.vis:
            self.visualizer.close()
