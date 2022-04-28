import numpy as np
import shutil
import os
import zipfile
import copy
import cProfile, pstats, io
from pprint import pformat

import paramparse

from target import Target
from input import Input, Detections, Annotations
from data import Data
from visualizer import Visualizer, ObjTypes
from utilities import linux_path, write, CustomLogger, MDPStates, DebugParams, BaseParams


class Trainer:
    """
    :type _params: Trainer.Params
    :type _logger: logging.RootLogger | logging.Logger
    :type target: Target
    :type enable_vis: bool
    """

    class Params(BaseParams):
        """
        :type input: Input.Params
        :type help: {str:str}


        :ivar mode: '0: run tracker as usual and let the target decide how to train, '
                '1: train only lost state policy,'
                '2: train only tracked state policy',
        :ivar active_load_path: 'optional path to load a pre-trained active model from and skip training it',
        :ivar overlap_occ: 'IOA threshold for deciding if an annotation is to be considered as occluded by another',
        :ivar ignore_too_difficult: 'dominate training without raising an error if all the trajectories are found to
        be too difficult to train on; disabling it would cause an assertion error to be raised indicating complete and
        utter failure of training and the corresponding garbage trained model due to the method that it is'
        :ivar max_iter: 'Maximum number of iterations allowed on any given sequence before training on that sequence is'
                    ' considered to be complete - an iteration is considered to have happened when one instance of'
                    'training over a particular trajectory has been completed',
        :ivar max_count: 'Maximum number of times training can occur for any given trajectory or the maximum number of '
                     'instances of training that can take place for any given trajectory - when the number of '
                     'instances exceeds this number, that particular trajectory is considered to be too difficult'
                     ' to train on',
        :ivar max_pass: 'Maximum number of training passes over all the trajectories in a given sequence - one pass is '
                    'considered to have been completed when one training instance has been run over '
                    'all the trajectories in that sequence ',
        :ivar exit_threshold: 'Minimum fraction of the area of an annotation that must be inside the of the image'
                  'boundaries if that annotation can be considered as the starting point of a '
                  'training trajectory - Same as its namesake in the tracked state policy module parameters',
        :ivar max_inactive_targets: 'maximum no. of inactive targets allowed to accumulate in the record'
                        ' before being removed',
        :ivar override: '1: override parameters in a loaded trained target with those specified at runtime; '
                    '2: do not load parameters from saved file at all'
                    'the learned model in the target might depend on several of these saved parameters '
                    'and might not function well or otherwise behave in an unpredictable manner'
                    ' if these are changed;',

        :ivar input: 'Input parameters',
        :ivar target: 'Target parameters',
        :ivar visualizer: 'Visualizer parameters',
        :ivar debug: 'Debugging parameters',
        :ivar profile: 'Enable code profiling',
        :ivar verbose: 'Enable printing of general diagnostic messages',

        """

        def __init__(self):
            self.mode = 0

            """training parameters"""
            self.max_iter = 10000  # max iterations in total
            self.max_count = 10  # max iterations per trajectory
            self.max_pass = 2
            self.ignore_too_difficult = 1

            """heuristics"""
            self.exit_threshold = 0.95

            """debugging"""
            self.vis = 0
            self.profile = 0
            self.verbose = 0
            self.verbose_depth = 1

            self.save_zip = 0
            # self.load_latest = 0

            self.override = 0
            self.yolo3d = 2

            self.summary_dir = 'log/summary'

            self.input = Input.Params()
            self.target = Target.Params()
            self.visualizer = Visualizer.Params()
            self.debug = DebugParams()

    class Modes:
        standard = 0
        lost_batch, lost_async = -1, 1
        tracked_batch, tracked_async = -2, 2
        active_batch, active_async = -3, 3

        batch = [-1, -2, -3]
        asynchronous = [1, 2, 3]

        to_str = {
            0: 'std',
            1: 'lost_async',
            2: 'tracked_async',
            3: 'active_async',
            -1: 'lost_batch',
            -2: 'tracked_batch',
            -3: 'active_batch',
        }
        # inactive, active, tracked, lost = ('inactive', 'active', 'tracked', 'lost')

    def __init__(self, params, logger, args_in=()):
        """
        :type params: Trainer.Params
        :type logger: logging.RootLogger | logging.Logger
        :rtype: None
        """
        self._logger = logger
        self._orig_logger = logger
        self._params = params

        """optional runtime parameter overriding"""
        self._args_in = [k.replace('trainer.', '') for k in args_in
                         if k.startswith('--trainer.')]

        self.target = None

        # only batch mode input is currently supported
        self._params.input.batch_mode = 1

        self.enable_vis = self._params.vis and \
                          np.array(self._params.visualizer.mode).any() and \
                          (self._params.visualizer.save or self._params.visualizer.show)
        self.visualizer = Visualizer(self._params.visualizer, self._logger)

        # temporary hack for debugging
        self._params.target.set_verbosity(self._params.verbose, self._params.verbose_depth)

        self._n_frames = -1
        self._frames = None
        self._frame_size = None
        self._annotations = None
        self._detections = None
        # self._seq_name = None

        if self._params.profile:
            self._logger.info('Profiling is enabled')
        self.profiler = cProfile.Profile()

        self.rgb_input = False
        self.is_initialized = False
        # self.tr = tracker.SummaryTracker()
        self.state_to_train = None

    def _set_rgb(self):
        # if not self._params.input.convert_to_gs and \
        #         self._params.target.templates.tracker in Trackers['LK']:
        #     self._logger.warning('LK only supports grayscale images')
        #     self._params.input.convert_to_gs = 1

        self.rgb_input = not self._params.input.convert_to_gs

        if self.rgb_input:
            self._logger.info('Using RGB input images')
        else:
            self._logger.info('Using grayscale input images')

    def run(self, data, logger=None):
        """
        :type data: Data
        :type logger: CustomLogger | logging.RootLogger
        :rtype: bool
        """

        if logger is not None:
            self._logger = logger

        self._set_rgb()

        if not self.initialize(data):
            raise AssertionError('Trainer initialization failed')

        # if self._params.yolo3d == 1:
        #     from utilities import build_targets_3d
        #     build_targets_3d(self._frames, self._annotations)
        #     exit()
        # elif self._params.yolo3d == 2:
        #     from utilities import build_targets_seq
        #     build_targets_seq(self._frames, self._annotations)
        #     exit()

        if not self.update():
            raise AssertionError('Trainer update failed')

        if self.enable_vis:
            self.visualizer.close()

        return True

    def initialize(self, data=None, frames=None, annotations=None, detections=None):
        """
        :type data: Data | None
        :type frames: list[np.ndarray]
        :type annotations: Annotations
        :type detections: Detections
        :rtype: bool
        """

        if self._params.mode in Trainer.Modes.batch:
            self._logger.info('Skipping initialization as unnecessary for batch training')
            return True

        # if self._params.mode == Trainer.Modes.active_async:
        #     """images not needed for active policy data generation though corresponding statistics are
        #     so pipeline initialization must be done"""
        #     self._params.input.batch_mode = 0

        if frames is None:
            assert data is not None, 'sequence data must be provided when external frames are not'

            _input = Input(self._params.input, self._logger)

            if not _input.initialize(data):
                raise IOError('Input pipeline could not be initialized')

            # read detections and annotations
            if not _input.read_annotations():
                raise IOError('Annotations could not be read')

            if not _input.read_detections():
                raise IOError('Detections could not be read')

            self._frame_size = _input.frame_size
            self._n_frames = _input.n_frames
            self._frames = _input.all_frames
            self._annotations = _input.annotations
            self._detections = _input.detections
            # self._seq_name = _input.seq_name
        else:
            """external image source is to be used"""
            assert annotations is not None and detections is not None, \
                "annotations and detections must be provided for external image source"
            self._annotations = annotations
            self._detections = detections
            self._frames = frames
            self._frame_size = (frames[0].shape[1], frames[0].shape[0])

        # initialize visualizer
        if self.enable_vis and not self.visualizer.initialize("trainer", self._frame_size):
            raise AssertionError('Visualizer could not be initialized')

        self.is_initialized = True
        return True

    def update(self):
        """
        :rtype: bool
        """

        batch_train = None

        # train_active_state = 0
        if self._params.mode == Trainer.Modes.standard:
            self.state_to_train = None
            # train_active_state = 1
        elif self._params.mode == Trainer.Modes.lost_batch:
            self._logger.info('Training lost policy in batch')
            # self.target.lost.batch_train()
            self.state_to_train = MDPStates.lost
            batch_train = MDPStates.lost
            # return True
        elif self._params.mode == Trainer.Modes.lost_async:
            self._logger.info('Training lost policy asynchronously')
            self.state_to_train = MDPStates.lost
        elif self._params.mode == Trainer.Modes.tracked_batch:
            self._logger.info('Training tracked policy in batch')
            # self.target.tracked.batch_train()
            self.state_to_train = MDPStates.tracked
            batch_train = MDPStates.tracked
            # return True
        elif self._params.mode == Trainer.Modes.tracked_async:
            self._logger.info('Training tracked policy asynchronously')
            self.state_to_train = MDPStates.tracked
        elif self._params.mode == Trainer.Modes.active_async:
            self._logger.info('Training active policy asynchronously')
            self.state_to_train = MDPStates.active
            # train_active_state = 1
        elif self._params.mode == Trainer.Modes.active_batch:
            self._logger.info('Training active policy in batch')
            # self.target.active.batch_train()
            self.state_to_train = MDPStates.active
            batch_train = MDPStates.active
            # return True
        else:
            raise SystemError('Invalid train mode: {}'.format(self._params.mode))

        if self.target is None:
            self._logger.debug('starting training...')
            """Using original (non-sequences specific) logger for object creation as a temporary hack to prevent
            incorrect sequence from being part of the logging message until recursive implementation of the set_logger 
            functionality is  completed"""
            self.target = Target(self._params.target, self.rgb_input, self._orig_logger,
                                 batch_train=batch_train)
        else:
            self._logger.debug('continuing training...')

        self.target.set_logger(self._logger)

        if batch_train is not None:
            self.target.batch_train(self.state_to_train)
            return True

        self.target.train_mode()

        n_frames, frame_size = self._n_frames, self._frame_size
        annotations, detections, frames = self._annotations, self._detections, self._frames

        assert annotations is not None, "annotations is None"
        assert detections is not None, "detections is None"

        # if frames is None:
        #     n_frames, frame_size = self._n_frames, self._frame_size
        # else:
        #     n_frames = len(frames)
        #     frame_size = (frames[0].shape[1], frames[0].shape[0])

        # if self._params.debug.write_state_info:
        #     """remove existing log files from previous runs to avoid conflicts"""
        #     if self._params.debug.cmp_root_dirs[0] and os.path.isdir(self._params.debug.cmp_root_dirs[0]):
        #         self._logger.info('Removing debug.cmp_root_dir {}'.format(self._params.debug.cmp_root_dirs[0]))
        #         shutil.rmtree(self._params.debug.cmp_root_dirs[0])
        #     if self._params.debug.cmp_root_dirs[1] and os.path.isdir(self._params.debug.cmp_root_dirs[1]):
        #         self._logger.info('Removing debug.cmp_root_dir {}'.format(self._params.debug.cmp_root_dirs[1]))
        #         shutil.rmtree(self._params.debug.cmp_root_dirs[1])
        #     # removeSubFolders(self.params.debug, 'target_')

        train_traj_ids = self._get_train_trajectories(
            annotations, detections, n_frames, frame_size)  # type: list
        if train_traj_ids is None:
            raise AssertionError('Failed to get training trajectories')

        # __target, __templates, __lost, __tracked, __history, __tracker, __active = \
        #     self.target, self.target.templates, self.target.lost, self.target.tracked, \
        #     self.target.history, self.target.templates._tracker, self.target.active

        # self.target.add_sequence(frame_size)

        """
        Train active state with the new sequence
        """
        self.target.active.train(self._frames, detections)

        if self._params.mode == 3:
            """
            async training of active policy
            """
            return True

        # if self._params.mode > 0:
        #     self._train_state_policy(frames, n_frames, annotations, detections, seq_name)
        #     return

        assert frames is not None, "frames is None"

        train_id = -1
        iter_id = 0
        reward = 0
        pass_count = 0
        n_train = len(train_traj_ids)
        counter = np.zeros((n_train, 1), dtype=np.int32)
        is_good = np.zeros((n_train, 1), dtype=np.bool)
        is_difficult = np.zeros((n_train, 1), dtype=np.bool)
        pause_for_debug = 0

        if self._params.profile:
            self.profiler.enable()

        if self._params.mode in Trainer.Modes.asynchronous:
            """Associate using annotations during policy specific asynchronous training"""
            assoc_with_ann = True
        else:
            """Associate using policy for synchronous training"""
            assoc_with_ann = False

        while True:
            """iterate over all trajectories multiple times
            """
            iter_id += 1
            if self._params.verbose:
                self._logger.debug('iter {:d}'.format(iter_id))
            else:
                write('.')
                if iter_id % 100 == 0:
                    write('\n')

            if iter_id > self._params.max_iter:
                self._logger.debug('max iterations reached')
                break

            if is_good.all():
                """one pass over all trajectories completed"""
                pass_count += 1
                if pass_count == self._params.max_pass:
                    self._logger.debug('max pass reached')
                    break
                else:
                    self._logger.debug('pass {:d} finished'.format(pass_count))
                    if is_difficult.all():
                        msg = 'all trajectories are too difficult to train on so terminating training'
                        if self._params.ignore_too_difficult:
                            self._logger.warning(msg)
                            break
                        raise AssertionError(msg)

                    """train only on non-difficult trajectories"""
                    is_good.fill(False)
                    is_good[is_difficult] = True
                    counter.fill(0)
                    train_id = -1

            """find the next trajectory to train on"""
            while True:
                """check the next trajectory, circularly if needed, and use it for training if it
                has not been marked as good thus far
                """
                train_id += 1
                if train_id >= n_train:
                    train_id = 0
                if not is_good[train_id]:
                    break

            traj_id, local_start_id = train_traj_ids[train_id]
            traj_idx = annotations.traj_idx[traj_id][local_start_id:]

            traj_start_id = traj_idx[0]
            traj_idx_by_frame = annotations.traj_idx_by_frame[traj_id]
            curr_ann_data = annotations.data[traj_idx, :]

            start_frame_id = int(np.amin(curr_ann_data[:, 0]))
            end_frame_id = int(np.amax(curr_ann_data[:, 0]))

            frame_id = int(annotations.data[traj_start_id, 0])
            obj_id = int(annotations.data[traj_start_id, 1])

            if self._params.verbose:
                self._logger.debug(
                    f'trajectory {traj_id} ({train_id + 1}/{n_train}) with {traj_idx.size} frames {start_frame_id} '
                    f'--> {end_frame_id}')

            self.target.reset_state(obj_id)

            """debug"""
            # curr_traj = annotations.subsetInTrajectory(train_id)

            """choose the location of the best matching detection with the initial annotation to initialize the target; 
            this detection is assumed to be correct so no decision needs to be made by the target
            """

            """moved to the main loop temporarily for debugging purposes"""

            # best_det_id = annotations.overlaps_idx[traj_start_id]
            # best_det_data = detections.data[best_det_id, :]
            # if self.params.verbose:
            #     write('\n iter {:d}, frame {:d}, state {:d}\n'.format(
            #         iter_id, frame_id + 1, self.target.state))
            #     print('Start: first frame overlap {:.2f}\n'.format(
            #         annotations.overlaps[traj_start_id]))
            # # get current frame
            # if self.params.input.batch_mode:
            #     curr_frame = all_frames[frame_id]
            # else:
            #     curr_frame = self.getFrame(frame_id)
            #
            # self.target.initialize(obj_id, frame_id, curr_frame, best_det_data, decide=False)
            # if self.params.debug.write_state_info:
            #     root_dir = '{:s}/target_{:d}'.format(self.params.debug.cmp_root_dirs[1],
            #                                          self.target.id)
            #     self.target.active.writeStateInfo(root_dir, self.params.debug.write_to_bin)

            # debug
            # self.target.templates.show()

            # frame_id += 1

            while frame_id < n_frames:

                try:
                    curr_ann_idx = traj_idx_by_frame[frame_id]
                except KeyError:
                    curr_ann_idx = None

                # if self._params.debug.write_state_info and \
                #         (frame_id + 1) >= self._params.debug.write_thresh[1] and \
                #         iter_id >= self._params.debug.write_thresh[0]:
                #     pause_for_debug = True
                #     self._params.target.pause_for_debug = pause_for_debug
                #     self._params.target.tracked.pause_for_debug = pause_for_debug
                #     self._params.target.lost.pause_for_debug = pause_for_debug
                #     self._params.target.templates.pause_for_debug = pause_for_debug
                #     self._params.target.history.pause_for_debug = pause_for_debug

                """all detections in the current frame"""
                det_ids = detections.idx[frame_id]
                if det_ids is not None:
                    curr_det_data = detections.data[det_ids, :]
                else:
                    curr_det_data = np.array([])

                if self._params.verbose:
                    write(
                        f'pass {pass_count}, '
                        f'iter {iter_id}, '
                        f'traj {traj_id} ({train_id + 1} / {n_train}), '
                        f'frame {frame_id} / {end_frame_id}, '
                        f'id: {self.target.id_}, '
                        f'state {self.target.state_str()}, '
                        f'dets {curr_det_data.shape[0]}\n')

                curr_frame = frames[frame_id]

                # """single annotation from the current trajectory in the current frame
                # """
                # curr_frame_ann_idx = np.flatnonzero(curr_ann_data[:, 0] == frame_id)
                # assert curr_frame_ann_idx.size <= 1, ' multiple annotations for object {} found  in frame {}'.format(
                #     traj_idx, frame_id
                # )

                if self.target.state == MDPStates.inactive:
                    if reward == 1:
                        is_good[train_id] = True
                        self._logger.debug(f'trajectory {traj_id} ({train_id + 1}/{n_train}) is good '
                                           f'(frame {frame_id} ({start_frame_id} --> {end_frame_id})')
                    break
                elif self.target.state == MDPStates.active:
                    """
                    create a new target on the max iou detection
                    trajectory filtering ensures that the first frame always has a matching detection  
                    """
                    best_det_id = annotations.max_cross_iou_idx[traj_start_id]
                    best_det_data = detections.data[best_det_id, :]
                    if self._params.verbose:
                        self._logger.debug('Start: first frame overlap {:.2f}\n'.format(
                            annotations.max_cross_iou[traj_start_id]))

                    self.target.initialize(obj_id, frame_id, curr_frame, best_det_data,
                                           annotations, traj_id, ann_status="tp")
                else:
                    """lost and tracked states"""
                    learn = False

                    if self._params.mode == Trainer.Modes.standard and self.target.state == MDPStates.lost:
                        learn = True
                        """have to take this circuitous route to deal with the case of empty "curr_ann_idx",
                        i.e. when there are no annotations in the current frame but we still want
                        to learn; an empty matrix will never give True on == nor will it give True on !=
                        which in this case will give the desired result since "learn" will remain true
                        """
                        if curr_ann_idx is not None:
                            """indexing with empty matrix returns another empty matrix"""
                            ann_ioa = annotations.max_ioa[curr_ann_idx[0]]
                            if ann_ioa > self._params.target.lost.ann_ioa_thresh:
                                """
                                annoying heuristic
                                ------------------
                                don't train if the current GT box is even slightly occluded by / occludes
                                another GT box in the same frame
                                """
                                learn = False
                                """
                                also don't use any of the current detections for the even more
                                annoying heuristical reason that lost state update does not occur without
                                detections so this turns off both learning nd updating
                                """
                                curr_det_data = np.array([])

                    # print('curr_frame.shape: ', curr_frame.shape)
                    # self.logger.profile('Updating target...')
                    # start_t = time.time()

                    self.target.update(curr_frame, frame_id, curr_det_data,
                                       associate=True, assoc_with_ann=assoc_with_ann)

                    # if self.params.debug.memory_tracking:
                    #     print('update:')
                    #     self.tr.print_diff()

                    # end_t = time.time()
                    # self.logger.profile('Updating target: Time taken: {:f}'.format(end_t - start_t))

                    if learn or self._params.mode != Trainer.Modes.standard:
                        # curr_detections = detections.subset(frame_id)
                        # curr_annotations = annotations.subsetInTrajectory(train_id)

                        # if curr_iou is not None and curr_iou.size > 0:

                        """
                        allow learning without annotations
                        """
                        # self.logger.profile('Learning target...')
                        is_end, reward = self.target.train(self.state_to_train)

                        # if self.params.debug.memory_tracking:
                        #     print('learn:')
                        #     self.tr.print_diff()

                        if is_end:
                            self._logger.debug(f'target {traj_id} ({train_id + 1}/{n_train}) exits due to '
                                               f'policy train decision')
                            self.target.set_state(MDPStates.inactive)

                    """
                    temporary allowance for some annoying ad-hoc heuristics in the original code
                    """
                    if self.target.lost.streak > self._params.target.lost.max_streak:
                        self.target.state = MDPStates.inactive
                        # if self._params.verbose:
                        self._logger.debug('target exits due to long time occlusion')

                    if self.target.state == MDPStates.inactive:
                        if self.target.prev_state == MDPStates.tracked:
                            """target has left the scene after being tracked to its edge
                            """
                            self._logger.debug('target has left the scene after being tracked to its edge')
                            reward = 1
                        elif self.target.prev_state == MDPStates.lost and \
                                annotations.idx[frame_id] is None:
                            """target has been lost for a long time and there are no annotations in this frame
                            """
                            self._logger.debug(
                                'target has been lost for a long time and there are no annotations in this frame')
                            reward = 1

                """
                temporary allowance for some annoying ad-hoc heuristics in the original code
                """
                if self.target.state == MDPStates.tracked and \
                        self.target.is_out_of_frame(self.target.location, curr_frame):
                    self.target.state = MDPStates.inactive
                    # if self._params.verbose:
                    self._logger.debug('target outside image by checking border')
                    reward = 1

                # if pause_for_debug:
                #     root_dirs = tuple(['{:s}/target_{:d}'.format(root_dir, self.target.id_) if root_dir else ''
                #                        for root_dir in self._params.debug.cmp_root_dirs])
                #     files = self.target.write_state_info(root_dirs[1], self._params.debug.write_to_bin)
                #     msg = 'Target {:d}, state: {:d}'.format(self.target.id_, self.target.state_str())
                #     if not compare_files(self._params.debug.write_to_bin, files, root_dirs,
                #                                sync_id=frame_id + 1, msg=msg):
                #         self._logger.error('All files are not identical')
                #         # self.logger.debug('paused')

                if self.enable_vis:
                    self._visualize(frame_id, curr_frame, annotations, detections)

                if self._params.mode == 0 and self.target.state == MDPStates.lost and \
                        self.target.prev_state == MDPStates.tracked:
                    """
                    try to reconnect if the target has just been lost
                    """
                    continue

                frame_id += 1

                if self._params.mode > 0 and frame_id > end_frame_id:
                    is_good[train_id] = True
                    self._logger.debug(f'done training on trajectory {traj_id} ({train_id + 1}/{n_train}) '
                                       f'(frame {frame_id} ({start_frame_id} --> {end_frame_id})')
                    break

            if frame_id >= n_frames:
                """all frames in the sequence have been processed"""
                is_good[train_id] = True
                self._logger.debug(f'trajectory {traj_id} ({train_id + 1}/{n_train}) is good')

            counter[train_id] += 1
            if counter[train_id] > self._params.max_count:
                is_good[train_id] = True
                is_difficult[train_id] = True
                self._logger.debug(f'trajectory {traj_id} ({train_id + 1}/{n_train}) max iteration')

        self._logger.info('Finished training')

        if self._params.profile:
            self.profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())

        return True

    def _get_train_trajectories(self, annotations, detections, n_frames, frame_size):
        """

        :param Annotations annotations:
        :param Detections detections:
        :param int n_frames:
        :param frame_size:
        :rtype: list | None
        """

        if not annotations.get_features(detections, n_frames, frame_size):
            raise AssertionError('Failed to get annotation features')

        overlap_pos = self._params.target.active.overlap_pos,
        exit_threshold = self._params.exit_threshold,
        # debug = self._params.debug

        detections.max_cross_iou = annotations.cross_overlaps.max_iou_1
        """index of the maximum overlapping annotation for each detection"""
        detections.max_cross_iou_idx = annotations.cross_overlaps.max_iou_1_idx

        # traj_id = 0
        train_traj_ids = []
        # valid_idx_by_frame = {}
        # obj_to_traj = {}
        # traj_to_obj = {}
        for traj_id in range(annotations.n_traj):
            """           
            indices where a valid training trajectory can start - we consider only frames where:
            1. annotations have corresponding detections
            2. are not covered by other annotations and 
            3. lie mostly inside the frame so that the object is not about to exit the scene and 
            is "properly" inside the scene
            """
            idx = annotations.traj_idx[traj_id]
            # idx_by_frame = annotations.traj_idx_by_frame[traj_id]

            # curr_traj_data = self.data[idx, :]

            # print('idx:\n', idx)
            # print('overlaps.shape: ', self.overlaps.shape)
            # print('covered.shape: ', self.covered.shape)
            # print('area_inside_frame.shape: ', self.area_inside_frame.shape)

            # curr_overlaps = self.overlaps[idx]
            # curr_covered = self.covered[idx]
            # curr_area_inside_frame = self.area_inside_frame[idx]

            valid_start_idx = np.flatnonzero(np.logical_and.reduce((
                annotations.max_cross_iou[idx] > overlap_pos,
                annotations.max_ioa[idx] == 0,
                annotations.area_inside_frame[idx] > exit_threshold)))

            if valid_start_idx.size == 0:
                """none of the frames in this trajectory meet the conditions so it must be removed"""
                # del annotations.traj_idx[traj_id]
                # obj_id = annotations.traj_to_obj[traj_id]
                # del annotations.obj_to_traj[obj_id]
                # del annotations.traj_to_obj[traj_id]
                #
                # annotations.n_traj -= 1
                pass
            else:
                train_traj_ids.append((traj_id, valid_start_idx[0]))

        n_train_traj = len(train_traj_ids)

        if n_train_traj == 0:
            raise AssertionError('No valid training trajectories found')
        elif n_train_traj == 1:
            self._logger.info('Found only one training trajectory')
        else:
            self._logger.info('Found {:d} training trajectories'.format(n_train_traj))

        # if debug.write_state_info and np.less_equal(debug.write_thresh, 1).all():
        #     fp_dtype = np.float32
        #     fp_fmt = '%.10f'
        #     root_dir = debug.cmp_root_dirs[1]
        #     files = []
        #     entries = [
        #         # (detections.labels, 'labels', np.int32, '%d'),
        #         (detections.max_cross_iou_idx, 'indices', np.int32, '%d'),
        #         (detections.max_cross_iou, 'overlaps', fp_dtype, fp_fmt)
        #     ]
        #     write_to_files('{:s}/detections'.format(root_dir),
        #                          debug.write_to_bin, entries)
        #     # files.extend(['detections/{:s}'.format(entry[1]) for entry in entries])
        #     files.extend([('detections/{:s}'.format(entry[1]), entry[2], entry[0].shape) for entry in entries])
        #
        #     entries = [
        #         (annotations.max_cross_iou, 'overlaps', fp_dtype, fp_fmt),
        #         (annotations.max_ioa, 'covered', fp_dtype, fp_fmt),
        #         (annotations.occluded, 'occluded', fp_dtype, fp_fmt),
        #         (annotations.area_inside_frame, 'area_inside', fp_dtype, fp_fmt),
        #         (annotations.scores, 'scores', fp_dtype, fp_fmt)
        #     ]
        #     write_to_files('{:s}/annotations'.format(root_dir),
        #                          debug.write_to_bin, entries)
        #     # files.extend(['annotations/{:s}'.format(entry[1]) for entry in entries])
        #     files.extend([('annotations/{:s}'.format(entry[1]), entry[2], entry[0].shape) for entry in entries])
        #
        #     if not compare_files(debug.write_to_bin, files,
        #                                debug.cmp_root_dirs, sync_id=0,
        #                                msg='Trajectory features'):
        #         self._logger.error('All files are not identical')

        return train_traj_ids

    def _visualize(self, frame_id, frame, annotations, detections):
        """
        :type frame_id: int
        :type frame: np.ndarray
        :type annotations: Annotations
        :type detections: Detections
        :rtype: None
        """
        frame_data = {}
        if self._params.visualizer.mode[0]:
            frame_data[ObjTypes.tracking_result] = self.target.get_data(True)
        if self._params.visualizer.mode[1]:
            det_ids = detections.idx[frame_id]
            if det_ids is not None:
                frame_data[ObjTypes.detection] = detections.data[det_ids, :]
        if self._params.visualizer.mode[2]:
            ann_ids = annotations.idx[frame_id]
            if ann_ids is not None:
                ann_data = annotations.data[ann_ids, :]
                # annotation in the current frame that is also in the current trajectory
                ann_idx = np.flatnonzero(np.equal(ann_data[:, 1], self.target.id_))
                frame_data[ObjTypes.annotation] = ann_data[ann_idx, :]
        self.visualizer.update(frame_id, frame, frame_data)

    def load(self, dir_name, no_load=0):
        """
        :type dir_name: str
        :rtype: bool

        """

        self._logger = CustomLogger(self._orig_logger, names=('load',), key='custom_header')

        if no_load:
            self._logger.warning('skipping trained target loading')
        else:
            if self._params.save_zip:
                file_path = os.path.abspath('{:s}.zip'.format(dir_name))
                if not os.path.exists(file_path):
                    raise IOError('Trained target file {:s} does not exist'.format(file_path))
                self._logger.info('Loading trained target from {:s}'.format(file_path))
                # extract saved zip file to temporary directory
                with zipfile.ZipFile(file_path, "r") as z:
                    z.extractall(os.path.abspath('{:s}/'.format(dir_name)))
            else:

                assert os.path.isdir(dir_name), 'Trained target folder {:s} does not exist'.format(dir_name)

                # if not os.path.isdir(dir_name):
                #     msg = 'Trained target folder {:s} does not exist'.format(dir_name)
                #     if self._params.allow_missing_load:
                #         self._logger.warning(msg)
                #         return True
                #     else:
                #         self._logger.error(msg)
                #         return False

                self._logger.info('Loading trained target from {:s}'.format(dir_name))

            if self._params.override == 2:
                self._logger.warning('Skipping loading of saved parameters\n')
            else:

                debug = self._params.debug
                verbose = self._params.verbose

                params = copy.copy(self._params)

                try:
                    paramparse.load(params, dir_name, prefix='trainer')
                    paramparse.read(params, dir_name, prefix='trainer',
                                    out_name='params.txt', allow_unknown=1)
                except AttributeError as e:
                    self._logger.warning('Exception in loading parameters: {}'.format(e))

                """optional runtime parameter overriding"""
                if self._params.override and self._args_in:
                    """temporary hack for debugging
                    partially override trained target parameters at test time"""

                    self._logger.warning('Overriding following trained target parameters:\n{}'.format(
                        pformat(self._args_in)))
                    paramparse.process(params, self._args_in, prog='target', usage=None)

                self._params = params

                """temporary hack for debugging"""
                self._params.debug = debug
                self._params.verbose = verbose

        """temporary hack for debugging"""
        self._params.target.set_verbosity(self._params.verbose, self._params.verbose_depth)

        self._set_rgb()

        self.target = Target(self._params.target, self.rgb_input, self._logger)

        if not no_load:
            if not self.target.load(dir_name):
                self._logger.error('Trained target loading failed')
                return False
            if self._params.save_zip:
                """remove temporary directory"""
                shutil.rmtree(dir_name)
        return True

    def save(self, dir_name, log_file=''):
        """
        :type dir_name: str
        :type log_file: str
        :rtype: bool
        """
        if dir_name.endswith('.zip'):
            dir_name = linux_path(os.path.dirname(dir_name),
                                  os.path.splitext(os.path.basename(dir_name))[0])
        os.makedirs(dir_name, exist_ok=True)

        if log_file:
            if not os.path.isfile(log_file):
                self._logger.warning('nonexistent log file: {}'.format(log_file))
            else:
                dst_log_file = linux_path(dir_name, os.path.basename(log_file))
                self._logger.info('Moving log to {:s}'.format(dst_log_file))

                shutil.copy(log_file, dst_log_file)
                try:
                    os.remove(log_file)
                except PermissionError:
                    pass

        paramparse.save(self._params, dir_name)
        paramparse.write(self._params, dir_name, prefix='trainer', out_name='params.txt')

        # if self.state_to_train == MDPStates.lost:
        #     self._target.lost.save(dir_name)
        # elif self.state_to_train == MDPStates.tracked:
        #     self._target.tracked.save(dir_name)
        # elif self.state_to_train == MDPStates.active:
        #     self._target.active.save(dir_name)
        # else:
        #     self._target.save(dir_name)

        self.target.save(dir_name, self._params.summary_dir)

        if self._params.save_zip:
            """compress directory to zip file"""
            shutil.make_archive(dir_name, 'zip', dir_name)
            """remove temporary directory"""
            shutil.rmtree(dir_name)
            self._logger.info('Trained target saved to {:s}.zip'.format(dir_name))
        else:
            self._logger.info('Trained target saved to {:s}'.format(dir_name))
