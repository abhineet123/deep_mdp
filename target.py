import numpy as np
import copy

from active import Active
from lost import Lost
from tracked import Tracked
from templates import Templates
from trackers.tracker_base import TrackerBase
from history import History
from input import Annotations

from paramparse import copy_recursive
from utilities import MDPStates, compute_overlap, CustomLogger, BaseParams, SaveModes


class Target:
    """
    :type params: Target.Params
    :type _logger: logging.RootLogger | CustomLogger
    :type _rgb_input: int | bool
    :type id_: int
    :type frame_id: int
    :type location: np.ndarray
    :type max_iou_det_score: float
    :type templates: Templates
    :type history: History
    :type active: Active
    :type tracked: Tracked
    :type lost: Lost
    :type state: int
    :type prev_state: int
    :type rep: Target | None
    """

    class Params(BaseParams):
        """
        :type pause_for_debug: int
        :type verbose: int
        :type history: History.Params
        :type templates: Templates.Params
        :type active: Active.Params
        :type lost: Lost.Params
        :type tracked: Tracked.Params


        :ivar shared_model: 'share model between lost and tracked policies: '
                '1: lost model with tracked '
                '2: tracked model with lost ',
        :ivar interp_on_assoc: fill in intervening frames on successful association and resultant lost --> tracked
        transition by interpolating between the two nearest tracked frames and change the states of all these frames
        to tracked
        :ivar history: 'History parameters (specified in History.py)',
        :ivar templates: 'Templates parameters (specified in Templates.py)',
        :ivar active: 'Active state parameters (specified in Active.py)',
        :ivar lost: 'Lost state parameters (specified in Lost.py)',
        :ivar tracked: 'Tracked state parameters (specified in Tracked.py)',
        :ivar pause_for_debug: 'pause execution for debugging',
        :ivar verbose: 'Enable printing of some general diagnostic messages',

        """

        def __init__(self):
            self.shared_model = 0

            """allow for some annoying ad-hoc heuristics in the original code where 
            transitions to inactive state happen outside of the policies
            """
            self.check_out_of_scene = 1
            self.check_next_frame = 1

            """allow for annoying ad-hoc heuristics"""
            self.next_frame_exit_threshold = 0.05

            """lost --> tracked transition: fill in intervening frames by interpolating 
                between the two nearest tracked frames and change the states of all these frames to tracked"""
            self.interp_on_assoc = 1

            self.history = History.Params()
            self.templates = Templates.Params()
            self.active = Active.Params()
            self.lost = Lost.Params()
            self.tracked = Tracked.Params()

            self._load_states = []
            self._save_states = []

            self._replacement = None
            self._copy_excluded = ['_replacement']

            self.pause_for_debug = 0
            self.verbose = 0

        @property
        def save_states(self):
            return self._save_states

        @save_states.setter
        def save_states(self, val):
            self._save_states = val

        @property
        def load_states(self):
            return self._load_states

        @load_states.setter
        def load_states(self, val):
            self._load_states = val

        @property
        def replacement(self):
            return self._replacement

        @replacement.setter
        def replacement(self, val):
            self._replacement = val

        def set_verbosity(self, verbose, depth):
            if depth >= 1:
                self.verbose = verbose
            if depth >= 2:
                self.tracked.verbose = verbose
                self.lost.verbose = verbose
                self.templates.verbose = verbose
            if depth >= 3:
                self.active.svm.verbose = verbose
                self.lost.svm.verbose = verbose
                self.active.mlp.verbose = verbose
                self.lost.mlp.verbose = verbose
                self.templates.lk.verbose = verbose
                self.templates.siamese.verbose = verbose

    def __init__(self, params, rgb_input, logger, parent=None, batch_train=None):
        """
        :type params: Target.Params
        :type rgb_input: int | bool
        :type logger: logging.RootLogger | CustomLogger
        :type parent: Target | None
        :rtype: None
        """
        self.params = Target.Params()
        copy_recursive(params, self.params, include_protected=1)

        self._logger = logger
        self._rgb_input = rgb_input

        if parent is not None:
            # tracker = target.tracker
            templates = parent.templates
            history = parent.history
            active = parent.active
            tracked = parent.tracked
            lost = parent.lost
            self.rep = parent.rep
            self.rep_params = parent.rep_params
            self.rep_prefix = parent.rep_prefix
        else:
            templates = history = None
            active = tracked = lost = None
            self._test_mode = 0
            self.rep = params.replacement
            # replace_modules = 0
            self.rep_params = None
            self.rep_prefix = []

            if self.rep is not None:
                self.rep_params = self.rep.params
                # replace_modules = 1

                active = self.rep.active
                tracked = self.rep.tracked
                lost = self.rep.lost
                templates = self.rep.templates
                history = self.rep.history

                if templates is not None:
                    self._logger.warning(
                        'replacing templates module with tracker switch: {} --> {}'.format(
                            self.params.templates.tracker, self.rep_params.templates.tracker
                        ))
                    self.params.templates = self.rep_params.templates
                    self.rep_prefix.append('tracker_{}'.format(self.rep_params.templates.tracker))

                if active is not None:
                    self._logger.warning(
                        'replacing active module with model switch: {} --> {}'.format(
                            self.params.active.model, self.rep_params.active.model
                        ))
                    self.params.active = self.rep_params.active
                    self.rep_prefix.append('active_{}'.format(self.rep_params.active.model))

                if lost is not None:
                    self._logger.warning(
                        'replacing lost module with model switch: {} --> {}'.format(
                            self.params.lost.model, self.rep_params.lost.model
                        ))
                    self.params.lost = self.rep_params.lost
                    self.rep_prefix.append('lost_{}'.format(self.rep_params.lost.model))

                if tracked is not None:
                    self._logger.warning(
                        'replacing tracked module with model switch: {} --> {}'.format(
                            self.params.tracked.model, self.rep_params.tracked.model
                        ))
                    self.params.tracked = self.rep_params.tracked
                    self.rep_prefix.append('tracked_{}'.format(self.rep_params.tracked.model))

                if history is not None:
                    self._logger.warning(
                        'using history module from replacement target with model switch: {} --> {}'.format(
                            self.params.history.predict.model, self.rep_params.history.predict.model
                        ))
                    self.params.history = self.rep_params.history
                    self.rep_prefix.append('history_{}'.format(self.rep_params.history.predict.model))

            if not self.params.interp_on_assoc:
                self._logger.warning(f'history interpolation on association is disabled')

        if batch_train in (None, MDPStates.tracked, MDPStates.lost):
            """hack to avoid unnecessary computations in templates if annoying heuristics are not needed in lost
             """
            annoying_heuristics = self.params.lost.model_heuristics
            """set of patches and features representing the target appearance over time"""
            self.templates = Templates(self.params.templates, self._rgb_input, annoying_heuristics,
                                       self._logger, templates)

        """MDP policies for the different states"""
        self.active = self.tracked = self.lost = None

        if batch_train in (None, MDPStates.active):
            self.active = Active(self.params.active, self._rgb_input, self._logger, active)

        if batch_train in (None, MDPStates.tracked):
            external_model = self.params.shared_model == 1
            self.tracked = Tracked(self.templates, self.params.tracked, self._rgb_input, external_model, self._logger,
                                   tracked)

        if batch_train in (None, MDPStates.lost):
            self.lost = Lost(self.templates, self.params.lost, self._logger, lost)

        if batch_train is not None:
            return

        """stuff not needed for batch training of individual policies
        """

        """unique ID associated with the target"""
        self.id_ = 0
        """ID of the first processed frame"""
        self.start_frame_id = 0
        """ID of the latest processed frame"""
        self.frame_id = 0
        """no. of frames for which target has existed"""
        self.n_frames = 0
        """ID of the latest processed frame relative to its trajectory (only training)"""
        # self.traj_start_id = 0
        """latest processed frame"""
        self._frame = None
        """latest processed detections"""
        self._detections = None
        """target location in the current frame"""
        self.location = np.zeros((1, 4))
        self._next_predicted_location = np.zeros((1, 4))
        """score of the maximally overlapping detection with the target location"""
        self.max_iou_det_score = 0

        """set of all processed frames for a particular object"""
        self.history = History(self.params.history, self._logger, history)

        if parent is None:
            if self.params.shared_model == 1:
                self._logger.info(f'Sharing lost policy model with tracked policy')
                self.tracked.set_model(self.lost.get_model())
            elif self.params.shared_model == 2:
                self._logger.info(f'Sharing tracked policy model with lost policy')
                self.lost.set_model(self.tracked.get_model())

        self.predicted_location = np.zeros((1, 4))
        self.state = MDPStates.active
        self.prev_state = MDPStates.active

        self.load_dir = self.save_dir = None

        """training and debugging"""
        self._annotations = None
        self._traj_id = None
        self._traj_idx_by_frame = None
        self._curr_ann_idx = None
        self.ann_status = None
        self.assoc_to_tracked_counter = 0

        self._tracker = self.templates.tracker  # type: TrackerBase

    def batch_train(self, state):
        if state == MDPStates.active:
            self.active.batch_train()
        elif state == MDPStates.lost:
            self.lost.batch_train()
        elif state == MDPStates.tracked:
            self.tracked.batch_train()

    def set_logger(self, logger):
        self._logger = logger

    def set_tb_writer(self, writer):
        self.lost.set_tb_writer(writer)
        self.tracked.set_tb_writer(writer)
        self.active.set_tb_writer(writer)

    def test_mode(self):
        """

        :param int mode:
        0: training mode,
        >0: testing mode,
        1: no samples saved,
        2: all samples saved,
        3: samples with incorrect decisions saved

        :return:
        """
        self._test_mode = 1
        self.lost.test_mode()
        self.tracked.test_mode()
        self.active.test_mode()
        self.templates.test_mode()
        self.history.test_mode()

    def train_mode(self):
        self._test_mode = 0
        self.lost.train_mode()
        self.tracked.train_mode()
        self.active.train_mode()
        self.templates.train_mode()
        self.history.train_mode()

    def state_str(self):
        return MDPStates.to_str[self.state]

    def initialize(self, obj_id, frame_id, frame, curr_det_data,
                   annotations=None, traj_id=None, ann_status=None):
        """
        to be called once for the first frame of the trajectory

        :type obj_id: int
        :type frame_id: int
        :type frame: np.ndarray
        :type curr_det_data: np.ndarray
        :type annotations: Annotations | None
        :type traj_id: int | None
        :type ann_status: str
        :rtype: None
        """
        self.id_ = obj_id
        self.start_frame_id = frame_id
        self.frame_id = frame_id
        self.location[:] = curr_det_data[2:6]
        self.max_iou_det_score = curr_det_data[6]
        self.n_frames = 1

        self.ann_status = ann_status
        self._annotations = annotations
        self._traj_id = traj_id

        if traj_id is None:
            self._traj_idx_by_frame = {}
            self._curr_ann_idx = None
        else:
            self._traj_idx_by_frame = self._annotations.traj_idx_by_frame[traj_id]
            try:
                self._curr_ann_idx = self._traj_idx_by_frame[frame_id]
            except KeyError:
                self._curr_ann_idx = None

        self._tracker.set_gt(self._annotations, self._traj_idx_by_frame)

        """ transition to the tracked state"""
        self.state = MDPStates.tracked
        """initialize the state policies"""
        self.active.initialize(self.id_)
        self.lost.initialize(self.id_, frame_id, frame, self.location,
                             self._annotations, self._curr_ann_idx, self.ann_status)
        self.tracked.initialize(self.id_, frame_id, frame, self.location,
                                self._annotations, self._curr_ann_idx, self.ann_status)
        """initialize templates and history"""
        self.templates.initialize(self.id_, frame_id, frame, self.location)
        self.history.initialize(self.id_, frame_id, frame, self.location, self.state, self.max_iou_det_score)

    def _check_detections(self, detections):
        if not detections.shape[0]:
            return detections

        valid_det_ids = []
        for i in range(detections.shape[0]):
            det = detections[i, :]
            if not np.isfinite(det).all() or np.any(np.isnan(det)):
                self._logger.warning('ignoring invalid detection {}: {}'.format(i, det))
            else:
                valid_det_ids.append(i)

        detections = detections[valid_det_ids, :]
        return detections

    def update(self, frame, frame_id, detections, associate, assoc_with_ann):
        """
        to be called for each subsequent frame of the trajectory

        :type frame: np.ndarray
        :type frame_id: int
        :type detections: np.ndarray
        :param int associate
        :param int assoc_with_ann

        :rtype: None | int
        """

        assert frame_id > 0, "Target cannot be updated in the first frame"

        # detections = self._check_detections(detections)

        self.frame_id = frame_id
        self._frame = frame
        self._detections = detections

        self._tracker.update_frame(self._frame, self.frame_id)

        self.n_frames += 1

        if self._traj_idx_by_frame is not None:
            try:
                self._curr_ann_idx = self._traj_idx_by_frame[frame_id]
            except KeyError:
                self._curr_ann_idx = None
        else:
            self._curr_ann_idx = None

        assert self.state != MDPStates.active, 'Target cannot be updated in the active state'

        if self.state == MDPStates.tracked:
            self.lost.reset_streak()
            self.prev_state = MDPStates.tracked
            self.predicted_location[:] = self.history.predict(self.frame_id, self._frame)

            tracking_scores = self.tracked.update(self._frame, self.frame_id, self._detections, self.predicted_location,
                                                  self._curr_ann_idx)  # type: np.ndarray

            if self.tracked.state == MDPStates.inactive:
                self.state = MDPStates.inactive
                return

            """for some annoying reason, similarity is updated irrespective of whether or not tracking was successful
            in fact is only used in _get_features and is updated for each detection right before its use so this seems 
            like another of the insidious bugs
            """
            # self.lost.update_similarity(frame, self.location)

            self.state = self.tracked.state

            if np.isfinite(self.tracked.location).all():
                self.location[:] = self.tracked.location
                score = 1.0
            else:
                score = self.history.scores[-1]

            """update history irrespective of the state"""
            self.history.update(self.id_, self.frame_id, self.location, self.state, score,
                                check_frame_id=False)

            if self.state == MDPStates.tracked:
                assert tracking_scores is not None, "tracking_scores cannot be None for successful tracking"

                """successful tracking --> tracked to tracked --> add new template but don't set it as anchor
                """
                self.templates.update(self._frame, self.frame_id, self._detections, self.location, tracking_scores,
                                      change_anchor=False)
            # if self.params.pause_for_debug:
            #     self._logger.debug('paused')

        elif self.state == MDPStates.lost:
            """target got lost in the previous frame or earlier
            """

            """temporarily disabling the resetting of tracked streak to maintain
            correspondence with the original code for debugging purposes
            """

            # self.tracked.resetStreak()

            self.predicted_location[:] = self.history.predict(self.frame_id, self._frame)
            self.prev_state = MDPStates.lost

            # if annotations is not None and ann_idx is None:
            #     ann_idx, annotations = self._infer_target(frame_id, annotations)

            self.lost.update(self._frame, self.frame_id, detections, self.predicted_location, self.history.locations[-1, :],
                             self._curr_ann_idx)

            if self.lost.state == MDPStates.inactive:
                self.state = MDPStates.inactive
                return

            # if self.params.pause_for_debug:
            #     self._logger.debug('paused')

            if associate:
                self.associate(assoc_with_ann=assoc_with_ann)
        else:
            """target is active or inactive"""
            raise AssertionError(f'Target cannot be updated in state: {MDPStates.to_str(self.state)}')

        # if self.params.pause_for_debug:
        #     self._logger.debug('paused')

    def associate(self, assoc_with_ann=False, assoc_det_id=None):
        """
        called in the lost state to associate target with the given detection

        :type assoc_det_id: int | None
        :type assoc_with_ann: bool
        :type ann_idx: np.ndarray
        :rtype: None
        """
        assert self.state == MDPStates.lost, "Association can only be done in the lost state"

        tracking_scores = self.lost.associate(assoc_with_ann, assoc_det_id)  # type: np.ndarray
        self.state = self.lost.state
        if self.state == MDPStates.tracked:
            assert tracking_scores is not None, "tracking_scores cannot be None for successful association"

            self.location[:] = self.lost.location
            self.max_iou_det_score = self.lost.score
            if self.params.verbose:
                print('Target {:d} associated to detection {:d}'.format(
                    self.id_, self.lost.assoc_det_id + 1))

            if self.params.interp_on_assoc:
                """lost --> tracked transition: fill in intervening frames by interpolating 
                between the two nearest tracked frames and change the states of all these frames to tracked
                """
                self.history.interpolate(self.id_, self.frame_id, self.location, self.state, 1,
                                         check_frame_id=True)

            """successful association --> lost to tracked --> add new template and set it as anchor
            """
            self.templates.update(self._frame, self.frame_id, self._detections, self.location,
                                  tracking_scores, change_anchor=True)

            """Allow the patch tracker template to be reinitialized in CTM mode since the target might now be in a 
            significantly different location from the last time it was actually tracked"""
            self.tracked.reinitialize(self._frame, self.frame_id, self._detections, self.location)
        else:
            if self.params.verbose:
                print('Target {:d} not associated'.format(self.id_))
                if self.lost.assoc_det_id >= 0:
                    print('\t best matching detection: {:d}'.format(self.lost.assoc_det_id + 1))

            """location and score are not updated in the history if target remains lost"""
            self.location[:] = self.history.locations[-1, :]
            self.max_iou_det_score = self.history.scores[-1]
            self.history.update(self.id_, self.frame_id, self.location, self.state, self.max_iou_det_score,
                                check_frame_id=True)

            # if self.params.pause_for_debug:
            #     self._logger.debug('paused')

    def train(self, state_to_train):
        """
        called once per frame while training to compare the results
        obtained on that frame with annotations and adapt target policies accordingly

        :type state_to_train: MDPStates | None
        :rtype: (bool, int)
        """

        if state_to_train is None:
            state_to_train = self.prev_state

        is_end = False
        reward = 0

        if state_to_train != self.prev_state:
            """
            Asynchronous Training 
            """
            # if self.traj_start_id == self._frame_id:
            #     """No asynchronous training in the first frame of the trajectory
            #     """
            #     return is_end, reward
            # reward = 1
            predicted_location = self.history.predict(self.frame_id, self._frame)

            """have to psdd frame specific data due to the asynchronous nature of training 
            wherein the last policy update might not be in this frame"""
            if state_to_train == MDPStates.tracked:
                self.tracked.train_async(self._frame, self.frame_id, self._detections, predicted_location)
            elif state_to_train == MDPStates.lost:
                prev_location = self.history.locations[-1, :]
                self.lost.train_async(self._frame, self.frame_id, self._detections, predicted_location, prev_location)
            else:
                raise AssertionError('Invalid state_to_train: {}'.format(state_to_train))
            return is_end, reward
        else:
            if state_to_train == MDPStates.tracked:
                reward = self.tracked.train()
            elif state_to_train == MDPStates.lost:
                # predicted_location = self.history.predict_motion(self.frame_id, self.frame)
                is_end, reward = self.lost.train()
            else:
                raise AssertionError('Invalid state_to_train: {}'.format(state_to_train))
            return is_end, reward

    def spawn(self, obj_id, frame_id, frame, curr_det_data, logger,
              annotations=None, traj_id=None, ann_status=None):
        """
        to be called for each new prospective target to be added while testing a trained target
        new target has same parameters as the old one but doesn't ned to share its internal state

        :param  int | None obj_id: ID of the new target it such a one is created,
        None means that only the corresponding sample will be added in active policy

        :type frame_id: int
        :type frame: np.ndarray
        :type curr_det_data: np.ndarray
        :type annotations: Annotations | None
        :type traj_id: int | None
        :type ann_status: str | None
        :type logger: logging.RootLogger | CustomLogger
        :rtype: Target | None
        """
        # if obj_id is None:
        #     is_synthetic = 1
        # else:
        #     is_synthetic = 0

        """decide if this detection corresponds to a valid target"""
        self.active.predict(frame, curr_det_data, ann_status)

        if self.active.state == MDPStates.inactive:
            return None

        target = Target(self.params, self._rgb_input, logger, self)
        """must pass annotations during initialization instead of construction since the trainer 
        shares same target for all sequences"""
        target.initialize(obj_id, frame_id, frame, curr_det_data,
                          annotations, traj_id, ann_status)

        return target

    def reset_stats(self):
        self.active.reset_stats()
        self.lost.reset_stats()
        self.tracked.reset_stats()

    def set_state(self, state):
        """
        :type state: int
        :rtype: None
        """
        self.state = state
        if state != MDPStates.inactive:
            """
            don't update history if target has become inactive
            """
            self.history.update(self.id_, self.frame_id, self.location, self.state, score=1,
                                check_frame_id=True)

    def reset_state(self, id_):
        """
        only training

        :type id_: int
        :rtype: None
        """
        self.state = MDPStates.active
        self.prev_state = MDPStates.active
        self.id_ = id_
        # self.traj_start_id = traj_start_id

    def load(self, load_dir):
        """
        :type load_dir: str
        :rtype: None
        """
        load_states = self.params.load_states
        if not load_states:
            load_states = ['active', 'lost', 'tracked']
            self._logger.info(f'Loading all states')
        else:
            self._logger.info(f'Selectively loading states: {",".join(load_states)}')

        if 'active' in load_states and not self.active.load('{:s}/active'.format(load_dir)):
            raise AssertionError('Active policy loading failed')

        if 'lost' in load_states and not self.lost.load('{:s}/lost'.format(load_dir)):
            raise AssertionError('Lost policy loading failed')

        if 'tracked' in load_states and not self.tracked.load('{:s}/tracked'.format(load_dir)):
            raise AssertionError('Tracked policy loading failed')

        self.load_dir = load_dir
        return True

    def save_test_samples(self, save_dir):
        self.save_dir = save_dir

        save_states = self.params.save_states
        if not save_states:
            save_states = ['active', 'lost', 'tracked']
        else:
            self._logger.info(f'saving test samples for states: {",".join(save_states)}')

        if 'active' in save_states:
            self.active.save_test_samples('{:s}/active'.format(save_dir))

        if 'lost' in save_states:
            self.lost.save_test_samples('{:s}/lost'.format(save_dir))

        if 'tracked' in save_states:
            self.tracked.save_test_samples('{:s}/tracked'.format(save_dir))

    def reset_test_samples(self):
        save_states = self.params.save_states
        if not save_states:
            save_states = ['active', 'lost', 'tracked']
        else:
            self._logger.info(f'resetting test samples for states: {",".join(save_states)}')

        if 'active' in save_states:
            self.active.reset_test_samples()

        if 'lost' in save_states:
            self.lost.reset_test_samples()

        if 'tracked' in save_states:
            self.tracked.reset_test_samples()

    def save(self, save_dir, summary_dir):
        """
        :type save_dir: str
        :type summary_dir: str
        :rtype: None
        """

        self.save_dir = save_dir

        save_states = self.params.save_states
        if not save_states:
            save_states = ['active', 'lost', 'tracked']
        else:
            self._logger.info(f'saving states: {",".join(save_states)}')

        if 'active' in save_states:
            self.active.save('{:s}/active'.format(save_dir))

        if 'lost' in save_states:
            self.lost.save('{:s}/lost'.format(save_dir), summary_dir)

        if 'tracked' in save_states:
            self.tracked.save('{:s}/tracked'.format(save_dir), summary_dir)

    def get_data(self, add_state):
        """
        :type add_state: bool | int
        :rtype: np.ndarray
        """
        x = self.location[0, 0]
        y = self.location[0, 1]
        w = self.location[0, 2]
        h = self.location[0, 3]
        data = [self.frame_id, self.id_, x, y, w, h, self.history.scores[-1], -1, -1, -1]
        if add_state:
            data.append(self.state)
        return np.array(data, dtype=np.float64).reshape((1, len(data)))

    def tracked_exit_heuristics(self, frame):
        """
        :rtype: None
        """
        """temporary allowance for some annoying ad-hoc heuristics in the original code"""
        if self.is_out_of_frame(self.location, frame):
            self.state = MDPStates.inactive
            if self.params.verbose:
                print('target outside image by checking borders')

    def lost_exit_heuristics(self, frame):
        """
        :rtype: None
        """

        # if not self._params.lost_heuristics:
        #     return

        """temporary allowance for some annoying ad-hoc heuristics in the original code"""
        if self.state == MDPStates.tracked:
            self.lost.reset_streak()

        self.lost.apply_heuristics()
        self.state = self.lost.state

        out_of_scene = 0
        if self.params.check_out_of_scene:
            out_of_scene = self.is_out_of_frame(self.location, frame)

        if self.state == MDPStates.lost and self.params.check_next_frame:
            self._next_predicted_location[:] = self.history.predict(self.frame_id + 1, self._frame)
            # temporary debugging measure to ensure correspondence with the original code
            annoying_heuristic_location = np.zeros((1, 4), dtype=np.float64)
            annoying_heuristic_location[0, :2] = self._next_predicted_location[0, :2] + self._next_predicted_location[0,
                                                                                        2:] / 2.0
            annoying_heuristic_location[0, 2:] = self.history.locations[-1, 2:]
            out_of_scene = self.is_out_of_frame(annoying_heuristic_location, frame,
                                                threshold=self.params.next_frame_exit_threshold)
        if out_of_scene:
            self.state = MDPStates.inactive
            if self.params.verbose:
                print('target outside image by checking borders')

    def is_out_of_frame(self, location, frame, threshold=0):
        """
        :type location: np.ndarray
        :type threshold: float
        :rtype: bool
        """
        frame_box = np.array((0, 0, frame.shape[1], frame.shape[0])).reshape((1, 4))

        if threshold <= 0:
            threshold = self.params.tracked.exit_threshold

        ioa_1 = np.empty((1,))
        try:
            compute_overlap(None, ioa_1, None, location.reshape((1, 4)), frame_box)
        except:
            ioa_1[0] = 0

        if ioa_1[0] < threshold:
            return True

        return False

    # def write_state_info(self, root_dir, write_to_bin, fp_fmt='%.4f', fp_dtype=np.float32):
    #     """
    #     :type root_dir: str
    #     :type write_to_bin: bool | int
    #     :type fp_fmt: str
    #     :type fp_dtype: Type(np.dtype)
    #     :rtype: list[str]
    #     """
    #     files = []
    #
    #     if self.prev_state == MDPStates.active:
    #         self.active.write_state_info(files, root_dir, write_to_bin)
    #         include_tracker = 0
    #     else:
    #         if self.prev_state == MDPStates.tracked:
    #             self.tracked.write_state_info(files, root_dir, write_to_bin, fp_fmt=fp_fmt, fp_dtype=fp_dtype)
    #         elif self.prev_state == MDPStates.lost:
    #             self.lost.write_state_info(files, root_dir, write_to_bin, write_roi=True,
    #                                        fp_fmt=fp_fmt, fp_dtype=fp_dtype)
    #         include_tracker = 1
    #     self.templates.write_state_info(files, root_dir, write_to_bin, write_roi=1, write_patterns=0,
    #                                     fp_fmt=fp_fmt, fp_dtype=fp_dtype, include_tracker=include_tracker)
    #     self.history.write_state_info(files, root_dir, write_to_bin, fp_fmt=fp_fmt, fp_dtype=fp_dtype)
    #
    #     return files

    def _get_best_matching_annotation(self, frame_id, annotations):
        frame_ann_idx = annotations.idx[frame_id]
        if frame_ann_idx is None:
            self._logger.warning('No annotations in frame {}'.format(frame_id))
            return None
        n_frame_ann = len(frame_ann_idx)
        target_iou = np.empty((n_frame_ann, 1))
        compute_overlap(target_iou, None, None, self.location.reshape((1, 4)),
                        annotations.data[frame_ann_idx, 2:6].reshape((-1, 4)))
        max_iou_idx = np.argmax(target_iou)
        target_id = int(annotations.data[frame_ann_idx[max_iou_idx], 1])
        # traj_idx = annotations.traj_idx[target_id]
        # curr_ann_data = annotations.data[traj_idx, :]
        return target_id
