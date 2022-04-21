import numpy as np
import sys
import cv2

from utilities import MDPStates, draw_box, resize_ar, write_to_files, BaseParams, annotate_and_show


class History:
    class Params(BaseParams):
        """

        :ivar predict: 'motion prediction parameters'
        :ivar max_interp_frame_diff: 'Maximum difference between the frame ID in the last entry in the history and '
                         'that of the current frame for interpolation to be used to fill in the gap '
                         'between them - if the frame difference is greater than this, then '
                         'interpolation is not used',
        :ivar vis: 'Enable diagnostic visualization',
        :ivar pause_for_debug: 'pause execution for debugging',

        """

        class Prediction:
            """

            :ivar n_frames: 'Maximum number of frames from the history to consider for computing the '
                           'predicted location',
            :ivar enable_size: 'Enable predicting the size of the object too-if this is disabled, then the size'
                    'is taken from the last entry in the history',
            """

            def __init__(self):
                self.model = 0
                self.n_frames = 10
                self.enable_size = 0

        def __init__(self):
            self.predict = History.Params.Prediction()
            self.max_interp_frame_diff = 5
            self.pause_for_debug = 0
            self.vis = 0

    def __init__(self, params, logger, parent=None):
        """
        :type params: History.Params
        :type logger: logging.RootLogger | CustomLogger
        :type parent: History | None
        :rtype: None
        """
        self._params = params
        self._logger = logger

        if parent is not None:
            self._pause = parent._pause
            self._test_mode = parent._test_mode
            self._predict_type = parent._predict_type
        else:
            self._pause = 1
            self._test_mode = 0

            if self._params.predict.model == -1:
                self._logger.info('using last known location as the predicted one')
                self._predict_type = "last_location"
            elif self._params.predict.model == 0:
                self._predict_type = "uniform_velocity"
                self._logger.info('using uniform velocity model for motion prediction')
            elif self._params.predict.model == 1:
                self._predict_type = "random_walk"
                self._logger.info('using random walk model for motion prediction')
            else:
                raise AssertionError(f'invalid prediction model: {self._params.predict.model}')

        self.locations = np.zeros((1, 4))
        self.ids = None
        self.states = None
        self.frame_ids = None
        self._predict = getattr(self, "_" + self._predict_type)

        """apparently not actually used for tracking – just stored as part of the Target past information"""
        self.scores = None

        self.size = 0

        self.n_lost = 0
        self.n_tracked = 0
        self.lost_ratio = 0

        self.frame_h = self.frame_w = None

    def train_mode(self):
        self._test_mode = 0

    def test_mode(self):
        self._test_mode = 1

    def initialize(self, id_, frame_id, frame, location, state, score):
        """
        :type id_: int
        :type frame_id: int
        :type location: np.ndarray
        :type state: int
        :type score: float
        :rtype: None
        """
        assert state == MDPStates.tracked, "History can only be initialized in tracked state"
        self.ids = np.array((id_,))
        self.frame_ids = np.array((frame_id,))
        self.locations = np.zeros((1, 4))
        self.locations[:] = location
        self.states = np.array((state,))
        self.scores = np.array((score,))
        self.n_lost = 0
        self.n_tracked = 1
        self.lost_ratio = 0
        self.size = 1
        self.frame_h, self.frame_w = frame.shape[:2]

    def update(self, id_, frame_id, location, state, score, check_frame_id):
        """
        :type id_: int
        :type frame_id: int
        :type location: np.ndarray
        :type state: int
        :type score: float
        :type check_frame_id: bool
        :rtype: None
        """
        if check_frame_id:
            if self.frame_ids[-1] == frame_id:
                """changing the info for an existing frame
                """
                self.reduce(np.array(range(self.frame_ids.size - 1)))

        self.locations = np.concatenate((self.locations, location.reshape((1, 4))), axis=0)
        self.ids = np.concatenate((self.ids, np.array((id_,))), axis=0)
        self.states = np.concatenate((self.states, np.array((state,))), axis=0)
        self.frame_ids = np.concatenate((self.frame_ids, np.array((frame_id,))), axis=0)
        self.scores = np.concatenate((self.scores, np.array((score,))), axis=0)
        self.size += 1

        if state == MDPStates.tracked:
            self.n_tracked += 1
        elif state == MDPStates.lost:
            self.n_lost += 1

        self.lost_ratio = float(self.n_lost) / float(self.n_tracked)

    def reduce(self, idx):
        """
        :type idx: np.ndarray
        :rtype: None
        """
        self.locations = self.locations[idx, :]
        self.ids = self.ids[idx]
        self.states = self.states[idx]
        self.frame_ids = self.frame_ids[idx]
        self.scores = self.scores[idx]
        self.size = self.frame_ids.size

        self.n_tracked = len([s for s in self.states if s == MDPStates.tracked])
        self.n_lost = len([s for s in self.states if s == MDPStates.lost])

        if self.n_tracked > 0:
            self.lost_ratio = float(self.n_lost) / float(self.n_tracked)
        else:
            self.lost_ratio = np.inf

    def interpolate(self, id_, frame_id, location, state, score, check_frame_id):
        """

        add interpolated object info (location, score, etc.) for all intervening frames between
        the current frame and the last one where it was in tracked state to  get a complete trajectory through
        frames it was in lost state

        :type id_: int
        :type frame_id: int
        :type location: np.ndarray
        :type state: int
        :type score: float
        :type check_frame_id: bool
        :rtype: None
        """
        # Only those bounding boxes that correspond to tracked states
        index = np.flatnonzero(self.states == MDPStates.tracked)
        if index.size:
            # last frame in the tracked state
            ind = index[-1]
            fr1 = self.frame_ids[ind]
            fr2 = frame_id
            frame_diff = fr2 - fr1
            if 1 < frame_diff <= self._params.max_interp_frame_diff:
                """remove all data after the last tracked state"""
                self.reduce(np.array(range(ind + 1)))

                last_location = self.locations[-1, :]
                last_score = self.scores[-1]
                # linear interpolation
                for fr in range(fr1 + 1, fr2):
                    interp_factor = float(fr - fr1) / float(frame_diff)
                    interp_location = last_location + ((location - last_location) * interp_factor)
                    interp_score = last_score + ((score - last_score) * interp_factor)
                    self.update(id_, fr, interp_location, state, interp_score, check_frame_id)
        self.update(id_, frame_id, location, state, score, check_frame_id)

    def predict(self, frame_id, frame):

        tracked_idx = np.flatnonzero(np.equal(self.states, MDPStates.tracked))
        if tracked_idx.size > self._params.predict.n_frames:
            # get the required number of frames from the end
            tracked_idx = tracked_idx[-self._params.predict.n_frames:]
        if tracked_idx.size == 0:
            # return the last location as the history doesn't have any tracked locations
            new_location = self.locations[-1, :].reshape((1, 4))
        elif tracked_idx.size == 1:
            # return the last tracked location as the history doesn't have enough tracked locations to compute
            # velocity
            new_location = self.locations[tracked_idx[0], :].reshape((1, 4))
        else:
            new_location = self._predict(frame_id, tracked_idx)

        min_x, min_y, w, h = new_location.squeeze()
        max_x, max_y = min_x + w, min_y + h
        if min_x >= max_x or min_y >= max_y or \
                min_x < 0 or min_y < 0:
            # max_x >= self.frame_w or max_y >= self.frame_h:
            new_location = self.locations[-1, :].reshape((1, 4))
            # self._logger.warning('invalid predicted location found: {} so reverting to the last one: {}'.format(
            #     [min_x, min_y, max_x, max_y], new_location))

        if self._params.vis:
            frame_disp = np.copy(frame)
            for _id in range(self.size):
                if self.states[_id] == MDPStates.tracked:
                    color = 'green'
                else:
                    color = 'red'
                bbox = self.locations[_id, :]
                draw_box(frame_disp, bbox, color=color, _id=_id)

            draw_box(frame_disp, new_location, color='blue')
            annotate_and_show('history predict', frame_disp)

        return new_location

    def _last_location(self, frame_id, tracked_idx):
        locations = self.locations[tracked_idx, :]

        new_location = locations[-1, :].reshape((1, 4))

        return new_location

    def _uniform_velocity(self, frame_id, tracked_idx):
        """
        :type frame_id: int
        :rtype: np.ndarray
        """

        sizes = self.locations[tracked_idx, 2:]
        centers = self.locations[tracked_idx, :2] + (sizes / 2.0)
        n_dim = 2
        if self._params.predict.enable_size:
            locations = np.concatenate((centers, sizes), axis=1)
            n_dim = 4
        else:
            locations = centers

        frame_ids = self.frame_ids[tracked_idx]

        locations_diff = locations[1:, :] - locations[:-1, :]
        frames_diff = (frame_ids[1:] - frame_ids[:-1]).reshape((-1, 1))
        velocities = locations_diff / frames_diff

        mean_velocity = np.mean(velocities, axis=0).reshape((1, 2))

        # if tracked_idx.size > 1:
        #     velocity /= tracked_idx.size - 1

        new_location = locations[-1, :].reshape((1, n_dim)) + mean_velocity * (frame_id - frame_ids[-1] + 1)

        if not self._params.predict.enable_size:
            new_location = np.concatenate(
                (new_location.reshape((1, 2)), sizes[-1, :].reshape((1, 2))), axis=1)

        # convert center to ul
        new_location[0, :2] -= new_location[0, 2:] / 2.0

        # new_location_size = new_location[0, :2] + new_location[0, 2:] - 1

        # if (new_location < 0).any():
        #     # predicted location has negative values so just return the last known tracked location instead
        #     return self.locations[tracked_idx[-1], :].reshape((1, 4))

        return new_location

    def _random_walk(self, frame_id, tracked_idx):
        locations = self.locations[tracked_idx, :]

        """magnitude of change in location between consecutive frames"""
        locations_diff = np.abs(locations[1:, :] - locations[:-1, :]).reshape((tracked_idx.size - 1, 4))

        mean_locations_diff = np.mean(locations_diff, axis=0)

        step_set = [-1, 0, 1]
        step_shape = (1, 4)
        steps = np.random.choice(a=step_set, size=step_shape)

        new_location = locations[-1, :].reshape((1, 4)) + steps * mean_locations_diff

        return new_location

# def write_state_info(self, files, root_dir, write_to_bin,
#                      fp_fmt='%.4f', fp_dtype=np.float32):
#     """
#     :type files: list[(str, Type(np.dtype), tuple)]
#     :type root_dir: str
#     :type write_to_bin: bool
#     :type fp_fmt: str
#     :type fp_dtype: Type(np.dtype)
#     :rtype: None
#     """
#     log_dir = '{:s}/history'.format(root_dir)
#     entries = (
#         (self.locations, 'locations', fp_dtype, fp_fmt),
#         (self.scores, 'scores', fp_dtype, fp_fmt),
#         (self.ids, 'ids', np.uint32, '%d'),
#         (self.frame_ids, 'frame_ids', np.uint32, '%d'),
#         (self.states, 'states', np.uint8, '%d')
#     )
#     write_to_files(log_dir, write_to_bin, entries)
#     # files.extend(['history/{:s}'.format(entry[1]) for entry in entries])
#     files.extend([('history/{:s}'.format(entry[1]), entry[2], entry[0].shape) for entry in entries])
