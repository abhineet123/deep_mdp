import numpy as np
from trackers.tracker_base import TrackerBase
from utilities import TrackingStatus


class IdentityTracker(TrackerBase):
    """
    returns input locations as output ones without any tracking
    """

    class Params(TrackerBase.Params):
        def __init__(self):
            TrackerBase.Params.__init__(self)

    class Result(TrackerBase.Result):
        def __init__(self, locations):
            TrackerBase.Result.__init__(self, locations,None, None, None, None, 1)

        def get_status(self, template_id=None, track_id=None):
            raise NotImplementedError('IdentityTracker has no status')

        def get_status_m(self, track_id):
            raise NotImplementedError('IdentityTracker has no status')

        def set_status(self, valid_idx, invalid_idx, track_id=0):
            raise NotImplementedError('IdentityTracker has no status')

        def get_scores(self, track_id):
            raise NotImplementedError('IdentityTracker has no scores')

        def get_success_ids(self, track_id):
            raise NotImplementedError('IdentityTracker has no success_ids')

        def _get_best_template(self, template_ids, track_id):
            return 0

    def __init__(self, **kwargs):
        """

        :param params:
        :param n_templates:
        :param update_location:
        :param rgb_input:
        :param logger:
        :param parent:
        :param policy_name:
        """
        TrackerBase.__init__(self, 'no_tracker', **kwargs)

        if self._parent is None:
            self.locations = next(self._copy_ids_gen)
            self._register()
        else:
            self._spawn()

        # self._create_object = functools.partial(IdentityTracker, **kwargs)

        """needed for intellisense"""
        self._params = kwargs["params"]  # type: IdentityTracker.Params

        assert not self._params.roi.enable, "ROI mode is not supported"

    def get_init_samples(self):
        raise NotImplementedError("init samples are not supported")

    def _initialize(self, template_id, frame, bbox):
        pass

    def _update(self, template_id, frame, bbox):
        pass

    def _track(self, frame, frame_id, locations):
        assert not self._heuristics, "heuristics are not supported"

        self.locations = np.expand_dims(locations, axis=1)
        self.locations = np.tile(self.locations, (1, self._n_templates, 1))

        return IdentityTracker.Result(self.locations)


class GTTracker(TrackerBase):
    class Params(TrackerBase.Params):
        def __init__(self):
            TrackerBase.Params.__init__(self)
            self.indicate_exit = 1

    class Result(TrackerBase.Result):
        def __init__(self, locations, features=None, status=None, conf=None):
            TrackerBase.Result.__init__(self, locations, features, status, conf)

        def _get_best_template(self, template_ids, track_id):
            return 0

    def __init__(self, **kwargs):
        """

        :param params:
        :param n_templates:
        :param update_location:
        :param rgb_input:
        :param logger:
        :param parent:
        :param policy_name:
        """
        TrackerBase.__init__(self, 'gt_tracker', **kwargs)

        self.locations = None

        if self._parent is None:
            self._register()
            if self._params.indicate_exit:
                self._logger.info('exit indication is enabled')
            else:
                self._logger.warning('exit indication is disabled')
        else:
            self._spawn()

        """needed for intellisense"""
        self._params = kwargs["params"]  # type: GTTracker.Params

        assert not self._params.roi.enable, "ROI mode is not supported"

    def get_init_samples(self):
        raise NotImplementedError("init samples are not supported")

    def _initialize(self, template_id, frame, bbox):
        self._init_location = np.copy(bbox)

    def _update(self, template_id, frame, bbox):
        pass

    def _track(self, frame, frame_id, locations):
        assert self._annotations is not None and self._traj_idx_by_frame is not None, \
            "valid annotations must be provided for the GT tracker to work"

        if locations is not None:
            n_objs = locations.shape[0]
        else:
            n_objs = 1

        self.n_objs = n_objs

        try:
            self._curr_ann_idx = self._traj_idx_by_frame[frame_id]
        except:
            if self._params.indicate_exit:
                self.locations = None
                return None

            if locations is not None:
                exp_locations = np.expand_dims(locations, axis=1)
                locations_tiled = np.tile(exp_locations, (1, self._n_templates, 1))
                self.locations = locations_tiled
            else:
                exp_locations = np.expand_dims(np.expand_dims(self._init_location, axis=0), axis=0)
                locations_tiled = np.tile(exp_locations, (n_objs, self._n_templates, 1))
                self.locations = locations_tiled

        else:
            gt_location = self._annotations.data[self._curr_ann_idx[0], 2:6].copy()
            self.gt_location = gt_location

            exp_locations = np.expand_dims(np.expand_dims(gt_location, axis=0), axis=0)
            self.locations = np.tile(exp_locations, (n_objs, self._n_templates, 1))

        self.features = np.ones((n_objs, self._n_templates, 1), dtype=np.float32)
        self.conf = np.ones((n_objs, self._n_templates), dtype=np.float32)
        self.status = np.full((n_objs, self._n_templates), TrackingStatus.success)

        assert self.locations.shape == (n_objs,  self._n_templates, 4), "annoying invalid locations"
        track_res = GTTracker.Result(self.locations, self.features, self.status, self.conf)
        assert track_res.locations.shape == (n_objs,  self._n_templates, 4), "annoying invalid track_res locations"

        return track_res
