import os
import copy
import functools

import numpy as np

from .siamx_rpn.siamx_rpn_utils import load_net
from .siamx_rpn.net import SiamRPNVGG, SiamRPNResNeXt22, SiamRPNPPRes50
from .siamx_models.builder import SiamFC, SiamVGG, SiamFCRes22, SiamFCIncep22, SiamFCNext22
from .siamx_fc.siam_fc_tracker import SiamFCTracker
from .siamx_rpn.siam_rpn_tracker import SiamRPNTracker
from .siamx_fc.siamx_siamese import SiameseNet


class SiamX:
    """
    :type params: SiamX.Params
    :type tracker_params: SiamFCTracker.Params | SiamRPNTracker.Params
    :type logger: logging.RootLogger
    """

    class Params:
        """
        :type fc: SiamFCTracker.Params
        :type rpn: SiamRPNTracker.Params
        :type pretrained_wts_dir: str


        :ivar model: {
            'rpn_vgg': ('SiamRPNVGG', 'SiamRPNVGG.pth'),
            'rpn_nxt': ('SiamRPNResNeXt22', 'SiamRPNResNeXt22.pth'),
            'rpnpp': ('SiamRPNPPRes50', 'SiamRPNPPRes50.pth'),

            'fc': ('SiamFC', 'SiamFC.pth'),
            'fc_vgg': ('SiamVGG', ''),
            'fc_res': ('SiamFCRes22', 'SiamFCRes22_400.pth'),
            'fc_incp': ('SiamFCIncep22', ''),
            'fc_nxt': ('SiamFCNext22', 'siamresnextcheckpoint.pth.tar'),
        }

        """

        def __init__(self):
            self.model = ''
            self.fc = SiamFCTracker.Params()
            self.rpn = SiamRPNTracker.Params()
            # self.rpnpp = SiamRPNPPTrackerParams()
            self.pretrained_wts_dir = 'trackers/siamX/pretrained_weights'

    def __init__(self, params, n_trackers, update_location, logger, parent=None):
        """
        :type params: SiamX.Params
        :type n_trackers: int
        :type logger: logging.RootLogger
        :type parent: SiamX | None
        :rtype: None
        """

        self._params = params
        self._logger = logger
        self._update_location = update_location
        self._n_trackers = n_trackers

        self.members_to_spawn = (
            'model',
            'create_tracker',

        )
        self.members_to_deepcopy = (
            # 'states',
        )

        if parent is not None:
            self.model = parent.model
            self.create_tracker = parent.create_tracker
            tracker = parent.tracker
        else:
            tracker = None
            models_rpn = {
                # RPN
                'rpn_vgg': (SiamRPNVGG, 'SiamRPNVGG', 'SiamRPNVGG.pth'),
                'rpn_nxt': (SiamRPNResNeXt22, 'SiamRPNResNeXt22', 'SiamRPNResNeXt22.pth'),
                # 'rpnpp': (SiamRPNPP, 'SiamRPNPP', 'SiamRPNPP.pth'),
                'rpnpp': (SiamRPNPPRes50, 'SiamRPNPPRes50', 'SiamRPNPPRes50.pth'),
            }
            models_fc = {
                # FC
                'fc': (SiamFC, 'SiamFC', 'SiamFC.pth'),
                'fc_vgg': (SiamVGG, 'SiamVGG', ''),
                'fc_res': (SiamFCRes22, 'SiamFCRes22', 'SiamFCRes22_400.pth'),
                'fc_incp': (SiamFCIncep22, 'SiamFCIncep22', ''),
                'fc_nxt': (SiamFCNext22, 'SiamFCNext22', 'siamresnextcheckpoint.pth.tar'),
            }
            models = {**models_rpn, **models_fc}

            try:
                model_type, model_type_str, pretrained_wts_name = models[self._params.model]
            except KeyError:
                raise AssertionError('Invalid model_type: {}'.format(self._params.model))
            else:
                self.model = model_type()
                pretrained_wts_path = os.path.join(self._params.pretrained_wts_dir, pretrained_wts_name)
                load_net(pretrained_wts_path, self.model)
                self.model.eval()
                self._logger.info(f'Using net type {model_type_str} with pretrained wts from: {pretrained_wts_path}')

            if self._params.model in models_rpn:
                # self.initialize = self.initialize_rpn
                # self.update = self.update_rpn

                # if 'SiamRPNPP' in self.params.model:
                #     self.tracker_params = SiamRPNPPTrackerParams()
                # else:

                self.tracker_params = SiamRPNTracker.Params()
                self.create_tracker = functools.partial(SiamRPNTracker, params=self.tracker_params)

            elif self._params.model in models_fc:
                # self.initialize = self.initialize_fc
                # self.update = self.update_fc

                self.model = SiameseNet(self.model)

                self.tracker_params = SiamFCTracker.Params()
                self.create_tracker = functools.partial(SiamFCTracker, params=self.tracker_params)

        self.tracker = self.create_tracker(net=self.model, update_location=self._update_location, logger=self._logger,
                                           parent=tracker)
        self.score_sz = self.tracker.score_sz

    def initialize(self, tracker_id, init_frame, init_bbox):
        """

        :param int tracker_id:
        :param np.ndarray init_frame:
        :param np.ndarray init_bbox:
        :return:
        """

        self.tracker.initialize(tracker_id, init_frame, init_bbox)

    def update(self, tracker_id, frame):
        """

        :param int tracker_id:
        :param np.ndarray frame:
        :return:
        """

        bbox, score_, score_id_ = self.tracker.track(tracker_id, frame)

        return bbox, score_, score_id_

    def set_region(self, tracker_id, frame, bbox):
        self.tracker.set_region(tracker_id, frame, bbox)

    def copy(self):
        """
        :rtype: SiamX
        """
        obj = SiamX(self._params, self._n_trackers, self._update_location, self._logger, self)

        for _attr in self.members_to_deepcopy:
            self_attr = getattr(self, _attr)
            try:
                setattr(obj, _attr, copy.deepcopy(self_attr))
            except TypeError as e:
                raise TypeError(e)

        return obj

    def restore(self, obj, deep_copy=False):
        """
        :type obj: SiamX
        :type deep_copy: bool
        """
        for _attr in obj.members_to_deepcopy:
            obj_attr = getattr(obj, _attr)
            if deep_copy:
                setattr(self, _attr, copy.deepcopy(obj_attr))
            else:
                setattr(self, _attr, obj_attr)

    def close(self):
        pass
