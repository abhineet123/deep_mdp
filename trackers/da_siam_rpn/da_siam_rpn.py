import os
import sys
import time
import math
import inspect
import copy

import numpy as np
import cv2
import torch
# from torch.autograd import Variable
import torch.nn.functional as F

from .da_siam_rpn_net import SiamRPNvot, SiamRPNBIG, SiamRPNotb, SiamRPN
from .run_da_siam_rpn import generate_anchor, tracker_eval
from .da_siam_rpn_utils import get_subwindow_tracking

from utilities import ids_to_member_names, get_unique_ids

class DaSiamRPN:
    """
    :type params: DaSiamRPN.Params
    :type logger: logging.RootLogger
    :type states: list[dict]
    :type net: SiamRPN
    """

    class Params:
        """

        :param int model: 0: SiamRPNvot 1: SiamRPNBIG 2: SiamRPNotb,
        :param str windowing: to penalize large displacements [cosine/uniform]
        :param int exemplar_size: input z size
        :param int instance_size: input x size (search region)
        :param float context_amount: context amount for the exemplar
        :param bool adaptive: adaptive change search region
        :param int score_size: size of score map
        :param int anchor_num: number of anchors
        """

        def __init__(self):
            self.windowing = 'cosine'
            self.exemplar_size = 127
            self.instance_size = 271
            self.total_stride = 8
            self.context_amount = 0.5
            self.ratios = (0.33, 0.5, 1, 2, 3)
            self.scales = (8,)

            self.penalty_k = 0.055
            self.window_influence = 0.42
            self.lr = 0.295
            self.adaptive = 0
            self.visualize = 0

            self.anchor_num = len(self.ratios) * len(self.scales)
            self.score_size = int((self.instance_size - self.exemplar_size) / self.total_stride + 1)

            self.model = 0
            self.pretrained_wts_dir = 'trackers/da_siam_rpn/pretrained_weights'

            self.help = {
            }

        def update(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)
            self.score_size = int((self.instance_size - self.exemplar_size) / self.total_stride + 1)
            self.anchor_num = len(self.ratios) * len(self.scales)

    def __init__(self, params, n_trackers, update_location, logger, parent=None):
        """
        :type params: DaSiamRPN.Params
        :type n_trackers: int
        :type logger: logging.RootLogger
        :type parent: DaSiamRPN | None
        :rtype: None
        """

        # self.tf_graph = tf.Graph()
        # avoid printing TF debugging information

        self._params = params
        self._logger = logger
        self._update_location = update_location
        self._n_trackers = n_trackers

        if parent is not None:
            # members = [k for k in dir(siam_fc) if not callable(getattr(siam_fc, k)) and not k.startswith('__')]
            self._members_to_spawn = parent._members_to_spawn
            for _member in self._members_to_spawn:
                setattr(self, _member, getattr(parent, _member))
        else:

            spawn_ids = []
            spawn_ids_gen = get_unique_ids(spawn_ids)

            self.net = next(spawn_ids_gen)
            self.score_sz = next(spawn_ids_gen)
            self._members_to_spawn = next(spawn_ids_gen)
            self._members_to_copy = next(spawn_ids_gen)

            self._members_to_spawn = ids_to_member_names(self, spawn_ids)

            self._members_to_copy = (
            )

            models = {
                0: (SiamRPNvot, 'SiamRPNVOT.model'),
                1: (SiamRPNBIG, 'SiamRPNBIG.model'),
                2: (SiamRPNotb, 'SiamRPNOTB.model')
            }
            try:
                net_type, pretrained_wts_name = models[self._params.model]
            except KeyError:
                raise AssertionError('Invalid model_type: {}'.format(self._params.model))
            else:
                net = net_type()  # type: SiamRPN
                state_dict = torch.load(os.path.join(self._params.pretrained_wts_dir, pretrained_wts_name))
                net.load_state_dict(state_dict)

            # if self.params.model == 0:
            #     net = SiamRPNvot()
            #     net.load_state_dict(torch.load(os.path.join(self.params.pretrained_wts_dir, 'SiamRPNVOT.model')))
            #     self.logger.info('Using SiamRPNVOT model')
            # elif self.params.model == 1:
            #     net = SiamRPNBIG()
            #     net.load_state_dict(torch.load(os.path.join(self.params.pretrained_wts_dir, 'SiamRPNBIG.model')))
            #     self.logger.info('Using SiamRPNBIG model')
            # elif self.params.model == 2:
            #     net = SiamRPNotb()
            #     net.load_state_dict(torch.load(os.path.join(self.params.pretrained_wts_dir, 'SiamRPNOTB.model')))
            #     self.logger.info('Using SiamRPNOTB model')
            # else:
            #     raise IOError('Invalid model_type: {}'.format(self.params.model))

            net.eval().cuda()

            self.net = net  # type: SiamRPN

            self._params.update(self.net.cfg)

            self.score_sz = self._params.score_size

        self.states = [None] * n_trackers
        self.anchor = []

    def initialize(self, tracker_id, init_frame, init_bbox):
        """

        :param int tracker_id:
        :param np.ndarray init_frame:
        :param np.ndarray init_bbox:
        :return:
        """

        # need a different net for each instance
        # net = copy.deepcopy(self.net)  # type: SiamRPN

        xmin, ymin, target_w, target_h = init_bbox
        cx = xmin + target_w / 2.0
        cy = ymin + target_h / 2.0

        target_pos = np.array([cx, cy])
        target_sz = np.array([target_w, target_h])

        state = dict()

        state['im_h'] = init_frame.shape[0]
        state['im_w'] = init_frame.shape[1]

        if self._params.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                self._params.instance_size = 287  # small object big search region
            else:
                self._params.instance_size = 271

            self._params.score_size = (
                                              self._params.instance_size - self._params.exemplar_size) / self._params.total_stride + 1

        self.anchor = generate_anchor(self._params.total_stride, self._params.scales, self._params.ratios,
                                      int(self._params.score_size))

        avg_chans = np.mean(init_frame, axis=(0, 1))

        wc_z = target_sz[0] + self._params.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self._params.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(init_frame, target_pos, self._params.exemplar_size, s_z, avg_chans)

        z = z_crop.unsqueeze(0)
        self.net.temple(tracker_id, z.cuda())

        if self._params.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_sz), np.hanning(self.score_sz))
        elif self._params.windowing == 'uniform':
            window = np.ones((self.score_sz, self.score_sz))
        else:
            raise IOError('Invalid windowing type: {}'.format(self._params.windowing))
        window = np.tile(window.flatten(), self._params.anchor_num)

        # state['p'] = self.params
        # state['net'] = net

        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        self.states[tracker_id] = state

    def update(self, tracker_id, frame):
        state = self.states[tracker_id]

        # p = state['p']
        # net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + self._params.context_amount * sum(target_sz)
        hc_z = target_sz[0] + self._params.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = self._params.exemplar_size / s_z
        d_search = (self._params.instance_size - self._params.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = get_subwindow_tracking(frame, target_pos, self._params.instance_size,
                                        round(s_x), avg_chans).unsqueeze(0)

        target_pos, target_sz, score_with_penalty, score, delta, score_id = tracker_eval(
            tracker_id, self.net, x_crop.cuda(), target_pos, target_sz * scale_z, window,
            scale_z, self._params, self.anchor)

        score_map = np.reshape(score, (-1, self.score_sz, self.score_sz))
        pscore_map = np.reshape(score_with_penalty, (-1, self.score_sz, self.score_sz))
        # delta_map = np.reshape(delta, (-1, self.score_sz, self.score_sz))

        unravel_id = np.unravel_index(score_id, score_map.shape)
        best_pscore_map = pscore_map[unravel_id[0], :, :].squeeze()

        # best_pscore_map_max_idx = np.argmax(best_pscore_map)
        # best_pscore_map_max_idx_ur = np.unravel_index(best_pscore_map_max_idx, best_pscore_map.shape)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        if self._update_location:
            state['target_pos'] = target_pos
            state['target_sz'] = target_sz

        state['score'] = score
        state['score_with_penalty'] = score_with_penalty

        pos_x, pos_y = target_pos
        target_w, target_h = target_sz

        bbox = [pos_x - target_w / 2,
                pos_y - target_h / 2,
                target_w, target_h]
        return bbox, best_pscore_map, unravel_id[1:]

    def set_region(self, tracker_id, frame, bbox):

        assert self._update_location, "set_region cannot be called if update_location is disabled"

        state = self.states[tracker_id]

        xmin, ymin, target_w, target_h = bbox
        cx = xmin + target_w / 2.0
        cy = ymin + target_h / 2.0

        target_pos = np.array([cx, cy])
        target_sz = np.array([target_w, target_h])

        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

    def close(self):
        pass
