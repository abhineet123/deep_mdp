import sys

sys.path.append('../')
import numpy as np
import torch
import collections
from .siamx_siamese import SiameseNet
from ..siamx_models.builder import SiamFC, SiamVGG, SiamFCRes22, SiamFCIncep22, SiamFCNext22


# Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

def load_net(fname, net):
    # pretrained_dict = torch.load(fname)['state_dict']
    pretrained_dict = torch.load(fname)
    pretrained_dict = {'model.' + key: value for key, value in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict)


def convert_bbox_format(bbox, to='center-based'):
    x, y, target_width, target_height = bbox
    if to == 'top-left-based':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center-based':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return [x * 1.0, y * 1.0, target_width * 1.0, target_height * 1.0]


def get_center(x):
    return (x - 1.) / 2.


class SiamFCTracker(object):
    """
    :type _net: SiameseNet
    """

    class Params(object):

        class Hyper:
            """
            :param int response_up:
            :param float scale_lr:
            :param int scale_max:
            :param float scale_min:
            :param int scale_num:
            :param float scale_penalty:
            :param float scale_step:
            :param float window_influence:
            :param float z_lr:
            """

            def __init__(self):
                self.cfg = ('',)
                self.response_up = 8
                self.scale_lr = 0.59
                self.scale_max = 5
                self.scale_min = 0.2
                self.scale_num = 3
                self.scale_penalty = 0.97
                self.scale_step = 1.04
                self.window_influence = 0.25
                self.z_lr = -0.01
                self.help = {
                    'response_up': '',
                    'scale_lr': '',
                    'scale_max': '',
                    'scale_min': '',
                    'scale_num': '',
                    'scale_penalty': '',
                    'scale_step': '',
                    'window_influence': '',
                    'z_lr': '',
                }

        class Design:
            """
            :param float context:
            :param int exemplar_sz:
            :param str join_method:
            :param str net:
            :param str net_gray:
            :param bool pad_with_image_mean:
            :param int score_sz:
            :param int search_sz:
            :param int tot_stride:
            :param str windowing:
            """

            def __init__(self):
                self.cfg = ('',)
                self.context = 0.5
                self.exemplar_sz = 127
                self.final_score_sz = 273
                self.join_method = 'xcorr'
                self.net = 'baseline-conv5_e55.mat'
                self.net_gray = ''
                self.pad_with_image_mean = True
                self.score_sz = 33
                self.search_sz = 255
                self.tot_stride = 4
                self.windowing = 'cosine_sum'
                self.help = {
                    'context': '',
                    'exemplar_sz': '',
                    'join_method': '',
                    'net': '',
                    'net_gray': '',
                    'pad_with_image_mean': '',
                    'score_sz': '',
                    'search_sz': '',
                    'tot_stride': '',
                    'windowing': '',
                }

        def __init__(self):
            self.hp = SiamFCTracker.Params.Hyper()
            self.design = SiamFCTracker.Params.Design()

    def __init__(self, net, params, update_location, logger, parent=None):
        """

        :param SiamFC | SiamVGG | SiamFCRes22 | SiamFCIncep22 | SiamFCNext22 net:
        :param SiamFCTracker.Params params:
        :param logger:
        """

        self._net = net
        self._params = params
        self._logger = logger
        self._update_location = update_location

        self.hp, self.design = self._params.hp, self._params.design

        # if self.siam is None:
        #     # init network
        #     self.siam = SiameseNet(tracker_name)
        #     if tracker_name == 'SiamFC':
        #         pretrained = 'cp/SiamFC.pth'
        #     elif tracker_name == 'SiamVGG':
        #         pretrained = None
        #     elif tracker_name == 'SiamFCRes22':
        #         pretrained = 'cp/temp/SiamFCRes22_400.pth'
        #     elif tracker_name == 'SiamFCIncep22':
        #         pretrained = None
        #     elif tracker_name == 'SiamFCNext22':
        #         pretrained = './cp/siamresnextcheckpoint.pth.tar'
        #
        #     load_net(pretrained, self.siam)
        #     self.siam.cuda()

        if parent is not None:
            self.scale_factors = parent.scale_factors
            self.penalty = parent.penalty
            self.score_sz = parent.score_sz
        else:
            self.score_sz = self.design.final_score_sz
            # init scale factor, penalty
            self.scale_factors = self.hp.scale_step ** np.linspace(-np.ceil(self.hp.scale_num / 2),
                                                                   np.ceil(self.hp.scale_num / 2), self.hp.scale_num)
            hann_1d = np.expand_dims(np.hanning(self.score_sz), axis=0)
            self.penalty = np.transpose(hann_1d) * hann_1d
            self.penalty = self.penalty / np.sum(self.penalty)

        self.x_sz = {}
        self.z_sz = {}
        self.pos_x = {}
        self.pos_y = {}
        self.target_w = {}
        self.target_h = {}
        self.templates_z_ = {}

    def set_region(self, _id, frame, region):
        assert self._update_location, "set_region cannot be called if update_location is disabled"

        bbox = convert_bbox_format(region, 'center-based')

        pos_x, pos_y, target_w, target_h = bbox

        self.pos_x[_id], self.pos_y[_id], self.target_w[_id], self.target_h[_id] = pos_x, pos_y, target_w, target_h

    def initialize(self, _id, frame, region):
        # init bbox
        bbox = convert_bbox_format(region, 'center-based')

        pos_x, pos_y, target_w, target_h = bbox

        context = self.design.context * (target_w + target_h)
        z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        x_sz = float(self.design.search_sz) / self.design.exemplar_sz * z_sz

        _, templates_z_ = self._net.cuda().get_template_z_new(pos_x, pos_y, z_sz, frame)

        self.x_sz[_id] = x_sz
        self.z_sz[_id] = z_sz
        self.pos_x[_id] = pos_x
        self.pos_y[_id] = pos_y
        self.target_w[_id] = target_w
        self.target_h[_id] = target_h
        self.templates_z_[_id] = templates_z_

    def track(self, _id, imagefile):
        x_sz = self.x_sz[_id]
        pos_x = self.pos_x[_id]
        pos_y = self.pos_y[_id]
        target_w = self.target_w[_id]
        target_h = self.target_h[_id]
        templates_z_ = self.templates_z_[_id]

        scaled_search_area = x_sz * self.scale_factors
        scaled_target_w = target_w * self.scale_factors
        scaled_target_h = target_h * self.scale_factors
        image_, scores_ = self._net.cuda().get_scores_new(pos_x, pos_y, scaled_search_area, templates_z_,
                                                          imagefile, self.design, self.score_sz)

        scores_ = np.squeeze(scores_)
        scores_[0, :, :] = self.hp.scale_penalty * scores_[0, :, :]
        scores_[2, :, :] = self.hp.scale_penalty * scores_[2, :, :]
        new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))

        x_sz = (1 - self.hp.scale_lr) * x_sz + self.hp.scale_lr * scaled_search_area[new_scale_id]
        target_w = (1 - self.hp.scale_lr) * target_w + self.hp.scale_lr * scaled_target_w[new_scale_id]
        target_h = (1 - self.hp.scale_lr) * target_h + self.hp.scale_lr * scaled_target_h[new_scale_id]
        score_ = scores_[new_scale_id, :, :]
        score_ = score_ - np.min(score_)
        score_ = score_ / np.sum(score_)
        score_ = (1 - self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty
        score_id_, pos_x, pos_y = _update_target_position(pos_x, pos_y, score_, self.score_sz,
                                                          self.design.tot_stride, self.design.search_sz,
                                                          self.hp.response_up, x_sz)

        bbox = [pos_x, pos_y, target_w, target_h]
        bbox = convert_bbox_format(bbox, 'top-left-based')

        if self._update_location:
            self.pos_x[_id], self.pos_y[_id], self.target_w[_id], self.target_h[_id] = pos_x, pos_y, target_w, target_h
            self.x_sz[_id] = x_sz

        return bbox, score_, score_id_


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    # score = score[32:241,32:241]
    # final_score_sz = final_score_sz - 64
    score_id_ = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = score_id_ - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return score_id_, pos_x, pos_y
