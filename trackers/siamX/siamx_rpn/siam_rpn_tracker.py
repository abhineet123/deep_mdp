# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import sys
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from .siamx_rpn_utils import get_subwindow_tracking, cxy_wh_2_rect, rect_2_cxy_wh

# from gpu_profile import gpu_profile

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


# class SiamRPNPPTrackerParams(object):
#     score_size = 25  # for siamrpn++
#     lr = 0.295
#
#     help = {}
#
#     def update(self, cfg):
#         for k, v in cfg.items():
#             setattr(self, k, v)
#         # self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1 # for siamrpn


class SiamRPNTracker(object):
    """
    :type params: SiamRPNTracker.Params
    """

    class Params(object):
        # These are the default hyper-params for DaSiamRPN 0.3827
        windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
        # Params from the network architecture, have to be consistent with the training
        exemplar_size = 127  # input z size
        instance_size = 255  # input x size (search region)
        total_stride = 8
        context_amount = 0.5  # context amount for the exemplar
        ratios = [0.33, 0.5, 1, 2, 3]
        scales = [8, ]
        penalty_k = 0.055
        window_influence = 0.42
        lr = 0.295
        score_size = 0
        # adaptive change search region #
        adaptive = True

        store_on_cpu = 1

        help = {}

        def update(self, cfg):
            for k, v in cfg.items():
                setattr(self, k, v)

    def __init__(self, net, params, update_location, logger, parent=None):
        """

        :param net:
        :param int update_location:
        :param SiamRPNTracker.Params params:
        :param logging.RootLogger logger:
        :param SiamRPNTracker parent:
        """

        self._logger = logger
        self._params = params
        self._update_location = update_location
        self._net = net

        self._params.update(net.cfg)

        if parent is None:
            if self._params.store_on_cpu:
                self._logger.warning('Storing templates on cpu might result in slowdown')

        if self._params.score_size == 0:
            self.score_sz = int((self._params.instance_size - self._params.exemplar_size) /
                                self._params.total_stride + 1)
        else:
            self.score_sz = self._params.score_size

        self.anchor = generate_anchor(self._params.total_stride, self._params.scales, self._params.ratios,
                                      int(self.score_sz))

        if self._params.windowing == 'cosine':
            window = np.outer(np.hanning(self.score_sz), np.hanning(self.score_sz))
        elif self._params.windowing == 'uniform':
            window = np.ones((self.score_sz, self.score_sz))
        else:
            raise AssertionError(f'Invalid windowing type: {params.windowing}')

        self.anchor_num = len(self._params.ratios) * len(self._params.scales)

        window = np.tile(window.flatten(), self.anchor_num)
        self.window = window

        self.avg_chans = {}
        # self.score = {}
        self.score_id = {}
        self.bbox = {}
        self.target_pos = {}
        self.target_sz = {}
        self.im_h = {}
        self.im_w = {}
        self.template = {}
        self.instance_size = {}

        # sys.settrace(gpu_profile)

    def set_region(self, _id, frame, bbox):
        assert self._update_location, "set_region cannot be called if update_location is disabled"

        target_pos, target_sz = rect_2_cxy_wh(bbox)

        self.target_pos[_id] = target_pos
        self.target_sz[_id] = target_sz

    def initialize(self, _id, im, bbox):
        """

        :param im:
        :param bbox:
        """

        # bbox = convert_bbox_format(bbox, 'center-based')

        target_pos, target_sz = rect_2_cxy_wh(bbox)

        # state = dict()

        # if 'SiamRPNPP' in net_name:
        #     p = TrackerConfig_SiamRPNPP()
        # else:
        #     p = TrackerConfig()

        # state['im_h'] = im.shape[0]
        # state['im_w'] = im.shape[1]

        im_h = im.shape[0]
        im_w = im.shape[1]

        if self._params.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(im_h * im_w)) < 0.004:
                instance_size = 287  # small object big search region
            else:
                instance_size = 255
        else:
            instance_size = self._params.instance_size

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + self._params.context_amount * sum(target_sz)
        hc_z = target_sz[1] + self._params.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))
        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, self._params.exemplar_size, s_z, avg_chans, out_mode='np')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        z = transform(z_crop).unsqueeze(0).cuda()

        _net = self._net.cuda()

        template = _net.temple(z)

        self.template[_id] = template

        if self._params.store_on_cpu:
            # if self.cls1_kernel[_id] is not None:
            #     self.cls1_kernel[_id] = cls1_kernel.cpu()
            # if self.r1_kernel[_id] is not None:
            #     self.r1_kernel[_id] = r1_kernel.cpu()
            del z
            del _net


        self.avg_chans[_id] = avg_chans
        self.bbox[_id] = bbox
        self.target_pos[_id] = target_pos
        self.target_sz[_id] = target_sz
        self.im_h[_id] = im_h
        self.im_w[_id] = im_w
        self.instance_size[_id] = instance_size

        pass

    def track(self, _id, im):
        params = self._params
        net = self._net.cuda()
        avg_chans = self.avg_chans[_id]
        window = self.window
        target_pos = self.target_pos[_id]
        target_sz = self.target_sz[_id]
        im_w = self.im_w[_id]
        im_h = self.im_h[_id]
        instance_size = self.instance_size[_id]

        template = self.template[_id]

        # if self._params.store_on_cpu:
        #     r1_kernel = r1_kernel.cuda()
        #     cls1_kernel = cls1_kernel.cuda()

        # params = state['params']
        # net = state['net']
        # avg_chans = state['avg_chans']
        # window = state['window']

        # target_pos = state['target_pos']
        # target_sz = state['target_sz']

        wc_z = target_sz[1] + params.context_amount * sum(target_sz)
        hc_z = target_sz[0] + params.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = params.exemplar_size / s_z
        d_search = (instance_size - params.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        x_crop = get_subwindow_tracking(im, target_pos, instance_size, round(s_x), avg_chans, out_mode='np')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        x_crop = transform(x_crop).unsqueeze(0).cuda()

        delta, score = net(template, x_crop)

        target_pos, target_sz, score, score_with_penalty, score_id = tracker_eval(
            delta, score, target_pos, target_sz * scale_z, window, scale_z, self.anchor,
            params.penalty_k, params.window_influence, params.lr)

        score_map = np.reshape(score, (-1, self.score_sz, self.score_sz))
        pscore_map = np.reshape(score_with_penalty, (-1, self.score_sz, self.score_sz))
        # delta_map = np.reshape(delta, (-1, self.score_sz, self.score_sz))

        unravel_id = np.unravel_index(score_id, score_map.shape)
        best_pscore_map = pscore_map[unravel_id[0], :, :].squeeze()

        target_pos[0] = max(0, min(im_w, target_pos[0]))
        target_pos[1] = max(0, min(im_h, target_pos[1]))
        target_sz[0] = max(10, min(im_w, target_sz[0]))
        target_sz[1] = max(10, min(im_h, target_sz[1]))

        bbox = cxy_wh_2_rect(target_pos, target_sz)

        if self._update_location:
            self.target_pos[_id] = target_pos
            self.target_sz[_id] = target_sz

        # self.score[_id] = score
        self.score_id[_id] = score_id
        self.bbox[_id] = bbox

        if self._params.store_on_cpu:
            del x_crop
            del net

        # gpu_profile(frame=sys._getframe(), event='line', arg=None)

        return bbox, best_pscore_map, unravel_id[1:]


def tracker_eval(delta, _score, target_pos, target_sz, window, scale_z, anchor, penalty_k, window_influence, lr):
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()

    score = F.softmax(_score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()
    delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
    delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - window_influence) + window * window_influence
    best_pscore_id = np.argmax(pscore)

    # best_pscore_id_ur = np.unravel_index(best_pscore_id, pscore.shape)
    # print('###################### {}'.format(best_pscore_id))

    target = delta[:, best_pscore_id] / scale_z
    target_sz = target_sz / scale_z
    lr = penalty[best_pscore_id] * score[best_pscore_id] * lr

    res_x = target[0] + target_pos[0]
    res_y = target[1] + target_pos[1]

    res_w = target_sz[0] * (1 - lr) + target[2] * lr
    res_h = target_sz[1] * (1 - lr) + target[3] * lr

    target_pos = np.array([res_x, res_y])
    target_sz = np.array([res_w, res_h])

    return target_pos, target_sz, pscore, score, best_pscore_id
