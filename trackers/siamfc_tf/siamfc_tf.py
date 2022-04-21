import os

import sys
# import csv
import numpy as np
# from PIL import Image
import copy
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append('.')

# try:
#     import src.siamese as siam
#     from src.parse_arguments import parse_arguments
#     from src.region_to_bbox import region_to_bbox
# except ImportError:

import trackers.siamfc_tf.src.siamese as siam
from trackers.siamfc_tf.src.siamese import DesignParams, EnvironmentParams, HyperParams


# from trackers.siamfc_tf.src import parse_arguments
# from trackers.siamfc_tf.src import region_to_bbox


# import matplotlib.pyplot as plt


# from src.visualization import show_frame, show_crops, show_scores

# gpu_device = 2
# os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)

class SiamFCTF:
    """
    :type params: SiamFCTF.Params
    :type logger: logging.RootLogger
    :type pos_x: list[float]
    :type pos_y: list[float]
    :type pos_y: list[float]
    :type target_w: list[int]
    :type target_h: list[int]
    :type z_sz: list[float]
    :type x_sz: list[float]
    :type templates_z_: list[np.ndarray]
    :type new_templates_z_: list[np.ndarray]
    :type hp: HyperParams
    :type design: DesignParams
    """

    class Params:
        def __init__(self):
            self.allow_gpu_memory_growth = 1
            self.per_process_gpu_memory_fraction = 1.0

            # self.gpu = -1
            self.vis = 0
            self.score_type = 1
            self.norm_score = 1

            self.design = DesignParams()
            self.env = EnvironmentParams()
            self.hp = HyperParams()

            self.help = {
                'score_type': 'score map to return for feature extraction: '
                              '0: raw 33x33 score map '
                              '1: raw 33x33 score map with post processing '
                              '2: resized 257x257 score map with post processing',
                'update_location': 'update template location from tracked result in each frame',
                'vis': 'Enable diagnostic visualization',
                'design': 'DesignParams',
                'env': 'EnvironmentParams',
                'hp': 'HyperParams',
            }

    def __init__(self, params, n_trackers, update_location, logger, parent=None):
        """
        :type params: SiamFCTF.Params
        :type n_trackers: int
        :type update_location: int
        :type logger: logging.RootLogger
        :type parent: SiamFCTF | None
        :rtype: None
        """

        # self.tf_graph = tf.Graph()
        # avoid printing TF debugging information

        self._logger = logger
        self._update_location = update_location

        self.members_to_spawn = (
            'tf_graph',
            'tf_sess',

            'image',
            'templates_z',
            'raw_scores',
            'scores',

            '_params',

            'final_score_sz',

            'pos_x_ph',
            'pos_y_ph',
            'z_sz_ph',
            'x_sz0_ph',
            'x_sz1_ph',
            'x_sz2_ph',

            'hp',
            'design',
            'score_sz',
            'scale_factors',
            'raw_hann_1d',
            'raw_penalty',
            'hann_1d',
            'penalty',
            'n_trackers',
        )
        self._members_to_copy = (
            'pos_x',
            'pos_y',
            'target_w',
            'target_h',
            'z_sz',
            'x_sz',
            'templates_z_',
            'new_templates_z_',
        )

        if parent is not None:
            # members = [k for k in dir(siam_fc) if not callable(getattr(siam_fc, k)) and not k.startswith('__')]
            for _member in self.members_to_spawn:
                setattr(self, _member, getattr(parent, _member))
        else:
            self._params = params
            self.design = self._params.design
            self.env = self._params.env
            self.hp = self._params.hp

            # if self.params.gpu >= 0:
            #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            #     os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(self.params.gpu)

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
            tf.autograph.set_verbosity(3)

            print('Using Tensorflow ' + tf.__version__)

            # session_config = tf.ConfigProto(allow_soft_placement=True,
            #                                 log_device_placement=False)

            session_config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                      log_device_placement=False)

            session_config.gpu_options.allow_growth = self._params.allow_gpu_memory_growth
            session_config.gpu_options.per_process_gpu_memory_fraction = self._params.per_process_gpu_memory_fraction

            self.tf_graph = tf.Graph()
            self.tf_sess = tf.compat.v1.Session(graph=self.tf_graph, config=session_config)

            # hp, _, _, env, design = parse_arguments()

            # Set size for use with tf.image.resize_images with align_corners=True.
            # For example,
            #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
            # instead of
            # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
            self.final_score_sz = self.hp.response_up * (self.design.score_sz - 1) + 1
            # self.tf_sess = tf.Session()

            # with self.tf_graph.as_default():

            # self.image = []
            # self.templates_z = []
            # self.raw_scores = []
            # self.scores = []
            #
            # self.pos_x_ph = []
            # self.pos_y_ph = []
            # self.z_sz_ph = []
            # self.x_sz0_ph = []
            # self.x_sz1_ph = []
            # self.x_sz2_ph = []

            # build TF graph once for all
            with self.tf_graph.as_default():
                # for tracker_id in range(n_trackers):
                image, templates_z, raw_scores, scores, pos, sizes = siam.build_tracking_graph2(
                    self.final_score_sz, self.design, self.env)

                pos_x_ph, pos_y_ph = pos
                z_sz_ph, x_sz0_ph, x_sz1_ph, x_sz2_ph = sizes

                # self.image.append(image)
                # self.templates_z.append(templates_z)
                # self.raw_scores.append(raw_scores)
                # self.scores.append(scores)
                #
                # self.pos_x_ph.append(pos_x_ph)
                # self.pos_y_ph.append(pos_y_ph)
                # self.z_sz_ph.append(z_sz_ph)
                # self.x_sz0_ph.append(x_sz0_ph)
                # self.x_sz1_ph.append(x_sz1_ph)
                # self.x_sz2_ph.append(x_sz2_ph)

            self.image = image
            self.templates_z = templates_z
            self.raw_scores = raw_scores
            self.scores = scores

            self.pos_x_ph = pos_x_ph
            self.pos_y_ph = pos_y_ph
            self.z_sz_ph = z_sz_ph
            self.x_sz0_ph = x_sz0_ph
            self.x_sz1_ph = x_sz1_ph
            self.x_sz2_ph = x_sz2_ph

            # with self.tf_graph.as_default():
            #     tf.variables_initializer(tf.get_collection(tf.GraphKeys.VARIABLES)).run(session=self.tf_sess)

            # self.start_frame = start_frame

            # self.coord = None
            # self.threads = None

            self.scale_factors = self.hp.scale_step ** np.linspace(-np.ceil(self.hp.scale_num / 2),
                                                                   np.ceil(self.hp.scale_num / 2),
                                                                   self.hp.scale_num)
            # cosine window to penalize large displacements
            self.raw_hann_1d = np.expand_dims(np.hanning(self.design.score_sz), axis=0)
            self.raw_penalty = np.transpose(self.raw_hann_1d) * self.raw_hann_1d
            self.raw_penalty = self.raw_penalty / np.sum(self.raw_penalty)

            self.hann_1d = np.expand_dims(np.hanning(self.final_score_sz), axis=0)
            self.penalty = np.transpose(self.hann_1d) * self.hann_1d
            self.penalty = self.penalty / np.sum(self.penalty)

            # run_metadata = tf.RunMetadata()
            # run_opts = {
            #     'options': tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            #     'run_metadata': run_metadata,
            # }

            # self.run_opts = {}

            self.n_trackers = n_trackers

            if self.hp.z_lr == 0:
                self._logger.info('Template updating is disabled')

            if self._params.score_type < 2:
                self.score_sz = self.design.score_sz
                self._logger.info('Returning raw scores for feature extraction')
                if self._params.score_type == 1:
                    self._logger.info('score map post processing is enabled')
                #     self.logger.info('min and sum normalization is enabled')
                # if self.params.score_type >= 2:
                #     self.logger.info('scale penalty is enabled')
                # if self.params.score_type >= 3:
                #     self.logger.info('displacement penalty is enabled')
            elif self._params.score_type == 2:
                self.score_sz = self.final_score_sz
                self._logger.info('Returning resized and post processed score map for feature extraction')
            else:
                raise IOError('Invalid score_type: {}'.format(self._params.score_type))

            # with self.tf_graph.as_default():
            # with self.tf_sess as sess:
            with self.tf_graph.as_default():
                tf.compat.v1.global_variables_initializer().run(session=self.tf_sess)

            # Coordinate the loading of image files.
            # self.coord = tf.train.Coordinator()
            # self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.tf_sess)

        self.pos_x = [None] * n_trackers
        self.pos_y = [None] * n_trackers
        self.target_w = [None] * n_trackers
        self.target_h = [None] * n_trackers

        self.z_sz = [None] * n_trackers
        self.x_sz = [None] * n_trackers

        self.templates_z_ = [None] * n_trackers
        self.new_templates_z_ = [None] * n_trackers
        self.templates = [None] * n_trackers

    def initialize(self, tracker_id, init_frame, init_bbox):
        """

        :param int tracker_id:
        :param np.ndarray init_frame:
        :param np.ndarray init_bbox:
        :return:
        """

        # if init_file_path == None:
        #     init_file_path = 'image_0.jpg'
        #     cv2.imwrite(init_file_path, init_frame)

        if self._params.vis:
            pt1 = (int(init_bbox[0]), int(init_bbox[1]))
            pt2 = (int(init_bbox[0] + init_bbox[2]),
                   int(init_bbox[1] + init_bbox[3]))
            init_frame_disp = np.copy(init_frame)

        self.templates[tracker_id] = (init_frame, init_bbox)

        init_frame = init_frame.astype(np.float32)

        xmin, ymin, target_w, target_h = init_bbox
        pos_x = xmin + target_w / 2.0
        pos_y = ymin + target_h / 2.0

        self.pos_x[tracker_id] = pos_x
        self.pos_y[tracker_id] = pos_y
        self.target_w[tracker_id] = target_w
        self.target_h[tracker_id] = target_h

        context = self.design.context * (target_w + target_h)
        self.z_sz[tracker_id] = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        self.x_sz[tracker_id] = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz[tracker_id]

        # thresholds to saturate patches shrinking/growing

        # min_z = self.hp.scale_min * self.z_sz[tracker_id]
        # max_z = self.hp.scale_max * self.z_sz[tracker_id]
        # min_x = self.hp.scale_min * self.x_sz[tracker_id]
        # max_x = self.hp.scale_max * self.x_sz[tracker_id]

        # bbox = np.zeros((1, 4))

        # save first frame position (from ground-truth)
        # bbox[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h

        templates_z_ = self.tf_sess.run([self.templates_z], feed_dict={
            self.pos_x_ph: self.pos_x[tracker_id],
            self.pos_y_ph: self.pos_y[tracker_id],
            self.z_sz_ph: self.z_sz[tracker_id],
            self.image: init_frame})

        self.templates_z_[tracker_id] = templates_z_
        self.new_templates_z_[tracker_id] = templates_z_

        if self._params.vis:
            cv2.rectangle(init_frame_disp, pt1, pt2, (0, 255, 0), thickness=2)
            cv2.imshow('init_frame {}'.format(tracker_id), init_frame_disp)
        # cv2.waitKey(0)

    def update(self, tracker_id, frame):
        # cx, cy, w, h = bbox
        # xmin = cx + w/2.0
        # ymin = cy + h/2.0

        frame = frame.astype(np.float32)

        # with self.tf_sess as sess:
        scaled_exemplar = self.z_sz[tracker_id] * self.scale_factors
        scaled_search_area = self.x_sz[tracker_id] * self.scale_factors
        scaled_target_w = self.target_w[tracker_id] * self.scale_factors
        scaled_target_h = self.target_h[tracker_id] * self.scale_factors

        # start_t = time.time()

        raw_scores_, scores_ = self.tf_sess.run((self.raw_scores, self.scores),
                                                feed_dict={
                                                    self.pos_x_ph: self.pos_x[tracker_id],
                                                    self.pos_y_ph: self.pos_y[tracker_id],
                                                    self.x_sz0_ph: scaled_search_area[0],
                                                    self.x_sz1_ph: scaled_search_area[1],
                                                    self.x_sz2_ph: scaled_search_area[2],
                                                    self.templates_z: np.squeeze(self.templates_z_[tracker_id]),
                                                    self.image: frame,
                                                },
                                                # **self.run_opts
                                                )
        # end_t = time.time()
        # tf_fps = 1.0 / (end_t - start_t)
        # self.logger.info('tracker {} :: tf_fps: {}'.format(tracker_id, tf_fps))

        scores_ = np.squeeze(scores_)
        # penalize change of scale
        scores_[0, :, :] = self.hp.scale_penalty * scores_[0, :, :]
        scores_[2, :, :] = self.hp.scale_penalty * scores_[2, :, :]
        # find scale with highest peak (after penalty)
        new_scale_id = np.argmax(np.amax(scores_, axis=(1, 2)))
        # update scaled sizes
        x_sz = (1 - self.hp.scale_lr) * self.x_sz[tracker_id] + self.hp.scale_lr * scaled_search_area[
            new_scale_id]
        target_w = (1 - self.hp.scale_lr) * self.target_w[tracker_id] + self.hp.scale_lr * \
                   scaled_target_w[new_scale_id]
        target_h = (1 - self.hp.scale_lr) * self.target_h[tracker_id] + self.hp.scale_lr * \
                   scaled_target_h[new_scale_id]
        # select response with new_scale_id
        score_ = scores_[new_scale_id, :, :]
        score_ = score_ - np.min(score_)
        score_ = score_ / np.sum(score_)

        # apply displacement penalty
        score_ = (1 - self.hp.window_influence) * score_ + self.hp.window_influence * self.penalty

        pos_x, pos_y, pos_idx = _update_target_position(
            self.pos_x[tracker_id], self.pos_y[tracker_id], score_, self.final_score_sz,
            self.design.tot_stride,
            self.design.search_sz, self.hp.response_up, x_sz)

        if self.hp.z_lr > 0:
            """update the target representation with a rolling average
            """
            self.new_templates_z_[tracker_id] = self.tf_sess.run(
                [self.templates_z], feed_dict={
                    self.pos_x_ph: pos_x,
                    self.pos_y_ph: pos_y,
                    self.z_sz_ph: self.z_sz[tracker_id],
                    self.image: frame
                })

            self.templates_z_[tracker_id] = (1 - self.hp.z_lr) * np.asarray(self.templates_z_[tracker_id]) + \
                                            self.hp.z_lr * np.asarray(
                self.new_templates_z_[tracker_id])

        # convert <cx,cy,w,h> to <x,y,w,h> and save output
        bbox = [pos_x - self.target_w[tracker_id] / 2,
                pos_y - self.target_h[tracker_id] / 2,
                self.target_w[tracker_id], self.target_h[tracker_id]]

        if self._update_location:
            self.pos_x[tracker_id], self.pos_y[tracker_id], self.target_w[tracker_id], self.target_h[tracker_id] = \
                pos_x, pos_y, target_w, target_h
            # update template patch size
            self.z_sz[tracker_id] = (1 - self.hp.scale_lr) * self.z_sz[tracker_id] + \
                                    self.hp.scale_lr * scaled_exemplar[new_scale_id]
            self.x_sz[tracker_id] = x_sz

        if self._params.score_type == 2:
            if self._params.norm_score:
                score_ = score_ / np.amax(score_)
            return bbox, score_, pos_idx
        else:
            raw_scores_ = np.squeeze(raw_scores_)
            """select response with new_scale_id"""
            raw_score_ = raw_scores_[new_scale_id, :, :]

            if self._params.score_type == 1:
                if new_scale_id != 1:
                    """penalize change of scale"""
                    raw_score_ = self.hp.scale_penalty * raw_score_
                """normalize"""
                raw_score_ = raw_score_ - np.min(raw_score_)
                raw_score_ = raw_score_ / np.sum(raw_score_)
                """apply displacement penalty"""
                raw_score_ = (1 - self.hp.window_influence) * raw_score_ + self.hp.window_influence * self.raw_penalty

            pos_idx = np.asarray(np.unravel_index(np.argmax(raw_score_), np.shape(raw_score_)))

            # """scale down pos_idx from resized score map to the original"""
            # x, y = pos_idx
            # x = int((x - 1) / self.hp.response_up + 1)
            # y = int((y - 1) / self.hp.response_up + 1)
            # pos_idx = [x, y]

            y, x = pos_idx
            if self._params.norm_score:
                raw_score_ = raw_score_ / raw_score_[y, x]

            return bbox, raw_score_, pos_idx

    def set_region(self, tracker_id, frame, bbox):

        assert self._update_location, "set_region cannot be called if update_location is disabled"

        xmin, ymin, target_w, target_h = bbox
        pos_x = xmin + target_w / 2.0
        pos_y = ymin + target_h / 2.0

        self.pos_x[tracker_id] = pos_x
        self.pos_y[tracker_id] = pos_y
        self.target_w[tracker_id] = target_w
        self.target_h[tracker_id] = target_h

        """not clear as to how z_sz and x_sz need to be changed so ignoring for now
        """
        # context = self.design.context * (target_w + target_h)
        # self.z_sz[tracker_id] = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        # self.x_sz[tracker_id] = float(self.design.search_sz) / self.design.exemplar_sz * self.z_sz[tracker_id]
        #
        # # update template patch size
        # self.z_sz[tracker_id] = (1 - self.hp.scale_lr) * self.z_sz[tracker_id] + \
        #                         self.hp.scale_lr * scaled_exemplar[new_scale_id]
        # self.x_sz[tracker_id] = x_sz

    def copy(self):
        """
        :rtype: dict
        """
        obj_dict = {}

        for _attr in self._members_to_copy:
            self_attr = getattr(self, _attr)
            try:
                obj_dict[_attr] = copy.deepcopy(self_attr)
            except TypeError as err:
                raise TypeError('Deep copying of attribute {} failed:'.format(_attr)) from err
        return obj_dict

    def restore(self, obj_dict, deep_copy=False):
        """
        :type obj_dict: dict
        :type deep_copy: bool
        """
        for _attr in self._members_to_copy:
            obj_attr = obj_dict[_attr]
            if deep_copy:
                setattr(self, _attr, copy.deepcopy(obj_attr))
            else:
                setattr(self, _attr, obj_attr)

    def close(self):
        # tf.reset_default_graph()
        self.tf_sess.close()


def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    pos_idx = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = pos_idx - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop * x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y, pos_idx
