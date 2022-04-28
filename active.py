import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
import os
import cv2

from input import Detections
from policy_base import PolicyBase
from models.model_base import ModelBase
from models.dummy import Oracle

from utilities import MDPStates, CustomLogger, AnnotationStatus, annotate_and_show, \
    draw_box, draw_boxes, get_patch, get_shifted_boxes, resize_ar


class Active(PolicyBase):
    """
    :type _params: Active.Params
    :type _model: ModelBase
    :type _feature_extractor: Active.FeatureExtractor

    """

    class Params(PolicyBase.Params):
        """


        :ivar overlap_pos: 'IOU threshold for deciding if a detection is considered as true positive '
                       'while training - iou with the maximally overlapping is taken; '
                       'this is also used as one of the conditions for deciding if a given annotation '
                       'can be used as the starting one for a training trajectory - its overlap or '
                       'IOU with the maximally overlapping detection must to be greater than this threshold '
                       'for it to be considered as a valid starting point for the trajectory',
        :ivar overlap_neg: 'IOU threshold for deciding if a detection is considered as '
                       'false negative while training - iou with the maximally overlapping etection is taken; '
                       'the detection is labeled as unknown if this iou is less than overlap_pos but greater '
                       'than overlap_neg',

        :ivar min_pos_ratio: 'minimum fraction of all decisions to be positive to deal with the rare case of '
                 'complete model failure where all decisions end up as negative due to '
                 'sequence specific factors like unusual camera angle compared to training set',

        :ivar feature_type: 0: heuristics on shape, size ad location of boxes,
        1: raw image patches corresponding to boxes
        """

        class ImagePatches:
            """
            :ivar size: size to reshape all patches to
            """

            def __init__(self):
                self.size = (64, 64)
                self.to_gs = 0
                self.vis = 0

        def __init__(self):
            PolicyBase.Params.__init__(self)

            """minimum overlap between a detection and an annotation in the same frame for the detection to be 
            considered a true positive
            """
            self.overlap_pos = 0.5
            """
            maximum overlap between a detection and an annotation in the same frame for the detection to be 
            considered a false positive
            """
            self.overlap_neg = 0.2

            self.min_pos_ratio = 0.1

            self.feature_type = 0
            self.img_patches = Active.Params.ImagePatches()

            self._pretrain_path = ''

        @property
        def pretrain_path(self): return self._pretrain_path

        @pretrain_path.setter
        def pretrain_path(self, val): self._pretrain_path = val

    class FeatureExtractor:
        def __init__(self):
            self.n_features = None
            self.feature_shape = None

        def get(self, detections, frames, valid_idx=None, labels=None):
            raise NotImplementedError()

        def vis(self, features, labels, gt_labels=None):
            raise NotImplementedError()

    class Heuristics(FeatureExtractor):
        def __init__(self):
            Active.FeatureExtractor.__init__(self)

            self.n_features = 6
            self.feature_shape = (6,)

        def get(self, det_data, frames, valid_idx=None, labels=None):
            h, w = frames[0].shape[:2]
            max_data = np.asarray((w, h, w, h, 1), dtype=np.float32).reshape((1, 5))
            valid_det_data = det_data[:, 2:7]
            if valid_idx is not None:
                valid_det_data = valid_det_data[valid_idx, :]
            n_valid_idx = valid_det_data.shape[0]

            train_features = np.concatenate((valid_det_data / max_data,
                                             np.ones((n_valid_idx, 1))), axis=1).astype(np.float32)
            return train_features

    class ImagePatches(FeatureExtractor):
        """
        :type _params: Active.Params.ImagePatches
        """

        def __init__(self, params, rgb_input):
            Active.FeatureExtractor.__init__(self)

            self._params = params
            self._rgb_input = rgb_input
            self.n_features = self._params.size[0] * self._params.size[1]
            self.feature_size = self.feature_shape = self._params.size
            if self._rgb_input and not self._params.to_gs:
                self.n_features *= 3
                self.feature_shape = tuple([3, ] + list(self.feature_shape))
                self._is_gs = 0
            else:
                self.feature_shape = tuple([1, ] + list(self.feature_shape))
                self._is_gs = 1

        def get(self, det_data, frames, valid_idx=None, labels=None):
            valid_det_data = det_data[:]
            if valid_idx is not None:
                valid_det_data = valid_det_data[valid_idx, :]
            n_valid = valid_det_data.shape[0]

            train_features = np.zeros((n_valid, self.n_features), dtype=np.float32)
            for det_idx in range(n_valid):
                frame_id = int(valid_det_data[det_idx, 0])
                frame = frames[frame_id]

                patch, resized_patch = get_patch(frame, valid_det_data[det_idx, 2:6],
                                                 self._params.to_gs, self.feature_size, context_ratio=0)

                resized_patch_feature = np.transpose(resized_patch, axes=[2, 0, 1]).astype(np.float32)
                train_features[det_idx, :] = resized_patch_feature.flatten() / 255.0

                if self._params.vis:
                    frame_disp = np.copy(frame)

                    if labels is not None:
                        col = 'green' if labels[det_idx] == 1 else 'red'
                    else:
                        col = 'blue'

                    draw_box(frame_disp, valid_det_data[det_idx, 2:6], color=col)

                    annotate_and_show('ImagePatches', [patch, resized_patch, frame_disp])

            return train_features

        def vis(self, features, labels, gt_labels=None):
            _h, _w = self.feature_shape[-2:]

            n_samples = labels.size
            concat_imgs = []

            for sample_id in range(n_samples):
                feature_map = features[sample_id, ...].reshape(self.feature_shape)

                if self._is_gs:
                    feature_map = feature_map.squeeze()
                    feature_map = np.stack((feature_map,) * 3, axis=2)
                else:
                    feature_map = np.transpose(feature_map, (1, 2, 0))

                concat_img = [feature_map, ]

                for _labels in (labels, gt_labels):

                    if _labels is None:
                        continue

                    label = _labels[sample_id]

                    if label < 0:
                        label = 0

                    if label == 1:
                        label_col = (0, 1, 0)
                    else:
                        label_col = (0, 0, 1)

                    label_img = np.full((feature_map.shape[0], 25, 3), label_col[0])
                    label_img[..., 1].fill(label_col[1])
                    label_img[..., 2].fill(label_col[2])

                    """5 pixel vertical border"""
                    border_img = np.zeros((label_img.shape[0], 5, 3), dtype=label_img.dtype)

                    concat_img += [border_img, label_img]

                if len(concat_img) > 1:
                    concat_img = np.concatenate(concat_img, axis=1)
                else:
                    concat_img = concat_img[0]

                sample_border_img = np.zeros((5, concat_img.shape[1], 3), dtype=label_img.dtype)

                concat_imgs.append(concat_img)
                concat_imgs.append(sample_border_img)

            concat_imgs = np.concatenate(concat_imgs, axis=0)

            concat_imgs = resize_ar(concat_imgs, width=1800, height=800)

            return concat_imgs

    def __init__(self, params, rgb_input, logger, parent=None):
        """
        :type params: Active.Params
        :type logger: logging.RootLogger | CustomLogger
        :type parent: Active | None
        :rtype: None
        """
        PolicyBase.__init__(self, params, 'active', logger, parent)

        self._params = params
        self._logger = logger
        self._rgb_input = rgb_input

        self._target_id = -1

        if parent is not None:
            # self.max_data = parent.max_data
            # self._train_features = active._train_features
            # self._train_labels = active._train_labels
            self._model = parent._model
            self._feature_extractor = parent._feature_extractor
            self._n_features = parent._n_features
            self._feature_shape = parent._feature_shape
            self.is_pretrained = parent.is_pretrained
        else:
            # self.max_data = None
            # self._train_features = None
            # self._train_labels = None
            self.is_test = 0

            if self._params.feature_type == 0:
                self._feature_extractor = Active.Heuristics()
            elif self._params.feature_type == 1:
                self._feature_extractor = Active.ImagePatches(self._params.img_patches, self._rgb_input)
            else:
                raise AssertionError('Invalid feature_type: {}'.format(self._params.feature_type))

            self._n_features = self._feature_extractor.n_features
            self._feature_shape = self._feature_extractor.feature_shape

            self._create_model(self._feature_shape, self._logger)

            self.is_pretrained = 0

            self.pretrained_load = self._params.pretrain_path

            if self.pretrained_load:
                self._logger.info('loading pretrained model from: {}'.format(
                    self.pretrained_load))
                if not self._model.load(self.pretrained_load, load_samples=False):
                    raise IOError('Model loading failed')
                self.is_pretrained = 1

            if not self._params.enable_stats:
                self._logger.warning('stats are disabled')

        self._features = None
        self.state = MDPStates.inactive

        self.batch_save_path = ''
        # self._ann_status = None
        self._gt_label = None

    def train(self, frames, detections):
        """
        asynchronous training

        :param list[np.ndarray] | tuple(np.ndarray) frames:
        :param Detections detections:
        :param np.ndarray ann_idx: index of the annotation belonging to the current trajectory
        :return: None
        """
        if self._model is None:
            self._logger.warning('No model to train')
            return True

        if self.is_pretrained:
            self._logger.warning('skipping training of pre-trained model')
            return True

        """all detections are false positives by default"""
        labels = np.full((detections.count, 1), -1)
        """true positive detections - does not seem to account for highly-overlapping duplicate 
        detections that should be false positives """
        tp_bool = detections.max_cross_iou > self._params.overlap_pos
        labels[np.flatnonzero(tp_bool)] = 1
        """unknown whether true positive or false positive detection"""
        unknown_ids = np.flatnonzero(np.logical_and(detections.max_cross_iou >= self._params.overlap_neg,
                                                    np.logical_not(tp_bool)))
        labels[unknown_ids] = 0

        """don't train on unknown detections"""
        valid_idx = np.flatnonzero(labels != 0)
        n_valid_idx = valid_idx.size
        if n_valid_idx == 0:
            raise AssertionError('No valid detections found')

        train_labels = labels[valid_idx].squeeze().astype(np.float32)

        train_features = self._feature_extractor.get(detections.data, frames, valid_idx, train_labels)

        self._model.train(train_labels, train_features)

        detections.labels = labels

    def batch_train(self):
        if self.is_pretrained:
            self._logger.warning('skipping batch training of pre-trained model')
            return

        save_path = self._model.batch_train()
        self.batch_save_path = save_path

    def initialize(self, target_id):
        """

        :param int target_id:
        :param str ann_status:
        :return:
        """

        self._target_id = target_id
        self._model.set_id(self._target_id)

    def _get_label(self, ann_status):
        assert ann_status in AnnotationStatus.types, f'Invalid ann_status: {ann_status}'

        _gt_label = np.array((-1,), dtype=np.int32)

        if self._oracle_type == Oracle.Types.absolute:
            """absolute oracle: new TP vs (FP or old TP)"""
            if ann_status in ('fp_deleted', 'fp_apart', 'fp_background'):
                _gt_label[0] = -1
            elif ann_status in ('tp', 'fp_concurrent'):
                _gt_label[0] = 1
            else:
                raise AssertionError(f'Invalid ann_status: {ann_status}')
        else:
            """relative oracle: TP vs FP"""
            if ann_status == 'fp_background':
                _gt_label[0] = -1
            else:
                _gt_label[0] = 1

        return _gt_label

    def predict(self, frame, det_data, ann_status):
        """
        :type frame: np.ndarray
        :type det_data: np.ndarray
        :type ann_status: str
        :rtype: None
        """

        # if ann_status is None:
        #     ann_status = self._ann_status

        if ann_status is not None:
            self._gt_label = self._get_label(ann_status)
            self._set_stats(ann_status)

        """frame IDs used to index frames so must be set to 0"""
        _det_data = det_data[:].reshape((1, -1))
        _det_data[0, 0] = 0
        self._features = self._feature_extractor.get(_det_data, [frame, ], labels=self._gt_label)

        # self._features = np.ones((1, self._n_features))
        # h, w = frame.shape[:2]
        # max_data = np.asarray((w, h, w, h, 1), dtype=np.float32).reshape((1, 5))
        # self._features[0, :5] = det_data[2:7].reshape((1, 5)) / max_data

        _labels, _probs = self._model.predict(self._features, gt_labels=self._gt_label)
        if self._params.save_mode:
            self._add_test_samples(self._features, self._gt_label, _labels, is_synthetic=0)

        # self._stats['total'] += 1

        if _labels[0] < 0:
            # if self._params.verbose:
            #     self._logger.info('False positive _det_data')
            self.state = MDPStates.inactive

        else:
            # if self._params.verbose:
            #     self._logger.info('True positive _det_data')
            self.state = MDPStates.tracked

        if ann_status is not None:
            if _labels[0] == self._gt_label[0]:
                correctness = 'correct'
            else:
                correctness = 'incorrect'

            if _labels[0] < 0:
                decision = 'negative'
            else:
                decision = 'positive'

            if self._params.verbose:
                self._logger.debug(f'{correctness} {decision} decision')

            if self._params.enable_stats:
                self._update_stats(correctness, decision)

            if self._writer is not None and self._model is not None:
                self._model.add_tb_image(self._writer, self._features, _labels, self._gt_label,
                                         iteration=None, stats=self._cmb_stats, batch_size=1,
                                         tb_vis_path=None, epoch=None, title='active')

        # pause = 1

    def load(self, load_dir):
        """
        :type load_dir: str
        :rtype: None
        """

        if self.is_pretrained:
            self._logger.warning('skipping loading as pretrained model already exists')
            return True

        if not self._model.load(load_dir):
            raise AssertionError('Model loading failed')

        # self.load_path = load_dir

        return True

    def save(self, save_dir):
        """
        :type save_dir: str
        :rtype: None
        """
        if self.is_pretrained:
            self._logger.warning('skipping saving of pretrained model')
            return

        os.makedirs(save_dir, exist_ok=True)

        self._model.save(save_dir)

    def add_synthetic_samples(self, frame, curr_det_data, gt_boxes, ann_status, all_sampled_boxes):

        if not self._params.syn_samples:
            # self._logger.warning('synthetic samples are disabled')
            return

        if ann_status == 'fp_background':
            # self._logger.warning('not adding synthetic samples for fp_background objects')
            return

        # vis = 1
        vis = self._params.vis

        # min_shift_ratio = 1 - self._params.synthetic_neg_iou
        min_shift_ratio = 0
        sampled_boxes = []
        sampled_labels = []

        """
        negative samples
        """
        min_anchor_iou, max_anchor_iou = self._params.syn_neg_iou
        min_anchor_iou = 0.01
        max_gt_iou = max_anchor_iou

        neg_boxes = get_shifted_boxes(curr_det_data[2:6], frame, self._params.syn_samples,
                                      min_anchor_iou=min_anchor_iou,
                                      max_anchor_iou=max_anchor_iou,
                                      min_shift_ratio=min_shift_ratio,
                                      max_shift_ratio=1.0,
                                      gt_boxes=gt_boxes,
                                      max_gt_iou=max_gt_iou,
                                      sampled_boxes=all_sampled_boxes,
                                      vis=self._params.vis,
                                      name='active neg',
                                      )

        sampled_boxes += neg_boxes
        all_sampled_boxes += neg_boxes
        sampled_labels += [-1, ] * len(neg_boxes)

        """
        positive samples
        """
        min_anchor_iou, max_anchor_iou = self._params.syn_pos_iou
        max_gt_iou = max_anchor_iou
        pos_boxes = get_shifted_boxes(curr_det_data[2:6], frame, self._params.syn_samples,
                                      min_anchor_iou=min_anchor_iou,
                                      max_anchor_iou=max_anchor_iou,
                                      min_shift_ratio=min_shift_ratio,
                                      max_shift_ratio=1.0,
                                      gt_boxes=gt_boxes,
                                      max_gt_iou=max_gt_iou,
                                      sampled_boxes=all_sampled_boxes,
                                      vis=self._params.vis,
                                      name='active pos',
                                      )

        sampled_boxes += pos_boxes
        all_sampled_boxes += pos_boxes
        sampled_labels += [1, ] * len(pos_boxes)

        if not sampled_boxes:
            # self._logger.warning('No synthetic samples could be found')
            return

        dummy_dets = np.tile(curr_det_data.reshape((1, -1)), (len(sampled_boxes), 1))
        dummy_dets[:, 2:6] = sampled_boxes
        """frame IDs used to index frames so must be set to 0"""
        dummy_dets[:, 0] = 0

        sampled_labels = np.asarray(sampled_labels)

        _features = self._feature_extractor.get(dummy_dets, [frame, ])

        if vis:
            vis_img = self._feature_extractor.vis(_features, sampled_labels)

            disp_img = np.copy(frame)
            if len(disp_img.shape) == 2:
                """grey scale to RGB"""
                disp_img = np.stack((disp_img,) * 3, axis=2)

            if gt_boxes is not None:
                draw_boxes(disp_img, gt_boxes, _id=None, color='blue', thickness=2,
                           is_dotted=0, transparency=0.)
            if neg_boxes:
                draw_boxes(disp_img, neg_boxes, _id=None, color='red', thickness=2,
                           is_dotted=1, transparency=0.)
            if pos_boxes:
                draw_boxes(disp_img, pos_boxes, _id=None, color='green', thickness=2,
                           is_dotted=1, transparency=0.)

            draw_box(disp_img, curr_det_data[2:6], _id=None, color='black', thickness=2,
                     is_dotted=0, transparency=0.)

            annotate_and_show('active add_synthetic_samples', [disp_img, vis_img], n_modules=0)

        self._add_test_samples(_features, sampled_labels, None, is_synthetic=1)
