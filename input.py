import os
import numpy as np
import time
import sys
import copy
import cv2

from data import Data
from objects import Annotations, Detections, TrackingResults
from utilities import ImageSequenceCapture, draw_box, linux_path, CustomLogger, BaseParams


class Input:
    """
     read images, annotations, detections and tracking results for a sequence

    :type params: Input.Params
    :type _logger: CustomLogger
    :type all_frames: list[np.ndarray]
    :type curr_frame: np.ndarray
    :type frame_id: int
    :type frame_size: tuple
    :type _cap: cv2.VideoCapture
    :type annotations: Annotations
    :type track_res: TrackingResults
    :type detections: Detections
    """

    class Params(BaseParams):
        """
        :type annotations: Annotations.Params
        :type track_res: TrackingResults.Params
        :type detections: Detections.Params


         :ivar path: path of the directory, video file or binary file from where
        images are to be read; if this is not specified then it is computed from
        db_root_path, sequence set/name (in Params.py), source_type and img_fmt/vid_fmt

        :ivar frame_ids: two element tuple specifying the IDs of the first and
        last frames in the sub sequence; if either is less than 0, it is computed
        from ratios and offsets

        :ivar db_root_path: path of the directory that contains all datasets

        :ivar source_type: 0: image sequence 1: video file

        :ivar img_fmt: (naming scheme, extension) of each image file in the sequence

        :ivar vid_fmt: file extension of the video file

        :ivar resize_factor: multiplicative factor by which to resize the input
        images

        :ivar convert_to_gs: convert all input images to greyscale (if they are
        RGB) before they are used

        :ivar read_from_bin: write all image data to a binary file for quick and
        lossless reading

        :ivar write_to_bin: write all image data to a binary file for quick and
        lossless reading

        :ivar batch_mode: read all frames in the input sequence at once to avoid
        repeated disk accesses

        :ivar roi: four element tuple specifying the region of interest (ROI)
        as (xmin, ymin, xmax, ymax) of the corresponding bounding box; ROI is
        only enabled if xmx > xmin and ymax > ymin

        :ivar annotations: parameters for Annotations in Objects module

        :ivar track_res: parameters for tracking results (same as Annotations
        in Objects module)

        :ivar detections: parameters for Detections in Objects module


        """

        def __init__(self):
            self.path = ''
            self.frame_ids = (-1, -1)

            self.db_root_path = '/data'
            self.source_type = 0
            self.img_fmt = ('image%06d', 'jpg')
            self.vid_fmt = 'mp4'

            self.resize_factor = 1
            self.convert_to_gs = 0
            self.read_from_bin = 0
            self.write_to_bin = 0
            self.batch_mode = 1
            self.roi = (0, 0, 0, 0)

            self.annotations = Annotations.Params()
            self.track_res = TrackingResults.Params()
            self.detections = Detections.Params()

    def __init__(self, params, logger):
        """
        :type params: Input.Params
        :type logger: CustomLogger
        :rtype: None
        """

        self.params = copy.deepcopy(params)
        self._logger = logger

        self.curr_frame = None
        self.all_frames = None
        self.frame_id = 0
        self.n_pix = 0
        self.frame_size = None
        self.bin_fid = None
        self.source_path = None
        self._cap = None

        self.roi = self.params.roi
        if self.roi[0] >= self.roi[2] or self.roi[1] >= self.roi[3]:
            self.roi = None

        self.annotations = None
        self.detections = None
        self.track_res = None

        self.read_from_bin = self.params.read_from_bin

        self.start_frame_id, self.end_frame_id = self.params.frame_ids
        self.seq_name = None
        self.seq_set = None
        self.seq_n_frames = 0
        self.n_frames = 0

        self.is_initialized = False

    def initialize(self, data, read_img=True, logger=None):
        """
        :type data: Data
        :type read_img: bool | int
        :type logger: CustomLogger | logging.RootLogger
        :rtype: bool
        """

        if logger is not None:
            self._logger = logger

        self.annotations = None
        self.detections = None
        self.track_res = None

        self.frame_id = 0
        self.start_frame_id, self.end_frame_id = self.params.frame_ids

        curr_path = self.params.path

        if curr_path:
            self.seq_name = os.path.splitext(os.path.basename(curr_path))[0]
            self.seq_set = "custom"
            self.seq_n_frames = self.get_n_frames(curr_path)
            if self.seq_n_frames <= 0:
                raise AssertionError('No. of frames could not be obtained from: {}'.format(curr_path))

            if self.start_frame_id < 0:
                self.start_frame_id = 0
            if self.end_frame_id < 0:
                self.end_frame_id = self.seq_n_frames - 1
        else:
            if not data.is_initialized:
                raise AssertionError('Source path must be provided with uninitialized data module')
            self.seq_name, self.seq_set, self.seq_n_frames = data.seq_name, data.seq_set, data.seq_n_frames

            if self.start_frame_id < 0:
                self.start_frame_id = data.start_frame_id
            if self.end_frame_id < 0:
                self.end_frame_id = data.end_frame_id
        self.n_frames = self.end_frame_id - self.start_frame_id + 1

        self._logger.info('seq_set: {:s}'.format(self.seq_set))
        self._logger.info('seq_name: {:s}'.format(self.seq_name))
        self._logger.info('seq_n_frames: {:d}'.format(self.seq_n_frames))
        self._logger.info('start_frame_id: {:d}'.format(self.start_frame_id))
        self._logger.info('end_frame_id: {:d}'.format(self.end_frame_id))
        self._logger.info('n_frames: {:d}'.format(self.n_frames))

        if not read_img:
            # input pipeline is needed only for reading objects
            self._logger.info('Skipping image pipeline initialization')
            return True

        self.read_from_bin = self.params.read_from_bin

        if self.read_from_bin:
            if curr_path:
                bin_path = curr_path
            else:
                bin_fname = '{:s}_{:d}_{:d}'.format(self.seq_name, self.start_frame_id + 1,
                                                    self.end_frame_id + 1)
                bin_path = linux_path(self.params.db_root_path, self.seq_set,
                                      'Images', bin_fname) + '.bin'

            if not os.path.isfile(bin_path):
                self._logger.info('Binary data file {:s} does not exist'.format(bin_path))
                if curr_path:
                    return False
                self.read_from_bin = 0
            else:
                self._logger.info('Reading image data from binary file: {:s}'.format(bin_path))
                self.bin_fid = open(bin_path, 'rb')
                self.frame_size = tuple(np.fromfile(self.bin_fid, dtype=np.uint32, count=2))
                self.n_pix = self.frame_size[0] * self.frame_size[1]
                actual_file_size = os.path.getsize(bin_path)
                expected_file_size = self.n_frames * self.n_pix + 8
                if expected_file_size != actual_file_size:
                    self._logger.info('Size of binary file: {:d} does not match the expected size: {:d}'.format(
                        actual_file_size, expected_file_size))
                    self.read_from_bin = 0
                    self.bin_fid.close()
                else:
                    self.curr_frame = np.fromfile(self.bin_fid, dtype=np.uint8, count=self.n_pix).reshape(
                        (self.frame_size[1], self.frame_size[0]))
                    self.params.write_to_bin = 0

        if not self.read_from_bin:
            if not curr_path:
                curr_path = linux_path(self.params.db_root_path, self.seq_set,
                                       'Images', self.seq_name)
            if self.params.source_type == 1:
                self.source_path = curr_path + '.' + self.params.vid_fmt
            else:
                self.source_path = curr_path
                if self.params.img_fmt:
                    self.source_path = linux_path(self.source_path, self.params.img_fmt[0]
                                                  + '.' + self.params.img_fmt[1])

            self._logger.info('Getting images from: {:s}'.format(self.source_path))

            if self.params.source_type == 1:
                self._cap = cv2.VideoCapture()
                if not self._cap.open(self.source_path):
                    err_txt = 'Video file ' + self.source_path + ' could not be opened'
                    raise AssertionError(err_txt)
            else:
                self._cap = ImageSequenceCapture()  # type: ImageSequenceCapture
                if not self._cap.open(self.source_path):
                    err_txt = 'Image sequence ' + self.source_path + ' could not be read'
                    raise AssertionError(err_txt)
                assert self.seq_n_frames == self._cap.n_src_files, \
                    f"Mismatch between seq_n_frames: {self.seq_n_frames} and n_src_files: {self._cap.n_src_files}"

            if self.start_frame_id > 0:
                if cv2.__version__.startswith('2'):
                    cv_prop = cv2.cv.CV_CAP_PROP_POS_FRAMES
                else:
                    cv_prop = cv2.CAP_PROP_POS_FRAMES

                self._cap.set(cv_prop, self.start_frame_id)
                next_frame_id = self._cap.get(cv_prop)
                if next_frame_id != self.start_frame_id:
                    if self.params.source_type == 1:
                        self._logger.info('OpenCV VideoCapture set property functionality is not available so '
                                          'manually skipping {:d} frames'.format(self.start_frame_id))
                        for i in range(self.start_frame_id):
                            ret, _ = self._cap.read()
                            if not ret:
                                raise AssertionError('Frame {:d} could not be read'.format(self.frame_id + 1))
                    else:
                        raise AssertionError(
                            'Something weird going on in ImageSequenceCapture'.format(self.start_frame_id))
            # curr_frame_id = self._cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            # self.logger.debug('curr_frame_id before initialize: {:f}'.format(curr_frame_id))
            ret, frame = self._cap.read()
            if not ret:
                raise AssertionError('Frame {:d} could not be read'.format(self.frame_id + 1))
            # curr_frame_id = self._cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            # self.logger.debug('curr_frame_id after initialize: {:f}'.format(curr_frame_id))
            # cv2.imshow('Frame',frame)
            # if cv2.waitKey(0) == 27:
            # exit()
            self._logger.info('Input pipeline initialized successfully with frames of size: {:d} x {:d}'.format(
                frame.shape[1], frame.shape[0]))
            if self.params.convert_to_gs:
                self.curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                self.curr_frame = frame

            if self.params.resize_factor != 1:
                self.curr_frame = cv2.resize(self.curr_frame, dsize=(0, 0), fx=self.params.resize_factor,
                                             fy=self.params.resize_factor)
                self._logger.info('Resizing input frames to : {:d} x {:d}'.format(
                    self.curr_frame.shape[1], self.curr_frame.shape[0]))

            # (width, height) format rather than the (n_rows, n_cols) of shape
            self.frame_size = (self.curr_frame.shape[1], self.curr_frame.shape[0])
            self.n_pix = self.frame_size[0] * self.frame_size[1]

            if self.params.write_to_bin:
                if self.params.source_type == 1:
                    bin_path = linux_path(os.path.dirname(self.source_path),
                                          os.path.splitext(os.path.basename(self.source_path))[0] + ".bin")
                else:
                    bin_path = curr_path + '.bin'
                self.bin_fid = open(bin_path, 'wb')
                self._logger.info('Writing image data to binary file: {:s}'.format(bin_path))
                np.array(self.frame_size, dtype=np.uint32).tofile(self.bin_fid)
                self.curr_frame.astype(np.uint8).tofile(self.bin_fid)

        if self.params.batch_mode:
            if not self._read_all_frames():
                return False

        self.is_initialized = True
        return True

    def update(self):
        """
        :rtype: bool
        """
        if self.frame_id >= self.n_frames - 1:
            return False

        if self.read_from_bin:
            self.curr_frame = np.fromfile(self.bin_fid, dtype=np.uint8, count=self.n_pix).reshape(
                (self.frame_size[1], self.frame_size[0]))
            self.frame_id += 1
            if self.frame_id == self.n_frames - 1:
                self.bin_fid.close()
        else:
            ret, curr_frame_rgb = self._cap.read()
            if not ret:
                raise AssertionError('Frame {:d} could not be read'.format(self.frame_id + 1))
                # raise IOError('End of sequence reached unexpectedly')
            if self.params.convert_to_gs:
                self.curr_frame = cv2.cvtColor(curr_frame_rgb, cv2.COLOR_BGR2GRAY)
                # self.curr_frame = cv2.cvtColor(curr_frame_rgb.astype(np.float32), cv2.COLOR_BGR2GRAY)
                # self.curr_frame = np.ceil(self.curr_frame).astype(np.uint8)
            else:
                self.curr_frame = curr_frame_rgb
            if self.params.resize_factor != 1:
                self.curr_frame = cv2.resize(self.curr_frame, dsize=(0, 0), fx=self.params.resize_factor,
                                             fy=self.params.resize_factor)
            self.frame_id += 1
            if self.params.write_to_bin:
                self.curr_frame.astype(np.uint8).tofile(self.bin_fid)
                if self.frame_id == self.n_frames - 1:
                    self.bin_fid.close()
        return True

    def _read_all_frames(self):
        """
        :rtype: bool
        """
        self._logger.info('Reading all frames...')
        start_t = time.time()
        self.all_frames = [None] * self.n_frames
        # print 'len(self.all_frames): ', len(self.all_frames)
        self.all_frames[self.frame_id] = self.curr_frame
        print_diff = max(1, int(5 * round(self.n_frames * 0.04 / 5)))
        while self.frame_id < self.n_frames - 1:
            if not self.update():
                sys.stdout.write('\n')
                return False
            # print 'self.frame_id: ', self.frame_id
            self.all_frames[self.frame_id] = self.curr_frame
            if (self.frame_id + 1) % print_diff == 0:
                sys.stdout.write('\rDone {:d}/{:d} frames'.format(
                    self.frame_id + 1, self.n_frames))
                sys.stdout.flush()
        end_t = time.time()
        sys.stdout.write('\rDone {:d}/{:d} frames\n'
                         'Time taken: {:f} secs\n'.format(
            self.n_frames, self.n_frames, end_t - start_t))
        sys.stdout.flush()
        return True

    def get_n_frames(self, path):
        return cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT)

    # def clearDetections(self, build_index):
    #     self.detections = None

    def read_detections(self):
        """
        :rtype: bool
        """
        detections_params = copy.deepcopy(self.params.detections)
        if not detections_params.path:
            detections_params.path = linux_path(self.params.db_root_path, self.seq_set,
                                                detections_params.src_dir, self.seq_name + '.txt')

        self.detections = Detections(detections_params, self._logger)
        self.detections.initialize(self.seq_n_frames,
                                   self.start_frame_id, self.end_frame_id)

        ignored_regions = None
        if self.annotations is not None:
            ignored_regions = self.annotations.ignored_regions

        if not self.detections.read(
                frame_size=self.frame_size,
                ignored_regions=ignored_regions,
                resize_factor=self.params.resize_factor):
            self._logger.error('Failed to read detections')
            self.detections = None
            return False

        return True

    # def clearAnnotations(self, build_index):
    #     self.annotations = None

    def read_annotations(self):
        """
        :rtype: bool
        """
        annotations_params = copy.deepcopy(self.params.annotations)
        if not annotations_params.path:
            annotations_params.path = linux_path(self.params.db_root_path, self.seq_set,
                                                 annotations_params.src_dir, self.seq_name + '.txt')

        self.annotations = Annotations(annotations_params, self._logger)
        self.annotations.initialize(self.seq_n_frames, self.start_frame_id, self.end_frame_id)

        if not self.annotations.read(resize_factor=self.params.resize_factor, frame_size=self.frame_size):
            self._logger.error('Failed to read annotations')
            self.annotations = None
            return False

        return True

    # def clearTrackingResults(self, build_index):
    #     self.tracking_res = None

    def read_tracking_results(self, res_path):
        """
        :type res_path: str
        :type build_index: bool
        :rtype: bool
        """
        self.params.track_res.path = res_path
        self.track_res = TrackingResults(self.params.track_res, self._logger)
        self.track_res.initialize(self.seq_n_frames,
                                  self.start_frame_id, self.end_frame_id)
        if not self.track_res.read(self.frame_size, self.params.resize_factor):
            self._logger.error('Failed to read tracking results')
            self.track_res = None
            return False
        return True

    def get_frame(self):
        return self.curr_frame

    def get_all_frames(self):
        return self.all_frames

    def show_detections(self):
        for frame_id in range(self.n_frames):
            frame_disp = cv2.cvtColor(self.all_frames[frame_id], cv2.COLOR_GRAY2BGR)
            det_ids = self.detections.idx[frame_id]
            if det_ids is None:
                continue
            for det_id in det_ids:
                draw_box(frame_disp, self.detections.data[det_id, 2:6])
            cv2.imshow('Detections', frame_disp)
            if cv2.waitKey(1) == 27:
                exit()

    def show_annotations(self):
        for frame_id in range(self.n_frames):
            if self.all_frames[frame_id] is None:
                raise SystemError('frame is None')
            frame_disp = np.copy(self.all_frames[frame_id]).astype(np.uint8)
            # frame_disp = cv2.cvtColor(self.all_frames[frame_id], cv2.COLOR_GRAY2BGR)
            ann_ids = self.annotations.idx[frame_id]
            if ann_ids is None:
                continue
            for ann_id in ann_ids:
                draw_box(frame_disp, self.annotations.data[ann_id, 2:6])
            cv2.imshow('Annotations', frame_disp)
            if cv2.waitKey(1) == 27:
                exit()
