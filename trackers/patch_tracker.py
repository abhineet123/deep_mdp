import os
import cv2
import numpy as np
import time
from utilities import draw_region, col_bgr, CVConstants

try:
    import pyMTF

    mtf_available = 1
except ImportError as e:
    print('Failed to import MTF: {}'.format(e))
    mtf_available = 0


class PatchTrackerParams:
    """
    :type use_mtf: int | bool
    :type mtf_cfg_dir: str
    :type tracker_type: int
    :type show: int | bool
    :type save: int | bool
    :type box_color: str
    :type text_fmt: tuple(str, int, float, int, int)
    :type save_fmt: tuple(str, str, int)
    """

    def __init__(self):
        self.use_mtf = 1
        self.mtf_cfg_dir = 'mtf'
        self.tracker_type = 0

        self.show = 1
        self.convert_to_rgb = 0
        self.thickness = 2
        self.box_color = 'red'
        self.resize_factor = 1.0
        self.show_text = 1
        self.text_fmt = ('green', 0, 5, 1.0, 1)
        self.save = 0
        self.save_fmt = ('avi', 'XVID', 30)
        self.save_dir = 'videos'

        self.help = {
            'use_mtf': 'use MTF patch tracker',
            'mtf_cfg_dir': 'directory containing the cfg files for MTF',
            'tracker_type': 'tracker type to use if use_mtf is disabled',
            'show': 'show the tracked object location drawn on the input image',
            'convert_to_rgb': 'convert the image to RGB before showing it; this is sometimes needed if the raw frame is'
                              ' in BGR format so that it does not show correctly (blue and red channels are '
                              'interchanged)',
            'thickness': 'thickness of the bounding box lines drawn on the image',
            'box_color': 'color of the bounding box used to represent the tracked object location',
            'resize_factor': 'multiplicative factor by which the images are resized before being shown or saved',
            'show_text': 'write text in the top left corner of the image to indicate the frame number and FPS',
            'text_fmt': '(color, location, font, font_size, thickness) of the text used to '
                        'indicate the frame number and FPS; '
                        'Available fonts: '
                        '0: cv2.FONT_HERSHEY_SIMPLEX, '
                        '1: cv2.FONT_HERSHEY_PLAIN, '
                        '2: cv2.FONT_HERSHEY_DUPLEX, '
                        '3: cv2.FONT_HERSHEY_COMPLEX, '
                        '4: cv2.FONT_HERSHEY_TRIPLEX, '
                        '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                        '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                        '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; '
                        'Locations: 0: top left, 1: top right, 2: bottom right, 3: bottom left',
            'save': 'save the visualization result with tracked object location drawn on the'
                    ' input image as a video file',
            'save_fmt': '(extension, encoder, FPS) of the saved video',
            'save_dir': 'directory where to save the video',
        }


class PatchTracker:
    def __init__(self, params, logger, target_id, _id, mtf_args=''):
        """
        :type params: PatchTrackerParams
        :type logger: logging.logger
        :type target_id: int
        :type _id: int
        :rtype None
        """
        self.params = params
        self.logger = logger
        self.target_id = target_id
        self._id = _id
        self.mtf_args = mtf_args

        self.is_created = False
        self.is_terminated = False
        self.is_initialized = False

        self.cv_tracker = None

        self.box_color = col_bgr[self.params.box_color]
        self.text_color = col_bgr[self.params.text_fmt[0]]
        self.text_font = CVConstants.fonts[self.params.text_fmt[2]]
        self.text_font_size = self.params.text_fmt[3]
        self.text_thickness = self.params.text_fmt[4]
        self.text_location = (5, 15)
        if cv2.__version__.startswith('2'):
            self.text_line_type = cv2.CV_AA
        else:
            self.text_line_type = cv2.LINE_AA

        if not mtf_available:
            print('MTF is not available')
            self.params.use_mtf = 0

        if self.params.use_mtf:
            self.logger.info('Using MTF tracker')
        else:

            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

            if major_ver < 3:
                self.logger.error('OpenCV trackers are not available')
                return

            tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
            tracker_type = tracker_types[self.params.tracker_type]

            self.logger.info('Using OpenCV {:s} tracker'.format(tracker_type))
            if int(minor_ver) < 3:
                self.cv_tracker = cv2.Tracker_create(tracker_type)
            else:
                if tracker_type == 'BOOSTING':
                    self.cv_tracker = cv2.TrackerBoosting_create()
                if tracker_type == 'MIL':
                    self.cv_tracker = cv2.TrackerMIL_create()
                if tracker_type == 'KCF':
                    self.cv_tracker = cv2.TrackerKCF_create()
                if tracker_type == 'TLD':
                    self.cv_tracker = cv2.TrackerTLD_create()
                if tracker_type == 'MEDIANFLOW':
                    self.cv_tracker = cv2.TrackerMedianFlow_create()
                if tracker_type == 'GOTURN':
                    self.cv_tracker = cv2.TrackerGOTURN_create()

        self.window_name = 'Target {:d} : Press Space/Esc to stop tracking'.format(
            self.target_id, self._id)

        self.curr_corners = np.zeros((2, 4), dtype=np.float64)
        self.out_bbox = None
        self.is_created = True
        self.video_writer = None

    def initialize(self, init_frame, init_bbox):

        # extract the true corners in the first frame and place them into a 2x4 array
        xmin = init_bbox['xmin']
        xmax = init_bbox['xmax']
        ymin = init_bbox['ymin']
        ymax = init_bbox['ymax']

        shape = init_frame.shape
        # print('init_frame.shape: ', init_frame.shape)
        if len(shape) == 3:
            n_rows, n_cols, n_ch = shape
        else:
            n_rows, n_cols = shape

        if self.params.text_fmt[1] == 1:
            self.text_location = (n_cols - 100, 15)
        elif self.params.text_fmt[1] == 2:
            self.text_location = (n_cols - 100, n_rows - 15)
        elif self.params.text_fmt[1] == 3:
            self.text_location = (5, n_rows - 15)
        else:
            self.text_location = (5, 15)

        if not self.params.use_mtf:
            width = xmax - xmin + 1
            height = ymax - ymin + 1
            roi = (xmin, ymin, width, height)
            ok = self.cv_tracker.init(init_frame, roi)
            if not ok:
                self.logger.error('Tracker initialization was unsuccessful')
                return
        else:
            # if len(init_frame.shape) == 3:
            #     init_frame_gs = cv2.cvtColor(init_frame, cv2.COLOR_BGR2GRAY)
            # else:
            #     init_frame_gs = init_frame

            init_corners = [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ]
            init_corners = np.array(init_corners).T
            try:
                tracker_created = pyMTF.create(init_frame.astype(np.uint8), init_corners.astype(np.float64),
                                               self.params.mtf_cfg_dir)
            except:
                tracker_created = False
            if not tracker_created:
                self.logger.error('Tracker creation was unsuccessful')
                return
            self._id = tracker_created

        if self.params.show:
            # window for displaying the tracking result
            cv2.namedWindow(self.window_name)

        if self.params.save:
            time_str = time.strftime("%y%m%d_%H%M", time.localtime())
            save_fname = 'target_{:d}_{:s}.{:s}'.format(self._id, time_str, self.params.save_fmt[0])
            save_path = os.path.join(self.params.save_dir, save_fname)
            os.makedirs(self.params.save_dir, exist_ok=True)
            frame_size = (init_frame.shape[1], init_frame.shape[0])
            if self.params.resize_factor != 1:
                frame_size = (int(frame_size[0] * self.params.resize_factor),
                              int(frame_size[1] * self.params.resize_factor))
            self.video_writer = cv2.VideoWriter()
            if cv2.__version__.startswith('2'):
                self.video_writer.open(filename=save_path, fourcc=cv2.cv.CV_FOURCC(*self.params.save_fmt[1]),
                                       fps=self.params.save_fmt[2], frameSize=frame_size)
            else:
                self.video_writer.open(filename=save_path, apiPreference=cv2.CAP_FFMPEG,
                                       fourcc=cv2.VideoWriter_fourcc(*self.params.save_fmt[1]),
                                       fps=int(self.params.save_fmt[2]), frameSize=frame_size)

            if not self.video_writer.isOpened():
                self.logger.error('Video file {:s} could not be opened'.format(save_path))
                return
            print('Saving tracking output to {:s}'.format(save_path))

        self.is_initialized = True

    def update(self, frame, frame_id):
        start_time = time.clock()

        if not self.params.use_mtf:
            ok, bbox = self.cv_tracker.update(frame)
            if not ok:
                self.logger.error('Tracker update was unsuccessful')
                self.out_bbox = None
                self.is_terminated = True
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                return

            xmin, ymin, width, height = bbox
            xmax = xmin + width - 1
            ymax = ymin + height - 1
            self.curr_corners[:, 0] = (xmin, ymin)
            self.curr_corners[:, 1] = (xmax, ymin)
            self.curr_corners[:, 2] = (xmax, ymax)
            self.curr_corners[:, 3] = (xmin, ymax)
        else:
            # if len(frame.shape) == 3:
            #     frame_gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # else:
            #     frame_gs = frame
            # update the tracker with the current frame
            success = pyMTF.getRegion(frame, self.curr_corners, self._id)
            if not success:
                self.logger.error('Tracker update was unsuccessful')
                self.out_bbox = None
                self.is_terminated = True
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                if self.params.use_mtf:
                    pyMTF.remove(self._id)
                return

        end_time = time.clock()
        # compute the tracking fps
        fps = 1.0 / (end_time - start_time)

        if self.params.show:
            if self.params.convert_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # draw the tracker location
            draw_region(frame, self.curr_corners, self.box_color, self.params.thickness)
            if self.params.show_text:
                # write statistics (error and fps) to the image
                cv2.putText(frame, "frame {:d} {:5.2f} fps".format(frame_id, fps), self.text_location,
                            self.text_font, self.text_font_size, self.text_color, self.text_thickness,
                            self.text_line_type)
            if self.params.resize_factor != 1:
                frame = cv2.resize(frame, (0, 0), fx=self.params.resize_factor,
                                   fy=self.params.resize_factor)
            # display the image
            cv2.imshow(self.window_name, frame)

            if self.video_writer is not None:
                self.video_writer.write(frame)

            k = cv2.waitKey(1)
            if k == 27 or k == 32:
                cv2.destroyWindow(self.window_name)
                if self.params.use_mtf:
                    pyMTF.remove(self._id)
                self.is_terminated = True
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None

        # print('curr_corners: ', curr_corners)
        xmin = int(self.curr_corners[0, 0])
        ymin = int(self.curr_corners[1, 0])
        xmax = int(self.curr_corners[0, 2])
        ymax = int(self.curr_corners[1, 2])

        self.out_bbox = dict(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
        )
