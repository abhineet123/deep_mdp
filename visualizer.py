import os
import sys
import numpy as np
import cv2

from input import Input
from objects import Objects, TrackingResults
# from gate import Gate
from utilities import MDPStates, draw_box, draw_trajectory, resize_ar, col_bgr, \
    CVConstants, get_date_time, linux_path, CustomLogger, SIIF, BaseParams, annotate_and_show, VideoWriterGPU


class ObjTypes:
    enum = (
        'tracking_result',
        'detection',
        'annotation',
    )
    postfix = ('res', 'det', 'ann')
    tracking_result, detection, annotation = enum


class ImageWriter:
    def __init__(self, file_path, logger):
        self.file_path = file_path
        self.logger = logger
        split_path = os.path.splitext(file_path)
        self.save_dir = split_path[0]
        self.ext = split_path[1][1:]

        os.makedirs(self.save_dir, exist_ok=True)

        self.frame_id = 0

        self.logger.info('Saving images of type {:s} to {:s}'.format(self.ext, self.save_dir))

    def write(self, frame):
        self.frame_id += 1
        cv2.imwrite(linux_path(self.save_dir, 'image{:06d}.{:s}'.format(self.frame_id, self.ext)), frame)

    def release(self):
        pass


class Visualizer:
    """
    :type _params: Visualizer.Params
    :type _logger: logging.RootLogger | logging.Logger
    :type _traj_data: list[dict{int:list[int]}]
    """

    class Params(BaseParams):
        """
        :type mode: (int, int, int)
        :type tracked_cols: tuple(str,)
        :type lost_cols: tuple(str,)
        :type inactive_cols: tuple(str,)
        :type det_cols: tuple(str,)
        :type ann_cols: tuple(str,)
        :type text_fmt: tuple(str, int, float, int, int)
        :type gate_fmt: tuple(str, float, float, int, int)
        :type pause_after_frame: bool
        :type show: int
        :type help: {str:str}


        :ivar mode: 'three element tuple to specify which kinds of objects are to be shown:'
        '(tracked, detections, annotations)',
        :ivar tracked_cols: 'bounding box colors in which to show the tracking result for objects in tracked state; '
                'if there are more objects than the number of specified colors, modulo indexing is used',
        :ivar lost_cols: 'bounding box colors in which to show the tracking result for objects in lost state',
        :ivar inactive_cols: 'bounding box colors in which to show the tracking result for objects in inactive state',
        :ivar det_cols: 'bounding box colors in which to show the detections',
        :ivar ann_cols: 'bounding box colors in which to show the annotations',
        :ivar convert_to_rgb: 'convert the image to RGB before showing it; this is sometimes needed if the raw frame is'
                  ' in BGR format so that it does not show correctly (blue and red channels are '
                  'interchanged)',
        :ivar pause_after_frame: 'pause execution after each frame till a key is pressed to continue;'
             'Esc: exit the program'
             'Spacebar: toggle this parameter',
        :ivar show_trajectory: 'show the trajectory of bounding boxes with associated unique IDs by drawing lines '
                   'connecting their centers across consecutive frames',
        :ivar box_thickness: 'thickness of lines used to draw the bounding boxes',
        :ivar traj_thickness: 'thickness of lines used to draw the trajectories',
        :ivar resize_factor: 'multiplicative factor by which the images are resized before being shown or saved',
        :ivar disp_size: 'Size of the displayed frame – Overrides resize_factor',
        :ivar text_fmt: '(color, location, font, font_size, font_thickness) of the text used to '
            'indicate the frame number; '
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
        :ivar gate_fmt: '(color, thickness, font, font_size, font_thickness) of the lines and labels used '
            'for showing the gates',
        :ivar show: 'Show the images with drawn objects; this can be disabled when running in batch mode'
        ' or on a system without GUI; the output can instead be saved as a video file',
        :ivar save: 'Save the images with drawn objects as video files',
        :ivar save_prefix: 'Prefix to be added to the name of the saved video files',
        :ivar save_dir: 'Directory in which to save the video files',
        :ivar save_fmt: '3 element tuple to specify the (extension, FOURCC format string, fps) of the saved video file;'
            'refer http://www.fourcc.org/codecs.php for a list of valid FOURCC strings; '
            'extension can be one of [jpg, bmp, png] to write to an image sequence instead of a video file',

        """

        def __init__(self):
            """
            :rtype: None
            """
            self.mode = [1, 1, 1]
            self.tracked_cols = (
                'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
                'dark_slate_gray', 'navy', 'turquoise')
            self.lost_cols = ()
            self.inactive_cols = ('none',)
            self.det_cols = ('green',)
            self.ann_cols = ('green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
                             'dark_slate_gray', 'navy', 'turquoise')

            self.convert_to_rgb = 0
            self.pause_after_frame = 0

            self.show_trajectory = 1
            self.show_invalid = 0
            self.box_thickness = 2
            self.traj_thickness = 2
            self.resize_factor = 1.0
            self.disp_size = (1280, 720)
            self.text_fmt = ('green', 0, 5, 1.0, 1)
            self.gate_fmt = ('black', 2.0, 5, 1.2, 1)

            self.show = 1
            self.save = 0
            # self.save_fmt = ('avi', 'XVID', 30)
            self.save_fmt = ('mkv', 'H264', 30)
            # self.save_fmt = ('mkv', 'H265', 30)
            # self.save_fmt = ('mp4', 'AVC1', 30)
            self.save_dir = 'log/videos'
            self.save_prefix = ''

    def __init__(self, params, logger):
        """

        :param Visualizer.Params params:
        :param logging.RootLogger | logging.Logger logger:
        """
        self._params = params
        self._logger = logger

        self.obj_types = ObjTypes.enum

        self._mode = {
            _obj_type: self._params.mode[i] for i, _obj_type in enumerate(self.obj_types)
        }
        self._writer = {
            _obj_type: None for _obj_type in self.obj_types

        }
        self._traj_data = {
            _obj_type: {} for _obj_type in self.obj_types

        }
        self._traj_data[ObjTypes.detection] = None

        self._objects = {}
        self._pause_after_frame = self._params.pause_after_frame

        """lost and inactive states to be shown in the same color as the tracked state
        or alternatively, the color is independent of the state
        """
        self.show_lost = 1
        self.show_inactive = 1
        if len(self._params.lost_cols) == 0:
            self._params.lost_cols = self._params.tracked_cols
        elif len(self._params.lost_cols) == 1 and self._params.lost_cols[0] == 'none':
            self.show_lost = 0

        if len(self._params.inactive_cols) == 0:
            self._params.inactive_cols = self._params.tracked_cols
        elif len(self._params.inactive_cols) == 1 and self._params.inactive_cols[0] == 'none':
            self.show_inactive = 0

        # if self.show_lost:
        #     self._res_win_title = 'tracked and lost'
        # else:
        self._res_win_title = 'tracking result'

        self._ann_win_title = 'annotations'
        self._det_win_title = 'detections'
        self._failed_targets_win_title = 'failed_targets'

        self.text_color = col_bgr[self._params.text_fmt[0]]
        self.text_font = CVConstants.fonts[self._params.text_fmt[2]]
        self.text_font_size = self._params.text_fmt[3]
        self.text_thickness = self._params.text_fmt[4]
        self.text_location = (5, 15)
        if cv2.__version__.startswith('2'):
            self.text_line_type = cv2.CV_AA
        else:
            self.text_line_type = cv2.LINE_AA

        self.gate_col = col_bgr[self._params.gate_fmt[0]]
        self.gate_thickness = self._params.gate_fmt[1]
        self.gate_font = self._params.gate_fmt[2]
        self.gate_font_size = self._params.gate_fmt[3]
        self.gate_font_thickness = self._params.gate_fmt[4]

        self.image_exts = ['jpg', 'bmp', 'png']

        self._siif = 0

        self.ignored_regions = None

    def initialize(self, save_fname_templ, frame_size, ignored_regions=None):
        """
        :type save_fname_templ: str
        :type frame_size: tuple(int, int)
        :type ignored_regions: np.ndarray | None
        :rtype: bool
        """
        n_cols, n_rows = frame_size
        # print('n_cols: {:d}', n_cols)
        # print('n_rows: {:d}', n_rows)

        self.ignored_regions = ignored_regions

        self._traj_data = {
            _obj_type: {} for _obj_type in self.obj_types

        }
        self._traj_data[ObjTypes.detection] = None

        if self._params.text_fmt[1] == 1:
            self.text_location = (n_cols - 200, 15)
        elif self._params.text_fmt[1] == 2:
            self.text_location = (n_cols - 200, n_rows - 15)
        elif self._params.text_fmt[1] == 3:
            self.text_location = (5, n_rows - 15)
        else:
            self.text_location = (5, 15)

        if not self._params.save:
            return True

        if self._params.save_prefix:
            save_fname_templ = '{:s}_{:s}'.format(self._params.save_prefix, save_fname_templ)

        os.makedirs(self._params.save_dir, exist_ok=True)

        for i, obj_type in enumerate(self.obj_types):
            if self._mode[obj_type]:
                save_fname = '{:s}_{:s}_{:s}.{:s}'.format(
                    save_fname_templ, ObjTypes.postfix[i], get_date_time(), self._params.save_fmt[0])
                obj_save_dir = linux_path(self._params.save_dir, obj_type)

                os.makedirs(obj_save_dir, exist_ok=True)

                save_path = linux_path(obj_save_dir, save_fname)
                if self._params.save_fmt[0] in self.image_exts:
                    writer = ImageWriter(save_path, self._logger)
                else:
                    if self._params.disp_size:
                        frame_size = self._params.disp_size
                    elif self._params.resize_factor != 1:
                        frame_size = (int(frame_size[0] * self._params.resize_factor),
                                      int(frame_size[1] * self._params.resize_factor))
                    if self._params.save_fmt[1].lower() in ("h265", "h265"):
                        writer = VideoWriterGPU(save_path, self._params.save_fmt[2], frame_size)
                    else:
                        writer = cv2.VideoWriter()
                        if cv2.__version__.startswith('2'):
                            writer.open(filename=save_path, fourcc=cv2.cv.CV_FOURCC(*self._params.save_fmt[1]),
                                        fps=self._params.save_fmt[2], frameSize=frame_size)
                        else:
                            writer.open(filename=save_path, apiPreference=cv2.CAP_FFMPEG,
                                        fourcc=cv2.VideoWriter_fourcc(*self._params.save_fmt[1]),
                                        fps=int(self._params.save_fmt[2]), frameSize=frame_size)

                    if not writer.isOpened():
                        raise AssertionError('Video file {:s} could not be opened'.format(save_path))
                self._writer[obj_type] = writer
                self._logger.info('Saving {:s} video to {:s}'.format(obj_type, save_path))
        return True

    def run(self, _input):
        """
        :type _input: Input
        :rtype: bool
        """
        if self._mode[ObjTypes.tracking_result]:
            if _input.track_res is None:
                raise AssertionError('Input tracking results are empty')
            self._objects[ObjTypes.tracking_result] = _input.track_res
        if self._mode[ObjTypes.detection]:
            if _input.detections is None:
                raise AssertionError('Input detections are empty')
            self._objects[ObjTypes.detection] = _input.detections
        if self._mode[ObjTypes.annotation]:
            if _input.annotations is None:
                raise AssertionError('Input annotations are empty')
            self._objects[ObjTypes.annotation] = _input.annotations

        for frame_id in range(_input.n_frames):
            if _input.params.batch_mode:
                frame = _input.all_frames[frame_id]
            else:
                # first frame was read during pipeline initialization
                if frame_id > 0 and not _input.update():
                    raise AssertionError('Input image {:d} could not be read'.format(frame_id))
                frame = _input.curr_frame
            frame_data = {}
            for obj_type in self.obj_types:
                frame_data[obj_type] = None

                if not self._mode[obj_type]:
                    continue

                _obj = self._objects[obj_type]  # type: Objects
                # all objects in the current frame
                ids = _obj.idx[frame_id]
                if ids is not None:
                    _frame_data = _obj.data[ids, :]
                    if obj_type == ObjTypes.tracking_result:
                        _obj = _obj  # type: TrackingResults
                        if _obj.states is not None:
                            _states = _obj.states[ids].reshape((_frame_data.shape[0], 1))
                            _frame_data = np.concatenate((_frame_data, _states), axis=1)
                        if _obj.is_valid is not None:
                            _is_valid = _obj.is_valid[ids].reshape((_frame_data.shape[0], 1))
                            _frame_data = np.concatenate((_frame_data, _is_valid), axis=1)
                    frame_data[obj_type] = _frame_data
                else:
                    # no objects in this frame
                    continue

            if not self.update(frame_id, frame, frame_data):
                return False
        return True

    def update(self, frame_id, frame, frame_data, gates=None,
               deleted_targets=None, failed_targets=None, msg=None, return_mode=0, label=''):
        """
        :type frame_id: int
        :type frame: np.ndarray
        :type frame_data: dict{str: np.ndarray | None}
        :type gates: dict[int:Gate] | None
        :type deleted_targets: list[int] | None
        :type msg: str | None
        :rtype: bool
        """
        assert frame is not None, 'Frame is None'

        assert frame.dtype == np.dtype('uint8'), 'Invalid frame type'

        if len(frame.shape) == 3:
            if self._params.convert_to_rgb:
                curr_frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                curr_frame_disp = np.copy(frame)
        else:
            curr_frame_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.ignored_regions is not None:
            for ignored_region in self.ignored_regions:
                draw_box(curr_frame_disp, ignored_region, color='black',
                         thickness=cv2.FILLED, transparency=0.5)

        if label:
            label = '{} frame {:d}'.format(label, frame_id)
        else:
            label = 'frame {:d}'.format(frame_id)
        cv2.putText(curr_frame_disp, label, self.text_location,
                    self.text_font, self.text_font_size, self.text_color, 1, self.text_line_type)

        out_images = {}
        for obj_type in self.obj_types:
            out_images[obj_type] = None

        if frame_data[ObjTypes.tracking_result] is not None:
            """Tracking results
            """
            # _curr_frame = None
            # traj_data = None

            tracked_frame = np.copy(curr_frame_disp)
            # if self.show_lost:
            #     lost_frame = np.copy(curr_frame_disp)

            res_data = frame_data[ObjTypes.tracking_result]
            if res_data.shape[1] > 10:
                states = res_data[:, 10].astype(np.int32)
            else:
                states = np.full((res_data.shape[0],), MDPStates.tracked, dtype=np.int32)

            if res_data.shape[1] > 11:
                is_valid = res_data[:, 11].astype(np.int32)
            else:
                is_valid = np.full((res_data.shape[0],), 1, dtype=np.int32)

            """check for duplicate target IDs"""
            target_ids = res_data[:, 1].astype(np.int32)
            u, c = np.unique(target_ids, return_counts=True)
            dup = u[c > 1]

            assert not dup, "Duplicate target IDs in the same frame found"

            for res_id in range(res_data.shape[0]):
                target_id = int(res_data[res_id, 1])
                state = states[res_id]
                is_dotted = 0
                traj_data = self._traj_data[ObjTypes.tracking_result]

                if self._params.show_trajectory and deleted_targets is not None:
                    for _id in deleted_targets:
                        if _id in traj_data:
                            del traj_data[_id]

                        # if _id in self.traj_data['lost']:
                        #     del self.traj_data['lost'][_id]

                colors = self._params.tracked_cols
                # _curr_frame = tracked_frame

                if state == MDPStates.lost:
                    is_dotted = 1
                    # if not self.show_lost:
                    #     continue
                    # colors = self._params.lost_cols
                    # traj_data = self.traj_data['lost']
                    # _curr_frame = lost_frame
                elif state == MDPStates.inactive:
                    if not self.show_inactive:
                        continue
                    colors = self._params.inactive_cols

                col_id = (target_id - 1) % len(colors)

                if not is_valid[res_id]:
                    if not self._params.show_invalid:
                        continue
                    color = 'black'
                    traj_thickness = 1
                    box_thickness = 1
                else:
                    color = colors[col_id]
                    traj_thickness = self._params.traj_thickness
                    box_thickness = self._params.box_thickness

                if self._params.show_trajectory:
                    if target_id not in traj_data.keys():
                        traj_data[target_id] = []
                    # lines joining the centers of the bottom edges of the bounding boxes usually look best
                    obj_center = np.array([res_data[res_id, 2] + res_data[res_id, 4] / 2.0,
                                           res_data[res_id, 3] + res_data[res_id, 5]])
                    traj_data[target_id].append(obj_center)
                    draw_trajectory(tracked_frame, traj_data[target_id], color=color,
                                    thickness=traj_thickness, is_dotted=is_dotted)
                draw_box(tracked_frame, res_data[res_id, 2:6], color=color, _id=target_id,
                         thickness=box_thickness, is_dotted=is_dotted)
            if gates is not None:
                for _id, gate in gates.viewitems():
                    p1 = (int(gate.x0), int(gate.y0))
                    p2 = (int(gate.x1), int(gate.y1))
                    cv2.line(tracked_frame, p1, p2, self.gate_col, self.gate_thickness)
                    gate_text = str(_id)
                    n_intersections = len(gate.intersections)
                    if n_intersections > 0:
                        gate_text = '{:s}:{:d}'.format(gate_text, n_intersections)
                    cv2.putText(tracked_frame, gate_text, (int(p1[0] - 1), int(p1[1] - 1)),
                                self.gate_font, self.gate_font_size, self.gate_col,
                                self.gate_font_thickness, self.text_line_type)

            out_images[ObjTypes.tracking_result] = tracked_frame

            if not return_mode:

                if self._params.disp_size:
                    tracked_frame = resize_ar(tracked_frame, *self._params.disp_size)
                elif self._params.resize_factor != 1:
                    tracked_frame = cv2.resize(tracked_frame, (0, 0), fx=self._params.resize_factor,
                                               fy=self._params.resize_factor)
                # if self.show_lost:
                #     if self._params.disp_size:
                #         lost_frame = resize_ar(lost_frame, *self._params.disp_size)
                #     elif self._params.resize_factor != 1:
                #         lost_frame = cv2.resize(lost_frame, (0, 0), fx=self._params.resize_factor,
                #                                 fy=self._params.resize_factor)
                #     res_frame = np.concatenate((tracked_frame, lost_frame), axis=0)
                # else:

                res_frame = tracked_frame

                if self._writer[ObjTypes.tracking_result] is not None:
                    self._writer[ObjTypes.tracking_result].write(res_frame)

                if self._params.show:
                    if msg is not None:
                        res_frame = annotate_and_show('', res_frame, msg, n_modules=0, only_annotate=1)
                    self._siif = SIIF.imshow(self._res_win_title, res_frame)

                self._mode[ObjTypes.tracking_result] = 1

        if frame_data[ObjTypes.detection] is not None:
            """Detections
            """
            det_frame = np.copy(curr_frame_disp)
            det_data = frame_data[ObjTypes.detection]
            for det_id in range(det_data.shape[0]):
                draw_box(det_frame, det_data[det_id, 2:6], color=self._params.det_cols[0],
                         thickness=self._params.box_thickness)

            out_images[ObjTypes.detection] = det_frame

            if not return_mode:
                if self._params.disp_size:
                    det_frame = resize_ar(det_frame, *self._params.disp_size)
                elif self._params.resize_factor != 1:
                    det_frame = cv2.resize(det_frame, (0, 0), fx=self._params.resize_factor,
                                           fy=self._params.resize_factor)

                if self._writer[ObjTypes.detection] is not None:
                    self._writer[ObjTypes.detection].write(det_frame)
                if self._params.show:
                    self._siif = SIIF.imshow(self._det_win_title, det_frame)
                self._mode[ObjTypes.detection] = 1

        if frame_data[ObjTypes.annotation] is not None:
            """Annotations
            """
            has_occluded = 0
            ann_frame = np.copy(curr_frame_disp)
            ann_data = frame_data[ObjTypes.annotation]
            traj_data = self._traj_data[ObjTypes.annotation]
            for ann_id in range(ann_data.shape[0]):
                target_id = int(ann_data[ann_id, 1])
                col_id = (target_id - 1) % len(self._params.ann_cols)
                is_dotted = 0
                if ann_data.shape[1] > 10 and ann_data[ann_id, 10] == 1:
                    """occluded"""
                    is_dotted = 1
                    has_occluded = 1

                if self._params.show_trajectory:
                    if target_id not in traj_data.keys():
                        traj_data[target_id] = []
                    obj_center = np.array([ann_data[ann_id, 2] + ann_data[ann_id, 4] / 2.0,
                                           ann_data[ann_id, 3] + ann_data[ann_id, 5]])
                    traj_data[target_id].append(obj_center)
                    draw_trajectory(ann_frame, traj_data[target_id], color=self._params.ann_cols[col_id],
                                    thickness=self._params.traj_thickness, is_dotted=is_dotted)
                draw_box(ann_frame, ann_data[ann_id, 2:6], color=self._params.ann_cols[col_id],
                         _id=target_id, thickness=self._params.box_thickness, is_dotted=is_dotted)

            out_images[ObjTypes.annotation] = ann_frame

            if not return_mode:
                if self._params.disp_size:
                    ann_frame = resize_ar(ann_frame, *self._params.disp_size)
                elif self._params.resize_factor != 1:
                    ann_frame = cv2.resize(ann_frame, (0, 0), fx=self._params.resize_factor,
                                           fy=self._params.resize_factor)

                if self._writer[ObjTypes.annotation] is not None:
                    self._writer[ObjTypes.annotation].write(ann_frame)
                if self._params.show:
                    self._siif = SIIF.imshow(self._ann_win_title, ann_frame)
                self._mode[ObjTypes.annotation] = 1

            if has_occluded:
                pause = 1

        if return_mode:
            return out_images

        if failed_targets:
            failed_target_frame = np.copy(curr_frame_disp)
            for failed_target in failed_targets:
                draw_box(failed_target_frame, failed_target, color='black',
                         _id=0, thickness=self._params.box_thickness)
            if self._params.disp_size:
                failed_target_frame = resize_ar(failed_target_frame, *self._params.disp_size)
            elif self._params.resize_factor != 1:
                failed_target_frame = cv2.resize(failed_target_frame, (0, 0), fx=self._params.resize_factor,
                                                 fy=self._params.resize_factor)
            self._siif = SIIF.imshow(self._failed_targets_win_title, failed_target_frame)

        if self._siif:
            return True

        key = cv2.waitKey(1 - self._pause_after_frame) % 256
        if key == 27:
            sys.exit()
        elif key == 32:
            self._pause_after_frame = 1 - self._pause_after_frame
        return True

    def close(self):
        if self._params.show and not self._siif:
            if self._mode[ObjTypes.tracking_result]:
                cv2.destroyWindow(self._res_win_title)
            if self._mode[ObjTypes.detection]:
                cv2.destroyWindow(self._det_win_title)
            if self._mode[ObjTypes.annotation]:
                cv2.destroyWindow(self._ann_win_title)

        for _, w in self._writer.items():
            if w is not None:
                w.release()
