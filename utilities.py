def x11_available():
    from subprocess import Popen, PIPE
    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0


from sys import platform

if platform in ("linux", "linux2") and not x11_available():
    """get rid of annoying error on ssh:
    Unable to init server: Could not connect: Connection refused
    Gdk-CRITICAL **: gdk_cursor_new_for_display: assertion 'GDK_IS_DISPLAY (display)' failed
    """
    import matplotlib as mpl

    mpl.use('Agg')

import numpy as np
import pandas as pd
import cv2
import subprocess
import filecmp
import os
import shutil
import copy
import sys
import time
import random
from ast import literal_eval
from pprint import pformat
import json
import math
import functools
import inspect
import logging
from io import StringIO
from contextlib import contextmanager
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
import psutil
import socket

from colorlog import ColoredFormatter

from zipfile import ZipFile

import paramiko

logging.getLogger("paramiko").setLevel(logging.WARNING)

import paramparse

"""BGR values for different colors"""
col_bgr = {
    'snow': (250, 250, 255),
    'snow_2': (233, 233, 238),
    'snow_3': (201, 201, 205),
    'snow_4': (137, 137, 139),
    'ghost_white': (255, 248, 248),
    'white_smoke': (245, 245, 245),
    'gainsboro': (220, 220, 220),
    'floral_white': (240, 250, 255),
    'old_lace': (230, 245, 253),
    'linen': (230, 240, 240),
    'antique_white': (215, 235, 250),
    'antique_white_2': (204, 223, 238),
    'antique_white_3': (176, 192, 205),
    'antique_white_4': (120, 131, 139),
    'papaya_whip': (213, 239, 255),
    'blanched_almond': (205, 235, 255),
    'bisque': (196, 228, 255),
    'bisque_2': (183, 213, 238),
    'bisque_3': (158, 183, 205),
    'bisque_4': (107, 125, 139),
    'peach_puff': (185, 218, 255),
    'peach_puff_2': (173, 203, 238),
    'peach_puff_3': (149, 175, 205),
    'peach_puff_4': (101, 119, 139),
    'navajo_white': (173, 222, 255),
    'moccasin': (181, 228, 255),
    'cornsilk': (220, 248, 255),
    'cornsilk_2': (205, 232, 238),
    'cornsilk_3': (177, 200, 205),
    'cornsilk_4': (120, 136, 139),
    'ivory': (240, 255, 255),
    'ivory_2': (224, 238, 238),
    'ivory_3': (193, 205, 205),
    'ivory_4': (131, 139, 139),
    'lemon_chiffon': (205, 250, 255),
    'seashell': (238, 245, 255),
    'seashell_2': (222, 229, 238),
    'seashell_3': (191, 197, 205),
    'seashell_4': (130, 134, 139),
    'honeydew': (240, 255, 240),
    'honeydew_2': (224, 238, 244),
    'honeydew_3': (193, 205, 193),
    'honeydew_4': (131, 139, 131),
    'mint_cream': (250, 255, 245),
    'azure': (255, 255, 240),
    'alice_blue': (255, 248, 240),
    'lavender': (250, 230, 230),
    'lavender_blush': (245, 240, 255),
    'misty_rose': (225, 228, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'dark_slate_gray': (79, 79, 49),
    'dim_gray': (105, 105, 105),
    'slate_gray': (144, 138, 112),
    'light_slate_gray': (153, 136, 119),
    'gray': (190, 190, 190),
    'light_gray': (211, 211, 211),
    'midnight_blue': (112, 25, 25),
    'navy': (128, 0, 0),
    'cornflower_blue': (237, 149, 100),
    'dark_slate_blue': (139, 61, 72),
    'slate_blue': (205, 90, 106),
    'medium_slate_blue': (238, 104, 123),
    'light_slate_blue': (255, 112, 132),
    'medium_blue': (205, 0, 0),
    'royal_blue': (225, 105, 65),
    'blue': (255, 0, 0),
    'dodger_blue': (255, 144, 30),
    'deep_sky_blue': (255, 191, 0),
    'sky_blue': (250, 206, 135),
    'light_sky_blue': (250, 206, 135),
    'steel_blue': (180, 130, 70),
    'light_steel_blue': (222, 196, 176),
    'light_blue': (230, 216, 173),
    'powder_blue': (230, 224, 176),
    'pale_turquoise': (238, 238, 175),
    'dark_turquoise': (209, 206, 0),
    'medium_turquoise': (204, 209, 72),
    'turquoise': (208, 224, 64),
    'cyan': (255, 255, 0),
    'light_cyan': (255, 255, 224),
    'cadet_blue': (160, 158, 95),
    'medium_aquamarine': (170, 205, 102),
    'aquamarine': (212, 255, 127),
    'dark_green': (0, 100, 0),
    'dark_olive_green': (47, 107, 85),
    'dark_sea_green': (143, 188, 143),
    'sea_green': (87, 139, 46),
    'medium_sea_green': (113, 179, 60),
    'light_sea_green': (170, 178, 32),
    'pale_green': (152, 251, 152),
    'spring_green': (127, 255, 0),
    'lawn_green': (0, 252, 124),
    'chartreuse': (0, 255, 127),
    'medium_spring_green': (154, 250, 0),
    'green_yellow': (47, 255, 173),
    'lime_green': (50, 205, 50),
    'yellow_green': (50, 205, 154),
    'forest_green': (34, 139, 34),
    'olive_drab': (35, 142, 107),
    'dark_khaki': (107, 183, 189),
    'khaki': (140, 230, 240),
    'pale_goldenrod': (170, 232, 238),
    'light_goldenrod_yellow': (210, 250, 250),
    'light_yellow': (224, 255, 255),
    'yellow': (0, 255, 255),
    'gold': (0, 215, 255),
    'light_goldenrod': (130, 221, 238),
    'goldenrod': (32, 165, 218),
    'dark_goldenrod': (11, 134, 184),
    'rosy_brown': (143, 143, 188),
    'indian_red': (92, 92, 205),
    'saddle_brown': (19, 69, 139),
    'sienna': (45, 82, 160),
    'peru': (63, 133, 205),
    'burlywood': (135, 184, 222),
    'beige': (220, 245, 245),
    'wheat': (179, 222, 245),
    'sandy_brown': (96, 164, 244),
    'tan': (140, 180, 210),
    'chocolate': (30, 105, 210),
    'firebrick': (34, 34, 178),
    'brown': (42, 42, 165),
    'dark_salmon': (122, 150, 233),
    'salmon': (114, 128, 250),
    'light_salmon': (122, 160, 255),
    'orange': (0, 165, 255),
    'dark_orange': (0, 140, 255),
    'coral': (80, 127, 255),
    'light_coral': (128, 128, 240),
    'tomato': (71, 99, 255),
    'orange_red': (0, 69, 255),
    'red': (0, 0, 255),
    'hot_pink': (180, 105, 255),
    'deep_pink': (147, 20, 255),
    'pink': (203, 192, 255),
    'light_pink': (193, 182, 255),
    'pale_violet_red': (147, 112, 219),
    'maroon': (96, 48, 176),
    'medium_violet_red': (133, 21, 199),
    'violet_red': (144, 32, 208),
    'violet': (238, 130, 238),
    'plum': (221, 160, 221),
    'orchid': (214, 112, 218),
    'medium_orchid': (211, 85, 186),
    'dark_orchid': (204, 50, 153),
    'dark_violet': (211, 0, 148),
    'blue_violet': (226, 43, 138),
    'purple': (240, 32, 160),
    'medium_purple': (219, 112, 147),
    'thistle': (216, 191, 216),
    'green': (0, 255, 0),
    'magenta': (255, 0, 255)
}

from paramparse import MultiPath, MultiCFG


class BaseParams:
    tee_log = ''


class MDPStates:
    inactive, active, tracked, lost = range(4)
    to_str = {
        0: 'inactive',
        1: 'active',
        2: 'tracked',
        3: 'lost',
    }
    # inactive, active, tracked, lost = ('inactive', 'active', 'tracked', 'lost')


class TrackingStatus:
    success, failure, unstable = range(1, 4)
    to_str = {
        1: 'success',
        2: 'failure',
        3: 'unstable',
    }


class AnnotationStatus:
    types = (
        'combined',
        'fp_background',
        'fp_deleted',
        'fp_apart',
        'fp_concurrent',
        'tp',
    )
    combined, fp_background, fp_deleted, fp_apart, fp_concurrent, tp = types


class SaveModes:
    valid = range(3)
    none, all, error = valid
    to_str = {
        0: 'none',
        1: 'all',
        2: 'error',
    }


def clamp(list_x, minx, maxx):
    return [max(min(x, maxx), minx) for x in list_x]


def get_class_idx(labels):
    pos_mask = labels == 1
    neg_mask = np.logical_not(pos_mask)
    pos_idx, neg_idx = np.squeeze(np.argwhere(pos_mask)), np.squeeze(np.argwhere(neg_mask))

    n_pos_samples, n_neg_samples = pos_idx.size, neg_idx.size
    # class_weights = [float(n_neg_samples) / n_samples, float(n_pos_samples) / n_samples]

    return pos_idx, neg_idx, n_pos_samples, n_neg_samples


def get_nested_idx(all_n_samples):
    linear_to_nested_idx = {}

    start_sample_id = 0

    for _id, _n_samples in enumerate(all_n_samples):
        end_sample_id = start_sample_id + _n_samples

        linear_idx = list(range(start_sample_id, end_sample_id))
        nested_idx = [(_id, k) for k in range(_n_samples)]

        temp_dict = dict(zip(linear_idx, nested_idx))

        linear_to_nested_idx.update(temp_dict)

        start_sample_id = end_sample_id

    return linear_to_nested_idx


def disable_vis(obj, args_in=None, prefix=''):
    """

    :param obj:
    :param list args_in:
    :param str prefix:
    :return:
    """
    if args_in is not None:
        for i, _arg in enumerate(args_in):
            if '+=' in _arg:
                _sep = '+='
            else:
                _sep = '='
            _arg_name, _arg_val = _arg.split(_sep)

            if _arg_name.endswith('.vis') or _arg_name.endswith('.visualize'):
                print(f'Disabling {_arg_name}')
                args_in[i] = '{}{}0'.format(_arg_name, _sep)

    obj_members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
    for member in obj_members:
        if member == 'help':
            continue
        member_val = getattr(obj, member)
        member_name = '{:s}.{:s}'.format(prefix, member) if prefix else member
        if not isinstance(member_val, (int, bool, float, str, tuple, list, dict,
                                       paramparse.MultiCFG, paramparse.MultiPath)):
            disable_vis(member_val, prefix=member_name)
        else:
            if member in ('vis', 'visualize') and isinstance(member_val, (int, bool)) and member_val:
                print(f'Disabling {member_name}')
                setattr(obj, member, 0)


def set_recursive(obj, name, val, prefix='', check_existence=1):
    """

    :param obj:
    :param list args_in:
    :param str prefix:
    :param check_existence int:
    :return:
    """
    if obj is None:
        return

    if not check_existence or hasattr(obj, name):
        setattr(obj, name, val)

    obj_members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]

    for member in obj_members:
        if member == 'help':
            continue

        member_val = getattr(obj, member)
        member_name = '{:s}.{:s}'.format(prefix, member) if prefix else member
        if not isinstance(member_val, (int, bool, float, str, tuple, list, dict,
                                       paramparse.MultiCFG, paramparse.MultiPath)):
            set_recursive(member_val, name, val, prefix=member_name, check_existence=check_existence)


def list_to_str(vals, fmt='', sep='\t'):
    """

    :param list vals:
    :param fmt:
    :param sep:
    :return:
    """
    type_to_fmt = {
        int: '%d',
        float: '%.3f',
        bool: '%r',
        str: '%s',
    }
    if fmt:
        fmts = [fmt, ] * len(vals)
    else:
        fmts = []
        for val in vals:
            try:
                fmt = type_to_fmt[type(val)]
            except KeyError:
                fmt = type_to_fmt[type(val.item())]
            fmts.append(fmt)

    return sep.join(fmt % val for fmt, val in zip(fmts, vals))


def dict_to_str(vals, fmt='%.3f', sep='\t'):
    """

    :param dict vals:
    :param fmt:
    :param sep:
    :return:
    """
    return sep.join('{}: '.format(key) + fmt % val for key, val in vals.items())


PY_DIST = -1


class CVConstants:
    similarity_types = {
        -1: PY_DIST,
        0: cv2.TM_CCOEFF_NORMED,
        1: cv2.TM_SQDIFF_NORMED,
        2: cv2.TM_CCORR_NORMED,
        3: cv2.TM_CCOEFF,
        4: cv2.TM_SQDIFF,
        5: cv2.TM_CCORR
    }
    interp_types = {
        0: cv2.INTER_NEAREST,
        1: cv2.INTER_LINEAR,
        2: cv2.INTER_AREA,
        3: cv2.INTER_CUBIC,
        4: cv2.INTER_LANCZOS4
    }
    fonts = {
        0: cv2.FONT_HERSHEY_SIMPLEX,
        1: cv2.FONT_HERSHEY_PLAIN,
        2: cv2.FONT_HERSHEY_DUPLEX,
        3: cv2.FONT_HERSHEY_COMPLEX,
        4: cv2.FONT_HERSHEY_TRIPLEX,
        5: cv2.FONT_HERSHEY_COMPLEX_SMALL,
        6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    }
    line_types = {
        0: cv2.LINE_4,
        1: cv2.LINE_8,
        2: cv2.LINE_AA,
    }


class CustomLogger:
    """
    :type _backend: logging.RootLogger | logging.logger

    """

    def __init__(self, logger, names, key='custom_module'):
        """
        modify the custom module name header to append one or more names

        :param CustomLogger | logging.RootLogger logger:
        :param tuple | list names:
        """
        try:
            self._backend = logger.get_backend()
        except AttributeError:
            self._backend = logger

        self.handlers = self._backend.handlers
        self.addHandler = self._backend.addHandler
        self.removeHandler = self._backend.removeHandler

        try:
            k = logger.info.keywords['extra'][key]
        except BaseException as e:
            custom_log_header_str = '{}'.format(':'.join(names))
        else:
            custom_log_header_str = '{}:{}'.format(k, ':'.join(names))

        self.custom_log_header_tokens = custom_log_header_str.split(':')

        try:
            custom_log_header = copy.deepcopy(logger.info.keywords['extra'])
        except BaseException as e:
            custom_log_header = {}

        custom_log_header.update({key: custom_log_header_str})

        self.info = functools.partial(self._backend.info, extra=custom_log_header)
        self.warning = functools.partial(self._backend.warning, extra=custom_log_header)
        self.debug = functools.partial(self._backend.debug, extra=custom_log_header)
        self.error = functools.partial(self._backend.error, extra=custom_log_header)

        # try:
        #     self.info = functools.partial(self._backend.info.func, extra=custom_log_header)
        # except BaseException as e:
        #     self.info = functools.partial(self._backend.info, extra=custom_log_header)
        #     self.warning = functools.partial(self._backend.warning, extra=custom_log_header)
        #     self.debug = functools.partial(self._backend.debug, extra=custom_log_header)
        #     self.error = functools.partial(self._backend.error, extra=custom_log_header)
        # else:
        #     self.warning = functools.partial(self._backend.warning.func, extra=custom_log_header)
        #     self.debug = functools.partial(self._backend.debug.func, extra=custom_log_header)
        #     self.error = functools.partial(self._backend.error.func, extra=custom_log_header)

    def get_backend(self):
        return self._backend

    @staticmethod
    def add_file_handler(log_dir, _prefix, logger):
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        log_file = linux_path(log_dir, '{}_{}.log'.format(_prefix, time_stamp))
        logging_handler = logging.FileHandler(log_file)
        logger.addHandler(logging_handler)
        logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s',
        )

        logger.handlers[-1].setFormatter(logging_fmt)
        return log_file, logging_handler

    # @staticmethod
    # def add_string_handler(logger):
    #     log_stream = StringIO()
    #     logging_handler = logging.StreamHandler(log_stream)
    #     # logging_handler.setFormatter(logger.handlers[0].formatter)
    #     logger.addHandler(logging_handler)
    #     # logger.string_stream = log_stream
    #     return logging_handler

    @staticmethod
    def remove_file_handler(logging_handler, logger):
        if logging_handler not in logger.handlers:
            return
        logging_handler.close()
        logger.removeHandler(logging_handler)

    @staticmethod
    def setup(name):
        # PROFILE_LEVEL_NUM = 9
        #
        # def profile(self, message, *args, **kws):
        #     if self.isEnabledFor(PROFILE_LEVEL_NUM):
        #         self._log(PROFILE_LEVEL_NUM, message, args, **kws)

        # logging.addLevelName(PROFILE_LEVEL_NUM, "PROFILE")
        # logging.Logger.profile = profile
        # logging.getLogger().addHandler(ColorHandler())

        # logging_level = logging.DEBUG
        # logging_level = PROFILE_LEVEL_NUM
        # logging.basicConfig(level=logging_level, format=logging_fmt)

        colored_logging_fmt = ColoredFormatter(
            '%(header_log_color)s%(custom_header)s:%(log_color)s%(custom_module)s:%(funcName)s:%(lineno)s : '
            ' %(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={
                'header': {
                    'DEBUG': 'white',
                    'INFO': 'white',
                    'WARNING': 'white',
                    'ERROR': 'white',
                    'CRITICAL': 'white',
                }
            },
            style='%'
        )

        nocolor_logging_fmt = logging.Formatter(
            '%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s  :::  %(message)s',
        )
        # nocolor_logging_fmt = colored_logging_fmt

        # logging_fmt = logging.Formatter('%(custom_header)s:%(custom_module)s:%(funcName)s:%(lineno)s :  %(message)s')
        # logging_fmt = logging.Formatter('%(levelname)s::%(module)s::%(funcName)s::%(lineno)s :  %(message)s')

        logging_level = logging.DEBUG
        logging.basicConfig(level=logging_level,
                            # format=colored_logging_fmt
                            )

        _logger = logging.getLogger(name)

        if _logger.hasHandlers():
            _logger.handlers.clear()

        _logger.setLevel(logging_level)

        col_handler = logging.StreamHandler()
        col_handler.setFormatter(colored_logging_fmt)
        col_handler.setLevel(logging_level)
        _logger.addHandler(col_handler)

        log_stream = StringIO()
        nocol_handler = logging.StreamHandler(log_stream)
        nocol_handler.setFormatter(nocolor_logging_fmt)
        nocol_handler.setLevel(logging_level)
        _logger.addHandler(nocol_handler)

        # CustomLogger.add_string_handler(_logger)

        class ContextFilter(logging.Filter):
            def filter(self, record):

                if not hasattr(record, 'custom_module'):
                    record.custom_module = record.module

                if not hasattr(record, 'custom_header'):
                    record.custom_header = record.levelname

                return True

        f = ContextFilter()
        _logger.addFilter(f)

        """avoid duplicate logging when logging used by other libraries"""
        _logger.propagate = False
        return _logger


@contextmanager
def profile(_id, _times=None, _rel_times=None, _fps=None, enable=1, show=1):
    """

    :param _id:
    :param dict _times:
    :param int enable:
    :return:
    """
    if not enable:
        yield None

    else:
        start_t = time.time()
        yield None
        end_t = time.time()
        _time = end_t - start_t

        if show:
            print(f'{_id} :: {_time}')

        if _fps is not None:
            if _time > 0:
                _fps[_id] = 1.0 / _time
            else:
                _fps[_id] = np.inf

        if _times is not None:

            _times[_id] = _time

            total_time = np.sum(list(_times.values()))

            if _rel_times is not None:

                for __id in _times:
                    rel__time = _times[__id] / total_time
                    _rel_times[__id] = rel__time

                rel_times = [(k, v) for k, v in sorted(_rel_times.items(), key=lambda item: item[1])]

                print(f'rel_times:\n {pformat(rel_times)}')


class VideoWriterGPU:
    def __init__(self, path, fps, size):
        self._path = path
        width, height = size

        command = ['ffmpeg',
                   '-y',
                   '-f', 'rawvideo',
                   '-codec', 'rawvideo',
                   '-s', f'{width}x{height}',  # size of one frame
                   '-pix_fmt', 'rgb24',
                   '-r', f'{fps}',  # frames per second
                   '-i', '-',  # The input comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-c:v', 'libx265',
                   # '-preset', 'medium',
                   '-preset', 'veryslow',
                   '-x265-params', 'lossless=0',
                   '-hide_banner',
                   '-loglevel', 'panic',

                   f'{self._path}']

        self._pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

        self._frame_id = 0

    def isOpened(self):
        if self._pipe is None:
            return False
        return True

    def write(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im = image_rgb.tostring()
        self._pipe.stdin.write(im)
        self._pipe.stdin.flush()

    def release(self):
        self._pipe.stdin.close()
        self._pipe.wait()


class ImageSequenceCapture:
    """
    :param str src_path
    :param int recursive
    """

    def __init__(self, src_path='', recursive=0, img_exts=(), logger=None):
        self.src_path = ''
        self.src_fmt = ''
        self.recursive = 0
        self.img_exts = ('.jpg', '.bmp', '.jpeg', '.png', '.tif', '.tiff', '.webp')
        self.src_files = []
        self.n_src_files = 0
        self.is_open = False
        self.frame_id = 0

        if src_path:
            if self.open(src_path, recursive, img_exts):
                self.is_open = True

    def is_opened(self, cv_prop):
        return self.is_open

    def read_iter(self, starting_frame=0, length=1, new_size=None):
        # start the iters
        cur_frame = starting_frame
        while (cur_frame + length) < self.n_src_files:
            frames = []
            for i in range(length):
                frame = cv2.imread(self.src_files[self.frame_id])

                if new_size is not None:
                    frame = cv2.resize(frame, new_size)

                frames.append(frame)

            cur_frame += length
            yield frames

    def read(self):
        if self.frame_id >= self.n_src_files:
            raise IOError('Invalid frame_id: {} for sequence with {} frames'.format(
                self.frame_id, self.n_src_files
            ))
        frame = cv2.imread(self.src_files[self.frame_id])
        self.frame_id += 1
        return True, frame

    def set(self, cv_prop, _id):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            print('Setting frame_id to : {}'.format(_id))
            self.frame_id = _id

    def get(self, cv_prop):
        if cv_prop == cv2.CAP_PROP_POS_FRAMES:
            return self.frame_id

    def open(self, src_path='', recursive=0, img_exts=()):
        if src_path:
            img_ext = os.path.splitext(os.path.basename(src_path))[1]
            if img_ext:
                self.src_path = os.path.dirname(src_path)
                self.src_fmt = os.path.basename(src_path)
                self.img_exts = (img_ext,)
            else:
                self.src_path = src_path
                self.src_fmt = ''

            self.recursive = recursive
        if img_exts:
            self.img_exts = img_exts

        if recursive:
            src_file_gen = [[linux_path(dirpath, f) for f in filenames if
                             os.path.splitext(f.lower())[1] in self.img_exts]
                            for (dirpath, dirnames, filenames) in os.walk(self.src_path, followlinks=True)]
            _src_files = [item for sublist in src_file_gen for item in sublist]
        else:
            _src_files = [linux_path(self.src_path, k) for k in os.listdir(self.src_path) if
                          os.path.splitext(k.lower())[1] in self.img_exts]

        if not _src_files:
            print('No images found in {}'.format(self.src_path))
            return False

        _src_files = [os.path.abspath(k) for k in _src_files]
        _src_files.sort(key=sort_key)

        self.src_files = _src_files
        self.n_src_files = len(self.src_files)

        if self.src_fmt:
            matching_files = [self.src_fmt % i for i in range(1, self.n_src_files + 1)]
            self.src_files = [k for k in self.src_files if os.path.basename(k) in matching_files]
            self.n_src_files = len(self.src_files)
        print('Found {} images in {}'.format(self.n_src_files, self.src_path))
        return True


class DebugParams:
    """
    :type write_state_info: bool | int
    :type write_to_bin: bool | int
    :type write_thresh: (int, int)
    :type cmp_root_dirs: (str, str)
    """

    def __init__(self):
        self.write_state_info = 0
        self.write_thresh = (0, 0)
        self.write_to_bin = 1
        self.memory_tracking = 0
        self.cmp_root_dirs = ('../../isl_labelling_tool/tracking_module/log', 'log')
        self.help = {
            'write_state_info': 'write matrices containing the target state information to files '
                                'on disk (for debugging purposes)',
            'write_thresh': 'two element tuple to indicate the minimum (iter_id, frame_id) after which '
                            'to start writing and comparing state info',
            'write_to_bin': 'write the matrices to binary files instead of human readable ASCII text files',
            'memory_tracking': 'track memory usage to find leaks',
            'cmp_root_dirs': 'root directories where the data files to be compared are written',
        }


# overlaps between two sets of labeled objects, typically the annotations and the detections
class CrossOverlaps:
    """
    :type iou: list[np.ndarray]
    :type ioa_1: list[np.ndarray]
    :type ioa_2: list[np.ndarray]
    :type max_iou_1: np.ndarray
    :type max_iou_1_idx: np.ndarray
    :type max_iou_2: np.ndarray
    :type max_iou_2_idx: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object 1
        self.ioa_1 = None
        # intersection over area of object 2
        self.ioa_2 = None
        # max iou of each object in first set over all objects in second set from the same frame
        self.max_iou_1 = None
        # index of the object in the second set that corresponds to the maximum iou
        self.max_iou_1_idx = None
        # max iou of each object in second set over all objects in first set from the same frame
        self.max_iou_2 = None
        # index of the object in the first set that corresponds to the maximum iou
        self.max_iou_2_idx = None

    def compute(self, objects_1, objects_2, index_1, index_2, n_frames):
        """
        :type objects_1: np.ndarray
        :type objects_2: np.ndarray
        :type index_1: list[np.ndarray]
        :type index_2: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        # for each frame, contains a matrix that stores the overlap between each pair of
        # annotations and detections in that frame
        self.iou = [None] * n_frames
        self.ioa_1 = [None] * n_frames
        self.ioa_2 = [None] * n_frames

        self.max_iou_1 = np.zeros((objects_1.shape[0],))
        self.max_iou_2 = np.zeros((objects_2.shape[0],))

        self.max_iou_1_idx = np.full((objects_1.shape[0],), -1, dtype=np.int32)
        self.max_iou_2_idx = np.full((objects_2.shape[0],), -1, dtype=np.int32)

        for frame_id in range(n_frames):
            idx1 = index_1[frame_id]
            idx2 = index_2[frame_id]

            if idx1 is None or idx2 is None:
                continue

            boxes_1 = objects_1[idx1, :]
            n1 = boxes_1.shape[0]
            ul_1 = boxes_1[:, :2]  # n1 x 2
            size_1 = boxes_1[:, 2:]  # n1 x 2
            br_1 = ul_1 + size_1 - 1  # n1 x 2
            area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1

            boxes_2 = objects_2[idx2, :]
            n2 = boxes_2.shape[0]
            ul_2 = boxes_2[:, :2]  # n2 x 2
            size_2 = boxes_2[:, 2:]  # n2 x 2
            br_2 = ul_2 + size_2 - 1  # n2 x 2
            area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1

            ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            ul_inter = np.maximum(ul_1_rep, ul_2_rep)  # n2 x 2 x n1

            # box size is defined in terms of  no. of pixels
            br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            br_inter = np.minimum(br_1_rep, br_2_rep)  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

            size_inter = br_inter - ul_inter + 1  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
            size_inter[size_inter < 0] = 0  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1]).reshape((n1, n2))

            area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
            area_union = area_1_rep + area_2_rep - area_inter  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep).reshape((n1, n2), order='F')  # n1 x n2
            # self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep).reshape((n1, n2), order='F')  # n1 x n2

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n1 x n2
            self.ioa_1[frame_id] = np.divide(area_inter, area_1_rep)  # n1 x n2
            self.ioa_2[frame_id] = np.divide(area_inter, area_2_rep)  # n1 x n2

            max_idx_1 = np.argmax(self.iou[frame_id], axis=1)
            max_idx_2 = np.argmax(self.iou[frame_id], axis=0).transpose()

            self.max_iou_1[idx1] = self.iou[frame_id][np.arange(n1), max_idx_1]
            self.max_iou_2[idx2] = self.iou[frame_id][max_idx_2, np.arange(n2)]

            # indices wrt the overall object arrays rather than their frame-wise subsets
            self.max_iou_1_idx[idx1] = idx2[max_idx_1]
            self.max_iou_2_idx[idx2] = idx1[max_idx_2]


# overlaps between each labeled object in a set with all other objects in that set from the same frame
class SelfOverlaps:
    """
    :type iou: np.ndarray
    :type ioa: np.ndarray
    :type max_iou: np.ndarray
    :type max_ioa: np.ndarray
    """

    def __init__(self):
        # intersection over union
        self.iou = None
        # intersection over area of object
        self.ioa = None
        # max iou of each object over all other objects from the same frame
        self.max_iou = None
        # max ioa of each object over all other objects from the same frame
        self.max_ioa = None

        self.br = None
        self.areas = None

    def compute(self, objects, index, n_frames):
        """
        :type objects: np.ndarray
        :type index: list[np.ndarray]
        :type n_frames: int
        :rtype: None
        """
        self.iou = [None] * n_frames
        self.ioa = [None] * n_frames

        self.max_ioa = np.zeros((objects.shape[0],))
        self.areas = np.zeros((objects.shape[0],))
        self.br = np.zeros((objects.shape[0], 2))

        for frame_id in range(n_frames):
            if index[frame_id] is None:
                continue

            end_id = index[frame_id]
            boxes = objects[index[frame_id], :]

            n = boxes.shape[0]

            ul = boxes[:, :2]  # n x 2
            ul_rep = np.tile(np.reshape(ul, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_2_rep = np.tile(np.reshape(ul, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            ul_inter = np.maximum(ul_rep, ul_2_rep)  # n x 2 x n

            size = boxes[:, 2:]  # n1 x 2
            br = ul + size - 1  # n x 2

            # size_ = boxes[:, 2:]  # n x 2
            # br = ul + size_ - 1  # n x 2
            br_rep = np.tile(np.reshape(br, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_2_rep = np.tile(np.reshape(br, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
            br_inter = np.minimum(br_rep, br_2_rep)  # n x 2 x n

            size_inter = br_inter - ul_inter + 1  # np(n x n x 2) -> std(n x 2 x n)
            size_inter[size_inter < 0] = 0
            # np(n x n x 1) -> std(n x 1 x n)
            area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

            area = np.multiply(size[:, 0], size[:, 1]).reshape((n, 1))  # n1 x 1
            # area = np.multiply(size_[:, :, 0], size_[:, :, 1])  # n x 1
            area_rep = np.tile(area, (1, n))  # np(n x n x 1) -> std(n x 1 x n)
            area_2_rep = np.tile(area.transpose(), (n, 1))  # np(n x n x 1) -> std(n x 1 x n)
            area_union = area_rep + area_2_rep - area_inter  # np(n x n x 1) -> std(n x 1 x n)

            # self.iou[frame_id] = np.divide(area_inter, area_union).reshape((n, n), order='F')  # n x n
            # self.ioa[frame_id] = np.divide(area_inter, area_rep).reshape((n, n), order='F')  # n x n

            self.iou[frame_id] = np.divide(area_inter, area_union)  # n x n
            self.ioa[frame_id] = np.divide(area_inter, area_rep)  # n x n

            # set box overlap with itself to 0
            idx = np.arange(n)
            self.ioa[frame_id][idx, idx] = 0
            self.iou[frame_id][idx, idx] = 0

            for i in range(n):
                invalid_idx = np.flatnonzero(np.greater(br[i, 1], br[:, 1]))
                self.ioa[frame_id][i, invalid_idx] = 0

            self.max_ioa[index[frame_id]] = np.amax(self.ioa[frame_id], axis=1)

            self.areas[index[frame_id]] = area.reshape((n,))
            self.br[index[frame_id], :] = br


def find_associations(frame, file_dets, file_gts, iou_thresh, use_hungarian, vis=0):
    """

    :param np.ndarray file_dets:
    :param np.ndarray file_gts:
    :param float iou_thresh:
    :return:
    """

    n_file_dets = file_dets.shape[0]
    n_file_gt = file_gts.shape[0]

    det_gt_iou = np.zeros((n_file_dets, n_file_gt), dtype=np.float32)

    for _det_idx in range(n_file_dets):
        bb_det = file_dets[_det_idx, :]

        """min_x, min_y, w, h to min_x, min_y, max_x, max_y"""
        bb_det = [bb_det[0], bb_det[1], bb_det[0] + bb_det[2], bb_det[1] + bb_det[3]]
        for _gt_idx in range(n_file_gt):
            bb_gt = file_gts[_gt_idx, :]
            """min_x, min_y, w, h to min_x, min_y, max_x, max_y"""
            bb_gt = [bb_gt[0], bb_gt[1], bb_gt[0] + bb_gt[2], bb_gt[1] + bb_gt[3]]

            bi = [max(bb_det[0], bb_gt[0]), max(bb_det[1], bb_gt[1]), min(bb_det[2], bb_gt[2]),
                  min(bb_det[3], bb_gt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw <= 0 or ih <= 0:
                continue

            # compute overlap (IoU) = area of intersection / area of union
            ua = (bb_det[2] - bb_det[0] + 1) * (bb_det[3] - bb_det[1] + 1) + (
                    bb_gt[2] - bb_gt[0] + 1) * (
                         bb_gt[3] - bb_gt[1] + 1) - iw * ih
            det_gt_iou[_det_idx, _gt_idx] = iw * ih / ua

    det_to_gt = {}
    gt_to_det = {}

    unassociated_dets = list(range(n_file_dets))
    unassociated_gts = list(range(n_file_gt))

    associated_dets = []
    associated_gts = []

    if use_hungarian:
        from scipy.optimize import linear_sum_assignment

        det_gt_cost = 1 - det_gt_iou

        det_inds, gt_inds = linear_sum_assignment(det_gt_cost)

        for det_ind, gt_ind in zip(det_inds, gt_inds):
            det_to_gt[det_ind] = gt_ind
            gt_to_det[gt_ind] = det_ind

            associated_dets.append(det_ind)
            associated_gts.append(gt_ind)

            unassociated_dets.remove(det_ind)
            unassociated_gts.remove(gt_ind)
    else:

        det_gt_iou_copy = np.copy(det_gt_iou)

        """
         Assign detections to ground truth objects
        """
        while True:
            det_max_iou = np.argmax(det_gt_iou_copy, axis=1)
            gt_max_iou = np.argmax(det_gt_iou_copy, axis=0)

            assoc_found = 0

            for _det in unassociated_dets:
                _assoc_gt_id = det_max_iou[_det]
                _assoc_det_id = gt_max_iou[_assoc_gt_id]

                if _assoc_gt_id in associated_gts or \
                        _assoc_det_id != _det or \
                        det_gt_iou_copy[_assoc_det_id, _assoc_gt_id] < iou_thresh:
                    continue

                associated_dets.append(_assoc_det_id)
                associated_gts.append(_assoc_gt_id)

                unassociated_dets.remove(_assoc_det_id)
                unassociated_gts.remove(_assoc_gt_id)

                det_to_gt[_assoc_det_id] = _assoc_gt_id
                gt_to_det[_assoc_gt_id] = _assoc_det_id

                det_gt_iou_copy[_assoc_det_id, :] = -1
                det_gt_iou_copy[:, _assoc_gt_id] = -1

                assoc_found = 1

                break

            if not assoc_found:
                break

    if vis:
        ann_cols = ('green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
                    'dark_slate_gray', 'navy', 'turquoise')

        n_cols = len(ann_cols)
        frame_disp_gt = copy_rgb(frame)
        for _gt_idx in range(n_file_gt):
            draw_box(frame_disp_gt, file_gts[_gt_idx, :], color=ann_cols[_gt_idx % n_cols], _id=_gt_idx)

        frame_disp_det = copy_rgb(frame)
        for _det_idx in associated_dets:
            _gt_idx = det_to_gt[_det_idx]
            draw_box(frame_disp_det, file_dets[_det_idx, :], color=ann_cols[_gt_idx % n_cols], _id=_gt_idx)

        for _det_idx in unassociated_dets:
            draw_box(frame_disp_det, file_dets[_det_idx, :], color='black', is_dotted=1)

        annotate_and_show('gt - det', [frame_disp_gt, frame_disp_det], grid_size=(1, 2))

    return det_to_gt, gt_to_det, unassociated_dets, unassociated_gts


def compute_overlaps_multi(iou, ioa_1, ioa_2, objects_1, objects_2, logger=None):
    """

    compute overlap between each pair of objects in two sets of objects
    can be used for computing overlap between all detections and annotations in a frame

    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :type logger: logging.RootLogger | None
    :rtype: None
    """
    # handle annoying singletons
    if len(objects_1.shape) == 1:
        objects_1 = objects_1.reshape((1, 4))

    if len(objects_2.shape) == 1:
        objects_2 = objects_2.reshape((1, 4))

    n1 = objects_1.shape[0]
    n2 = objects_2.shape[0]

    ul_1 = objects_1[:, :2]  # n1 x 2
    ul_1_rep = np.tile(np.reshape(ul_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    ul_2 = objects_2[:, :2]  # n2 x 2
    ul_2_rep = np.tile(np.reshape(ul_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_1 = objects_1[:, 2:]  # n1 x 2
    size_2 = objects_2[:, 2:]  # n2 x 2

    # if logger is not None:
    #     logger.debug('objects_1.shape: %(1)s', {'1': objects_1.shape})
    #     logger.debug('objects_2.shape: %(1)s', {'1': objects_2.shape})
    #     logger.debug('objects_1: %(1)s', {'1': objects_1})
    #     logger.debug('objects_2: %(1)s', {'1': objects_2})
    #     logger.debug('ul_1: %(1)s', {'1': ul_1})
    #     logger.debug('ul_2: %(1)s', {'1': ul_2})
    #     logger.debug('size_1: %(1)s', {'1': size_1})
    #     logger.debug('size_2: %(1)s', {'1': size_2})

    br_1 = ul_1 + size_1 - 1  # n1 x 2
    br_1_rep = np.tile(np.reshape(br_1, (n1, 1, 2)), (1, n2, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)
    br_2 = ul_2 + size_2 - 1  # n2 x 2
    br_2_rep = np.tile(np.reshape(br_2, (1, n2, 2)), (n1, 1, 1))  # np(n1 x n2 x 2) -> std(n2 x 2 x n1)

    size_inter = np.minimum(br_1_rep, br_2_rep) - np.maximum(ul_1_rep, ul_2_rep) + 1  # n2 x 2 x n1
    size_inter[size_inter < 0] = 0
    # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area_1 = np.multiply(size_1[:, 0], size_1[:, 1]).reshape((n1, 1))  # n1 x 1
    area_1_rep = np.tile(area_1, (1, n2))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_2 = np.multiply(size_2[:, 0], size_2[:, 1]).reshape((n2, 1))  # n2 x 1
    area_2_rep = np.tile(area_2.transpose(), (n1, 1))  # np(n1 x n2 x 1) -> std(n2 x 1 x n1)
    area_union = area_1_rep + area_2_rep - area_inter  # n2 x 1 x n1

    if iou is not None:
        # write('iou.shape: {}\n'.format(iou.shape))
        # write('area_inter.shape: {}\n'.format(area_inter.shape))
        # write('area_union.shape: {}\n'.format(area_union.shape))
        iou[:] = np.divide(area_inter, area_union)  # n1 x n2
    if ioa_1 is not None:
        ioa_1[:] = np.divide(area_inter, area_1_rep)  # n1 x n2
    if ioa_2 is not None:
        ioa_2[:] = np.divide(area_inter, area_2_rep)  # n1 x n2


def compute_iou_single(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2

    # bb1 = [bb1[0], bb1[1], bb1[0] + bb1[2], bb1[1] + bb1[3]]
    # bb2 = [bb2[0], bb2[1], bb2[0] + bb2[2], bb2[1] + bb2[3]]

    bb_intersect = [max(x1, x2),
                    max(y1, y2),
                    min(x1 + w1, x2 + w2),
                    min(y1 + h1, y2 + h2)]

    iw = bb_intersect[2] - bb_intersect[0] + 1
    ih = bb_intersect[3] - bb_intersect[1] + 1

    if iw <= 0 or ih <= 0:
        return 0
    area_intersect = (iw * ih)
    area_union = (w1 + 1) * (h1 + 1) + (w2 + 1) * (h2 + 1) - area_intersect
    iou = area_intersect / area_union
    return iou


def compute_overlap(iou, ioa_1, ioa_2, object_1, objects_2, logger=None, debug=False):
    """

    compute overlap of a single object with one or more objects
    specialized version for greater speed

    :type iou: np.ndarray | None
    :type ioa_1: np.ndarray | None
    :type ioa_2: np.ndarray | None
    :type object_1: np.ndarray
    :type objects_2: np.ndarray
    :type logger: logging.RootLogger | None
    :rtype: None
    """

    n1 = object_1.shape[0]

    assert n1 == 1, "object_1 should be a single object"

    n = objects_2.shape[0]

    ul_coord_1 = object_1[0, :2].reshape((1, 2))
    ul_coords_2 = objects_2[:, :2]  # n x 2
    ul_coords_inter = np.maximum(ul_coord_1, ul_coords_2)  # n x 2

    size_1 = object_1[0, 2:].reshape((1, 2))
    sizes_2 = objects_2[:, 2:]  # n x 2

    br_coord_1 = ul_coord_1 + size_1 - 1
    br_coords_2 = ul_coords_2 + sizes_2 - 1  # n x 2
    br_coords_inter = np.minimum(br_coord_1, br_coords_2)  # n x 2

    sizes_inter = br_coords_inter - ul_coords_inter + 1
    sizes_inter[sizes_inter < 0] = 0

    # valid_idx = np.flatnonzero((sizes_inter >= 0).all(axis=1))
    # valid_count = valid_idx.size
    # if valid_count == 0:
    #     if iou is not None:
    #         iou.fill(0)
    #     if ioa_1 is not None:
    #         ioa_1.fill(0)
    #     if ioa_2 is not None:
    #         ioa_2.fill(0)
    #     return

    areas_inter = np.multiply(sizes_inter[:, 0], sizes_inter[:, 1]).reshape((n, 1))  # n x 1

    # if logger is not None:
    #     logger.debug('object_1.shape: %(1)s', {'1': object_1.shape})
    #     logger.debug('objects_2.shape: %(1)s', {'1': objects_2.shape})
    #     logger.debug('object_1: %(1)s', {'1': object_1})
    #     logger.debug('objects_2: %(1)s', {'1': objects_2})
    #     logger.debug('ul_coord_1: %(1)s', {'1': ul_coord_1})
    #     logger.debug('ul_coords_2: %(1)s', {'1': ul_coords_2})
    #     logger.debug('size_1: %(1)s', {'1': size_1})
    #     logger.debug('sizes_2: %(1)s', {'1': sizes_2})
    #     logger.debug('areas_inter: %(1)s', {'1': areas_inter})
    #     logger.debug('sizes_inter: %(1)s', {'1': sizes_inter})

    areas_2 = None
    if iou is not None:
        # iou.fill(0)
        areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1]).reshape((n, 1))  # n x 1
        area_union = size_1[0, 0] * size_1[0, 1] + areas_2 - areas_inter
        # if logger is not None:
        #     logger.debug('iou.shape: %(1)s', {'1': iou.shape})
        #     logger.debug('area_union.shape: %(1)s', {'1': area_union.shape})
        #     logger.debug('area_union: %(1)s', {'1': area_union})
        iou[:] = np.divide(areas_inter, area_union)
    if ioa_1 is not None:
        # ioa_1.fill(0)
        ioa_1[:] = np.divide(areas_inter, size_1[0, 0] * size_1[0, 1])
    if ioa_2 is not None:
        # ioa_2.fill(0)
        if areas_2 is None:
            areas_2 = np.multiply(sizes_2[:, 0], sizes_2[:, 1])
        ioa_2[:] = np.divide(areas_inter, areas_2)
    if debug:
        logger.debug('paused')


# faster version for single frame operations
def compute_self_overlaps(iou, ioa, boxes):
    """
    :type iou: np.ndarray | None
    :type ioa: np.ndarray | None
    :type boxes: np.ndarray
    :rtype: None
    """
    n = boxes.shape[0]

    ul = boxes[:, :2].reshape((n, 2))  # n x 2
    ul_rep = np.tile(np.reshape(ul, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
    ul_2_rep = np.tile(np.reshape(ul, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
    ul_inter = np.maximum(ul_rep, ul_2_rep)  # n x 2 x n

    sizes = boxes[:, 2:].reshape((n, 2))  # n1 x 2
    br = ul + sizes - 1  # n1 x 2
    # size_ = boxes[:, 2:]  # n x 2
    # br = ul + size_ - 1  # n x 2
    br_rep = np.tile(np.reshape(br, (n, 1, 2)), (1, n, 1))  # np(n x n x 2) -> std(n x 2 x n)
    br_2_rep = np.tile(np.reshape(br, (1, n, 2)), (n, 1, 1))  # np(n x n x 2) -> std(n x 2 x n)
    br_inter = np.minimum(br_rep, br_2_rep)  # n x 2 x n

    size_inter = br_inter - ul_inter + 1  # np(n x n x 2) -> std(n x 2 x n)
    size_inter[size_inter < 0] = 0
    # np(n x n x 1) -> std(n x 1 x n)
    area_inter = np.multiply(size_inter[:, :, 0], size_inter[:, :, 1])

    area = np.multiply(sizes[:, 0], sizes[:, 1]).reshape((n, 1))  # n x 1
    area_rep = np.tile(area, (1, n))  # np(n x n x 1) -> std(n x 1 x n)
    area_2_rep = np.tile(area.transpose(), (n, 1))  # np(n x n x 1) -> std(n x 1 x n)
    area_union = area_rep + area_2_rep - area_inter  # n x 1 x n

    if iou is not None:
        iou[:] = np.divide(area_inter, area_union)  # n x n
        idx = np.arange(n)
        iou[idx, idx] = 0
    if ioa is not None:
        ioa[:] = np.divide(area_inter, area)  # n x n
        idx = np.arange(n)
        ioa[idx, idx] = 0


def get_max_overlap_obj(objects, location, _logger):
    """

    :param objects:
    :param location:
    :param _logger:
    :return:
    """
    if objects.shape[0] == 0:
        """no objects"""
        max_iou = 0
        max_iou_idx = None
    elif objects.shape[0] == 1:
        """single object"""
        iou = np.empty((1, 1))
        compute_overlap(iou, None, None, objects[0, 2:6].reshape((1, 4)),
                        location, _logger)
        max_iou_idx = 0
        max_iou = iou
    else:
        """get object with maximum overlap with the location"""
        iou = np.empty((objects.shape[0], 1))
        compute_overlaps_multi(iou, None, None, objects[:, 2:6],
                               location, _logger)
        max_iou_idx = np.argmax(iou, axis=0).item()
        max_iou = iou[max_iou_idx, 0]

    return max_iou, max_iou_idx


def resize_ar(src_img, width=0, height=0, return_factors=False,
              placement_type=0, only_border=0, only_shrink=0):
    src_height, src_width = src_img.shape[:2]
    src_aspect_ratio = float(src_width) / float(src_height)

    if len(src_img.shape) == 3:
        n_channels = src_img.shape[2]
    else:
        n_channels = 1

    if width <= 0 and height <= 0:
        raise AssertionError('Both width and height cannot be zero')
    elif height <= 0:
        if only_shrink and width > src_width:
            width = src_width
        if only_border:
            height = src_height
        else:
            height = int(width / src_aspect_ratio)
    elif width <= 0:
        if only_shrink and height > src_height:
            height = src_height
        if only_border:
            width = src_width
        else:
            width = int(height * src_aspect_ratio)

    aspect_ratio = float(width) / float(height)

    if only_border:
        dst_width = width
        dst_height = height
        if placement_type == 0:
            start_row = start_col = 0
        elif placement_type == 1:
            start_row = int((dst_height - src_height) / 2.0)
            start_col = int((dst_width - src_width) / 2.0)
        elif placement_type == 2:
            start_row = int(dst_height - src_height)
            start_col = int(dst_width - src_width)
        else:
            raise AssertionError('Invalid placement_type: {}'.format(placement_type))
    else:

        if src_aspect_ratio == aspect_ratio:
            dst_width = src_width
            dst_height = src_height
            start_row = start_col = 0
        elif src_aspect_ratio > aspect_ratio:
            dst_width = src_width
            dst_height = int(src_width / aspect_ratio)
            start_row = int((dst_height - src_height) / 2.0)
            if placement_type == 0:
                start_row = 0
            elif placement_type == 1:
                start_row = int((dst_height - src_height) / 2.0)
            elif placement_type == 2:
                start_row = int(dst_height - src_height)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_col = 0
        else:
            dst_height = src_height
            dst_width = int(src_height * aspect_ratio)
            start_col = int((dst_width - src_width) / 2.0)
            if placement_type == 0:
                start_col = 0
            elif placement_type == 1:
                start_col = int((dst_width - src_width) / 2.0)
            elif placement_type == 2:
                start_col = int(dst_width - src_width)
            else:
                raise AssertionError('Invalid placement_type: {}'.format(placement_type))
            start_row = 0

    dst_img = np.zeros((dst_height, dst_width, n_channels), dtype=src_img.dtype)
    dst_img = dst_img.squeeze()

    dst_img[start_row:start_row + src_height, start_col:start_col + src_width, ...] = src_img
    if not only_border:
        dst_img = cv2.resize(dst_img, (width, height))

    if return_factors:
        resize_factor = float(height) / float(dst_height)
        return dst_img, resize_factor, start_row, start_col
    else:
        return dst_img


def log_debug_multi(logger, vars, names):
    log_str = ''
    log_dict = {}
    for i in range(len(vars)):
        log_str += '{:s}: %%({:d})s'.format(names[i], i + 1)
        log_dict['{:d}'.format(i + 1)] = vars[i]
    logger.debug(log_str, log_dict)


def draw_pts(img, pts):
    for _pt in pts:
        _pt = (int(_pt[0]), int(_pt[1]))
        cv2.circle(img, _pt, 1, (0, 0, 0), 2)


def draw_region(img, corners, color, thickness=1):
    # draw the bounding box specified by the given corners
    for i in range(4):
        p1 = (int(corners[0, i]), int(corners[1, i]))
        p2 = (int(corners[0, (i + 1) % 4]), int(corners[1, (i + 1) % 4]))
        cv2.line(img, p1, p2, color, thickness)


def draw_traj2(frame, _obj_centers_rec, obj_boxes=None, color='blue', thickness=1):
    n_traj = len(_obj_centers_rec)
    for __i in range(1, n_traj):
        pt1 = _obj_centers_rec[__i - 1]
        _obj_cx_rec, _obj_cy_rec = pt1
        pt1 = (int(_obj_cx_rec), int(_obj_cy_rec))

        pt2 = _obj_centers_rec[__i]
        _obj_cx_rec, _obj_cy_rec = pt2
        pt2 = (int(_obj_cx_rec), int(_obj_cy_rec))

        cv2.line(frame, pt1, pt2, col_bgr[color], thickness=thickness)

        if obj_boxes is not None:
            draw_box(frame, obj_boxes[__i], color=color, thickness=thickness)


def draw_boxes(frame, boxes, _id=None, color='black', thickness=2,
               is_dotted=0, transparency=0.):
    boxes = np.asarray(boxes).reshape((-1, 4))

    # if isinstance(boxes, list):
    #
    # if len(boxes.shape) == 1:
    #     boxes = np.expand_dims(boxes, axis=0)

    for box in boxes:
        draw_box(frame, box, _id, color, thickness,
                 is_dotted, transparency)


def get_shifted_boxes(anchor_box, img, n_samples,
                      min_anchor_iou, max_anchor_iou,
                      min_shift_ratio, max_shift_ratio,
                      min_resize_ratio=None, max_resize_ratio=None,
                      min_size=20, max_size_ratio=0.8,
                      gt_boxes=None,
                      max_gt_iou=None,
                      sampled_boxes=None,
                      max_sampled_iou=0.5,
                      max_iters=100,
                      name='',
                      vis=0):
    """

    :param anchor_box:
    :param img:
    :param n_samples:
    :param min_anchor_iou:
    :param max_anchor_iou:
    :param min_shift_ratio:
    :param max_shift_ratio:
    :param min_resize_ratio:
    :param max_resize_ratio:
    :param min_size:
    :param max_size_ratio:
    :param gt_boxes:
    :param max_gt_iou:
    :param sampled_boxes:
    :param max_sampled_iou:
    :param max_iters:
    :param name:
    :param vis:
    :return:
    """

    # vis = 1

    if not name:
        name = 'get_shifted_boxes'

    if np.any(np.isnan(anchor_box)):
        print('invalid location provided: {}'.format(anchor_box))
        return

    assert 0 <= min_shift_ratio <= max_shift_ratio, "Invalid shift ratios provided"
    if max_gt_iou is None:
        max_gt_iou = max_anchor_iou

    if min_resize_ratio is None:
        min_resize_ratio = min_shift_ratio

    if max_resize_ratio is None:
        max_resize_ratio = max_shift_ratio

    img_h, img_w = img.shape[:2]
    x, y, w, h = anchor_box.squeeze()
    x2, y2 = x + w, y + h

    _x, _x2 = clamp((x, x2), 0, img_w)
    _y, _y2 = clamp((y, y2), 0, img_h)

    _w, _h = _x2 - _x, _y2 - _y

    if vis:
        disp_img = np.copy(img)
        if len(disp_img.shape) == 2:
            """grey scale to RGB"""
            disp_img = np.stack((disp_img,) * 3, axis=2)

    valid_boxes = []

    max_box_w, max_box_h = float(img_w) * max_size_ratio, float(img_h) * max_size_ratio

    if _w < min_size or _h < min_size or _w > img_w or _h > img_h or (_w > max_box_w and _h > max_box_h):
        if vis:
            draw_box(disp_img, [x, y, w, h], _id=None, color='blue', thickness=2,
                     is_dotted=0, transparency=0.)
            draw_box(disp_img, [_x, _y, _w, _h], _id=None, color='blue', thickness=2,
                     is_dotted=1, transparency=0.)

            annotate_and_show(name, disp_img, 'annoying crap')

        # print('\nignoring amazingly annoying vile filthy invalid box: {} clamped to: {}\n'.format(
        #     (x, y, w, h), (_x, _y, _w, _h)
        # ))
        return valid_boxes

    x, y, w, h = _x, _y, _w, _h

    # min_dx, min_dy = shift_coeff*x, shift_coeff*y
    # min_dw, min_dh= shift_coeff*w, shift_coeff*h

    if sampled_boxes is None:
        _sampled_boxes = []
    else:
        _sampled_boxes = sampled_boxes.copy()

    if vis:
        if gt_boxes is not None:
            draw_boxes(disp_img, gt_boxes, _id=None, color='green', thickness=2,
                       is_dotted=0, transparency=0.)

        if _sampled_boxes is not None:
            draw_boxes(disp_img, _sampled_boxes, _id=None, color='magenta', thickness=2,
                       is_dotted=1, transparency=0.)

        draw_box(disp_img, anchor_box, _id=None, color='blue', thickness=2,
                 is_dotted=0, transparency=0.)

    n_valid_boxes = 0
    txt = ''

    for iter_id in range(max_iters):

        txt = ''

        shift_x = random.uniform(min_shift_ratio, max_shift_ratio) * random.choice([1, 0, -1])
        shift_y = random.uniform(min_shift_ratio, max_shift_ratio) * random.choice([1, 0, -1])
        shift_w = random.uniform(min_resize_ratio, max_resize_ratio) * random.choice([1, 0, -1])
        shift_h = random.uniform(min_resize_ratio, max_resize_ratio) * random.choice([1, 0, -1])

        x2 = x + w * shift_x
        y2 = y + h * shift_y
        w2 = w * (1 + shift_w)
        h2 = h * (1 + shift_h)

        if w2 <= min_size:
            continue

        if h2 <= min_size:
            continue

        if x2 <= 0 or x2 + w2 >= img_w:
            continue

        if y2 <= 0 or y2 + h2 >= img_h:
            continue

        if w2 > max_box_w and h2 > max_box_h:
            continue

        shifted_box = np.asarray([x2, y2, w2, h2])

        # iou = np.empty((1, 1))
        iou = compute_iou_single(shifted_box, anchor_box)

        if iou < min_anchor_iou or iou > max_anchor_iou:
            continue

        txt = 'shift: [{:.2f},{:.2f},{:.2f},{:.2f}] iou: {:.2f}'.format(shift_x, shift_y, shift_w, shift_h, iou)

        if _sampled_boxes:
            sampled_iou = np.empty((len(_sampled_boxes), 1))
            compute_overlap(sampled_iou, None, None, shifted_box.reshape((1, 4)),
                            np.asarray(_sampled_boxes).reshape((-1, 4)))

            _max_sampled_iou = np.amax(sampled_iou)
            if _max_sampled_iou > max_sampled_iou:
                continue

            txt += ' sampled_iou: {:.2f}'.format(_max_sampled_iou)

        if gt_boxes is not None:
            gt_iou = np.empty((gt_boxes.shape[0], 1))
            compute_overlap(gt_iou, None, None, shifted_box.reshape((1, 4)), gt_boxes)

            _max_gt_iou = np.amax(gt_iou)
            if _max_gt_iou > max_gt_iou:
                continue

            txt += ' gt_iou: {:.2f}'.format(_max_gt_iou)

        # iou = iou.item()

        valid_boxes.append(shifted_box)
        _sampled_boxes.append(shifted_box)

        n_valid_boxes += 1

        if vis:
            txt += ' iters: {:d}'.format(iter_id)

            draw_box(disp_img, shifted_box, _id=None, color='red', thickness=2,
                     is_dotted=0, transparency=0.)

            annotate_and_show(name, disp_img, txt)

        if n_valid_boxes >= n_samples:
            break
    else:
        if vis:
            txt += ' iters: {:d}'.format(iter_id)

            draw_box(disp_img, shifted_box, _id=None, color='red', thickness=2,
                     is_dotted=0, transparency=0.)

            annotate_and_show(name, disp_img, txt)

        # print('max iters {} reached with only {} / {} valid sampled boxes found'.format(
        #     max_iters, n_valid_boxes, n_samples))

    # valid_boxes = np.asarray(valid_boxes).reshape((n_samples, 4))
    return valid_boxes


def draw_box(frame, box, _id=None, color='black', thickness=2,
             is_dotted=0, transparency=0., text_col=None):
    """
    :type frame: np.ndarray
    :type box: np.ndarray
    :type _id: int | str | None
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :type transparency: float
    :rtype: None
    """
    box = np.asarray(box)

    if np.any(np.isnan(box)):
        print('invalid location provided: {}'.format(box))
        return

    box = box.squeeze()
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[0] + box[2]),
           int(box[1] + box[3]))

    if transparency > 0:
        _frame = np.copy(frame)
    else:
        _frame = frame

    if is_dotted:
        draw_dotted_rect(_frame, pt1, pt2, col_bgr[color], thickness=thickness)
    else:
        cv2.rectangle(_frame, pt1, pt2, col_bgr[color], thickness=thickness)

    if transparency > 0:
        frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...] = (
                frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * (1 - transparency) +
                _frame[pt1[1]:pt2[1], pt1[0]:pt2[0], ...].astype(np.float32) * transparency
        ).astype(frame.dtype)

    if _id is not None:
        if text_col is None:
            text_col = color
        text_col = col_bgr[text_col]
        if cv2.__version__.startswith('2'):
            font_line_type = cv2.CV_AA
        else:
            font_line_type = cv2.LINE_AA

        cv2.putText(frame, str(_id), (int(box[0] - 1), int(box[1] - 10)),
                    # cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.50, text_col, 1, font_line_type)


def draw_trajectory(frame, trajectory, color='black', thickness=2, is_dotted=0):
    """
    :type frame: np.ndarraycol_bgr
    :type trajectory: list[np.ndarray]
    :param color: indexes into col_rgb
    :type color: str
    :type thickness: int
    :type is_dotted: int
    :rtype: None
    """

    n_traj = len(trajectory)
    for i in range(1, n_traj):
        pt1 = tuple(trajectory[i - 1].astype(np.int64))
        pt2 = tuple(trajectory[i].astype(np.int64))

        if is_dotted:
            draw_dotted_line(frame, pt1, pt2, col_bgr[color], thickness)
        else:
            try:
                cv2.line(frame, pt1, pt2, col_bgr[color], thickness=thickness)

            except TypeError:
                print('frame.dtype', frame.dtype)
                print('pt1', pt1)
                print('pt2', pt2)


def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=7):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    # if style == 'dotted':
    for p in pts:
        cv2.circle(img, p, thickness, color, -1)
    # else:
    #     s = pts[0]
    #     e = pts[0]
    #     i = 0
    #     for p in pts:
    #         s = e
    #         e = p
    #         if i % 2 == 1:
    #             cv2.line(img, s, e, color, thickness)
    #         i += 1


def draw_dotted_poly(img, pts, color, thickness=1):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        draw_dotted_line(img, s, e, color, thickness)


def draw_dotted_rect(img, pt1, pt2, color, thickness=1):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    draw_dotted_poly(img, pts, color, thickness)


def write_to_files(root_dir, write_to_bin, entries):
    os.makedirs(root_dir, exist_ok=True)

    if write_to_bin:
        file_ext = 'bin'
    else:
        file_ext = 'txt'
    for entry in entries:
        array = entry[0]
        fname = '{:s}/{:s}.{:s}'.format(root_dir, entry[1], file_ext)
        if write_to_bin:
            dtype = entry[2]
            array.astype(dtype).tofile(open(fname, 'wb'))
        else:
            fmt = entry[3]
            np.savetxt(fname, array, delimiter='\t', fmt=fmt)


def prob_to_rgb(value, minimum=0., maximum=1.):
    ratio = 2 * (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def prob_to_rgb2(value, minimum=0., maximum=1.):
    ratio = (value - minimum) / (maximum - minimum)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(255 - b)
    g = 0
    return r, g, b


def compare_files(read_from_bin, files, dirs=None, sync_id=-1, msg=''):
    """
    :type target: target
    :type read_from_bin: bool | int
    :type files: list[(str, Type(np.dtype), tuple)]
    :type dirs: (str, ...) | None
    :type sync_id: int
    :type msg: str
    :rtype: bool
    """

    if dirs is None:
        params = DebugParams()
        dirs = params.cmp_root_dirs

    if read_from_bin:
        file_ext = 'bin'
    else:
        file_ext = 'txt'

    self_dir = os.path.abspath(dirs[1])
    # print('self_dir: {}'.format(self_dir))

    if not dirs[0]:
        if sync_id >= 0:
            sync_fname = '{:s}/write_{:d}.sync'.format(self_dir, sync_id)
            open(sync_fname, 'w').close()

            # sys.stdout.write('{:s} Wrote {:s}...\n'.format(msg, sync_fname))
            # sys.stdout.flush()

            sync_fname = '{:s}/read_{:d}.sync'.format(self_dir, sync_id)
            sys.stdout.write('\n{:s} Waiting for {:s}...'.format(msg, sync_fname))
            sys.stdout.flush()
            iter_id = 0
            while not os.path.isfile(sync_fname):
                time.sleep(0.5)

            sys.stdout.write('\n')
            sys.stdout.flush()

            while True:
                try:
                    os.remove(sync_fname)
                except PermissionError:
                    time.sleep(0.5)
                else:
                    break
        return

    other_dir = os.path.abspath(dirs[0])
    # print('other_dir: {}'.format(other_dir))

    if sync_id >= 0:
        sync_fname = '{:s}/write_{:d}.sync'.format(other_dir, sync_id)
        sys.stdout.write('\n{:s} Waiting for {:s}...'.format(msg, sync_fname))
        sys.stdout.flush()
        iter_id = 0
        while not os.path.isfile(sync_fname):
            time.sleep(0.5)
            # iter_id += 1
            # if iter_id==10:
            #     return False
            # sys.stdout.write('.')
            # sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        while True:
            try:
                os.remove(sync_fname)
            except PermissionError:
                time.sleep(0.5)
            else:
                break

    files_are_same = True
    array_1 = {}
    array_2 = {}
    diff_array = {}
    equality_array = {}
    for fname, ftype, fshape in files:
        path_1 = '{:s}/{:s}.{:s}'.format(other_dir, fname, file_ext)
        path_2 = '{:s}/{:s}.{:s}'.format(self_dir, fname, file_ext)

        if not os.path.isfile(path_1):
            print('{:s} does not exist'.format(path_1))
            continue
        if not os.path.isfile(path_2):
            print('{:s} does not exist'.format(path_2))
            continue
        if not read_from_bin:
            subprocess.call('dos2unix -q {:s}'.format(path_1), shell=True)
            subprocess.call('dos2unix -q {:s}'.format(path_2), shell=True)
            subprocess.call('sed -i -e \'s/NaN/nan/g\' {:s}'.format(path_1), shell=True)
        if not filecmp.cmp(path_1, path_2):
            print('Files {:s} and {:s} are different'.format(path_1, path_2))
            files_are_same = False
            if read_from_bin:
                array_1[fname] = np.fromfile(path_1, dtype=ftype).reshape(fshape)
                array_2[fname] = np.fromfile(path_2, dtype=ftype).reshape(fshape)
                diff_array[fname] = np.abs(array_1[fname] - array_2[fname])
                equality_array[fname] = array_1[fname] == array_2[fname]
            else:
                subprocess.call('diff {:s} {:s} > {:s}/{:s}.diff'.format(
                    path_1, path_2, other_dir, fname), shell=True)
    if not files_are_same:
        print('paused')

    if sync_id >= 0:
        sync_fname = '{:s}/read_{:d}.sync'.format(other_dir, sync_id)
        open(sync_fname, 'w').close()

        # sys.stdout.write('{:s} Wrote {:s}...\n'.format(msg, sync_fname))
        # sys.stdout.flush()

    return files_are_same


class SIIF:
    @staticmethod
    def setup():
        # os.environ["SIIF_DUMP_IMAGES"] = "0"
        procs = []
        for proc in psutil.process_iter():
            try:
                # Get process name & pid from process object.
                process_name = proc.name()
                process_id = proc.pid
                cmdline = proc.cmdline()
                cmdline_txt = ' '.join(cmdline)
                # for _cmd in cmdline:
                #     cmdline_txt += ' ' + _cmd
                procs.append((process_name, cmdline_txt, process_id))
                # print(process_name, ' ::: ', process_id)

                siif_path = [k for k in cmdline if 'show_images_in_folder.py' in k]

                if process_name.startswith('python3') and siif_path:
                    siif_path = os.path.abspath(siif_path[0])
                    siif_dir = os.path.dirname(siif_path)
                    siif_log_path = os.path.join(siif_dir, 'siif_log.txt')
                    if not os.path.isfile(siif_log_path):
                        raise IOError(f'siif_log_path does not exist: {siif_log_path}')
                    with open(siif_log_path, 'r') as fid:
                        siif_src_path = fid.readline()

                    print("SIIF is active at {}".format(siif_src_path))
                    # os.environ["SIIF_DUMP_IMAGES"] = "1"
                    os.environ["SIIF_PATH"] = siif_src_path

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        # procs.sort(key=lambda x: x[0])
        # print(pformat(procs))
        # print(pformat(os.environ))
        # exit()

    @staticmethod
    def imshow(title, img):
        try:
            siif_path = os.environ["SIIF_PATH"]
        except KeyError:
            cv2.imshow(title, img)
            return 0

        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
        out_path = os.path.join(siif_path, '{}___{}.bmp'.format(title, time_stamp))

        # cv2.imshow('siif', img)
        cv2.imwrite(out_path, img)

        while os.path.exists(out_path):
            continue

        return 1


class CVText:
    def __init__(self, color='white', bkg_color='black', location=0, font=5,
                 size=0.8, thickness=1, line_type=2, offset=(5, 25)):
        self.color = color
        self.bkg_color = bkg_color
        self.location = location
        self.font = font
        self.size = size
        self.thickness = thickness
        self.line_type = line_type
        self.offset = offset

        self.help = {
            'font': 'Available fonts: '
                    '0: cv2.FONT_HERSHEY_SIMPLEX, '
                    '1: cv2.FONT_HERSHEY_PLAIN, '
                    '2: cv2.FONT_HERSHEY_DUPLEX, '
                    '3: cv2.FONT_HERSHEY_COMPLEX, '
                    '4: cv2.FONT_HERSHEY_TRIPLEX, '
                    '5: cv2.FONT_HERSHEY_COMPLEX_SMALL, '
                    '6: cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'
                    '7: cv2.FONT_HERSHEY_SCRIPT_COMPLEX; ',
            'location': '0: top left, 1: top right, 2: bottom right, 3: bottom left; ',
            'bkg_color': 'should be empty for no background',
        }


def show(title, frame_disp, _pause=1):
    if SIIF.imshow(title, frame_disp):
        return _pause

    # cv2.imshow(title, frame_disp)

    if _pause > 1:
        wait = _pause
    else:
        wait = 1 - _pause

    k = cv2.waitKey(wait)

    if k == 27:
        sys.exit()
    if k == 32:
        _pause = 1 - _pause

    return _pause


import traceback


def modules_from_trace(call_stack, n_modules, start_module=1):
    """

    :param list[traceback.FrameSummary] call_stack:
    :param int n_modules:
    :param int start_module:
    :return:
    """
    call_stack = call_stack[::-1]

    modules = []

    for module_id in range(start_module, start_module + n_modules):
        module_fs = call_stack[module_id]
        file = os.path.splitext(os.path.basename(module_fs.filename))[0]
        line = module_fs.lineno
        func = module_fs.name

        modules.append('{}:{}:{}'.format(file, func, line))

    modules_str = '<'.join(modules)
    return modules_str


def annotate_and_show(title, img_list, text=None, pause=1,
                      fmt=CVText(), no_resize=1, grid_size=(-1, 1), n_modules=3,
                      use_plt=0, max_width=0, max_height=0, only_annotate=0,
                      key_processor=None, pause_time=100, img_labels=None):
    """

    :param str title:
    :param np.ndarray | list | tuple img_list:
    :param str | logging.RootLogger | CustomLogger text:
    :param int pause:
    :param CVText fmt:
    :param int no_resize:
    :param int n_modules:
    :param int use_plt:
    :param tuple(int) grid_size:
    :return:
    """

    # call_stack = traceback.format_stack()
    # print(pformat(call_stack))
    # for line in traceback.format_stack():
    #     print(line.strip())

    if isinstance(text, (logging.RootLogger, CustomLogger)):
        string_stream = [k.stream for k in text.handlers if isinstance(k.stream, StringIO)]
        assert string_stream, "No string streams in logger"
        _str = string_stream[0].getvalue()
        _str_list = _str.split('\n')[-2].split('  :::  ')

        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules - 1, start_module=2)
            _str_list[0] = '{} ({})'.format(_str_list[0], modules_str)
        text = '\n'.join(_str_list)
    else:
        if n_modules:
            modules_str = modules_from_trace(traceback.extract_stack(), n_modules, start_module=1)
            if text is None:
                text = modules_str
            else:
                text = '{}\n({})'.format(text, modules_str)
        else:
            if text is None:
                text = title

    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list, ]

    if img_labels is not None:
        assert len(img_labels) == len(img_list), "img_labels and img_list must have same length"

    size = fmt.size

    # print('self.size: {}'.format(self.size))

    color = col_bgr[fmt.color]
    font = CVConstants.fonts[fmt.font]
    line_type = CVConstants.line_types[fmt.line_type]

    location = list(fmt.offset)

    if '\n' in text:
        text_list = text.split('\n')
    else:
        text_list = [text, ]

    max_text_width = 0
    text_height = 0
    text_heights = []

    for _text in text_list:
        (_text_width, _text_height) = cv2.getTextSize(_text, font, fontScale=fmt.size, thickness=fmt.thickness)[0]
        if _text_width > max_text_width:
            max_text_width = _text_width
        text_height += _text_height + 5
        text_heights.append(_text_height)

    text_width = max_text_width + 10
    text_height += 30

    text_img = np.zeros((text_height, text_width), dtype=np.uint8)
    for _id, _text in enumerate(text_list):
        cv2.putText(text_img, _text, tuple(location), font, size, color, fmt.thickness, line_type)
        location[1] += text_heights[_id] + 5

    text_img = text_img.astype(np.float32) / 255.0

    text_img = np.stack([text_img, ] * 3, axis=2)

    for _id, _img in enumerate(img_list):
        if len(_img.shape) == 2:
            _img = np.stack([_img, ] * 3, axis=2)

        if _img.dtype == np.uint8:
            _img = _img.astype(np.float32) / 255.0

        if img_labels is not None:
            img_label = img_labels[_id]
            label_img = np.zeros((text_height, text_width), dtype=np.uint8)
            cv2.putText(label_img, img_label, tuple(fmt.offset), font, size, color, fmt.thickness, line_type)
            label_img = label_img.astype(np.float32) / 255.0

            if len(_img.shape) == 3:
                label_img = np.stack([label_img, ] * 3, axis=2)

            img_list_label = [label_img, _img]

            _img = stack_images_with_resize(img_list_label, grid_size=(2, 1), preserve_order=1,
                                            only_border=no_resize)

        img_list[_id] = _img

    img_stacked = stack_images_with_resize(img_list, grid_size=grid_size, preserve_order=1,
                                           only_border=no_resize)
    img_list_txt = [text_img, img_stacked]

    img_stacked_txt = stack_images_with_resize(img_list_txt, grid_size=(2, 1), preserve_order=1,
                                               only_border=no_resize)
    # img_stacked_txt_res = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
    # img_stacked_txt_res_gs = cv2.cvtColor(img_stacked_txt_res, cv2.COLOR_BGR2GRAY)

    img_stacked_txt = (img_stacked_txt * 255).astype(np.uint8)

    if img_stacked_txt.shape[0] > max_height > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, height=max_height)

    if img_stacked_txt.shape[1] > max_width > 0:
        img_stacked_txt = resize_ar(img_stacked_txt, width=max_width)

    if only_annotate:
        return img_stacked_txt

    k = 0
    if use_plt:
        img_stacked_txt = cv2.resize(img_stacked_txt, (300, 300), fx=0, fy=0)
        plt.imshow(img_stacked_txt)
        plt.pause(0.0001)
    else:
        _siif = SIIF.imshow(title, img_stacked_txt)
        if _siif:
            return pause

        _pause_time = int((1 - pause) * pause_time)

        k = cv2.waitKey(_pause_time)
        if k == 27:
            cv2.destroyWindow(title)
            exit()
        if k == 32:
            pause = 1 - pause

    return k


def stack_images(img_list, stack_order=0, grid_size=None):
    """

    :param img_list:
    :param int stack_order:
    :param list | None | tuple grid_size:
    :return:
    """
    if isinstance(img_list, (tuple, list)):
        n_images = len(img_list)
        img_shape = img_list[0].shape
        is_list = 1
    else:
        n_images = img_list.shape[0]
        img_shape = img_list.shape[1:]
        is_list = 0

    if grid_size is None:
        grid_size = [int(np.ceil(np.sqrt(n_images))), ] * 2
    else:
        if len(grid_size) == 1:
            grid_size = [grid_size[0], grid_size[0]]
        elif grid_size[0] == -1:
            grid_size = [int(math.ceil(n_images / grid_size[1])), grid_size[1]]
        elif grid_size[1] == -1:
            grid_size = [grid_size[0], int(math.ceil(n_images / grid_size[0]))]

    stacked_img = None
    list_ended = False
    inner_axis = 1 - stack_order
    for row_id in range(grid_size[0]):
        start_id = grid_size[1] * row_id
        curr_row = None
        for col_id in range(grid_size[1]):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_shape, dtype=np.uint8)
                list_ended = True
            else:
                if is_list:
                    curr_img = img_list[img_id]
                else:
                    curr_img = img_list[img_id, :, :].squeeze()
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)
        if stacked_img is None:
            stacked_img = curr_row
        else:
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)
        if list_ended:
            break
    return stacked_img


# import gmpy
def stack_images_with_resize(img_list, grid_size=None, stack_order=0, borderless=1,
                             preserve_order=0, return_idx=0,
                             # annotations=None,
                             # ann_fmt=(0, 5, 15, 1, 1, 255, 255, 255, 0, 0, 0),
                             only_height=0, only_border=1):
    n_images = len(img_list)
    # print('grid_size: {}'.format(grid_size))

    if grid_size is None or not grid_size:
        n_cols = n_rows = int(np.ceil(np.sqrt(n_images)))
    else:
        n_rows, n_cols = grid_size

        if n_rows < 0:
            n_rows = int(np.ceil(n_images / n_cols))
        elif n_cols < 0:
            n_cols = int(np.ceil(n_images / n_rows))

    target_ar = 1920.0 / 1080.0
    if n_cols <= n_rows:
        target_ar /= 2.0
    shape_img_id = 0
    min_ar_diff = np.inf
    img_heights = np.zeros((n_images,), dtype=np.int32)
    for _img_id in range(n_images):
        height, width = img_list[_img_id].shape[:2]
        img_heights[_img_id] = height
        img_ar = float(n_cols * width) / float(n_rows * height)
        ar_diff = abs(img_ar - target_ar)
        if ar_diff < min_ar_diff:
            min_ar_diff = ar_diff
            shape_img_id = _img_id

    img_heights_sort_idx = np.argsort(-img_heights)
    row_start_idx = img_heights_sort_idx[:n_rows]
    img_idx = img_heights_sort_idx[n_rows:]
    # print('img_heights: {}'.format(img_heights))
    # print('img_heights_sort_idx: {}'.format(img_heights_sort_idx))
    # print('img_idx: {}'.format(img_idx))

    # grid_size = [n_rows, n_cols]
    img_size = img_list[shape_img_id].shape
    height, width = img_size[:2]

    if only_height:
        width = 0
    # grid_size = [n_rows, n_cols]
    # print 'img_size: ', img_size
    # print 'n_images: ', n_images
    # print 'grid_size: ', grid_size

    # print()
    stacked_img = None
    list_ended = False
    img_idx_id = 0
    inner_axis = 1 - stack_order
    stack_idx = []
    stack_locations = []
    start_row = 0
    # curr_ann = ''
    for row_id in range(n_rows):
        start_id = n_cols * row_id
        curr_row = None
        start_col = 0
        for col_id in range(n_cols):
            img_id = start_id + col_id
            if img_id >= n_images:
                curr_img = np.zeros(img_size, dtype=np.uint8)
                list_ended = True
            else:
                if preserve_order:
                    _curr_img_id = img_id
                elif col_id == 0:
                    _curr_img_id = row_start_idx[row_id]
                else:
                    _curr_img_id = img_idx[img_idx_id]
                    img_idx_id += 1

                curr_img = img_list[_curr_img_id]
                # if annotations:
                #     curr_ann = annotations[_curr_img_id]
                stack_idx.append(_curr_img_id)
                # print(curr_img.shape[:2])

                # if curr_ann:
                #     putTextWithBackground(curr_img, curr_ann, fmt=ann_fmt)

                if not borderless:
                    curr_img = resize_ar(curr_img, width, height, only_border=only_border)
                if img_id == n_images - 1:
                    list_ended = True
            if curr_row is None:
                curr_row = curr_img
            else:
                if borderless:
                    if curr_row.shape[0] < curr_img.shape[0]:
                        curr_row = resize_ar(curr_row, 0, curr_img.shape[0], only_border=only_border)
                    elif curr_img.shape[0] < curr_row.shape[0]:
                        curr_img = resize_ar(curr_img, 0, curr_row.shape[0], only_border=only_border)
                # print('curr_row.shape: ', curr_row.shape)
                # print('curr_img.shape: ', curr_img.shape)
                curr_row = np.concatenate((curr_row, curr_img), axis=inner_axis)

            curr_h, curr_w = curr_img.shape[:2]
            stack_locations.append((start_row, start_col, start_row + curr_h, start_col + curr_w))
            start_col += curr_w

        if stacked_img is None:
            stacked_img = curr_row
        else:
            if borderless:
                resize_factor = float(curr_row.shape[1]) / float(stacked_img.shape[1])
                if curr_row.shape[1] < stacked_img.shape[1]:
                    curr_row = resize_ar(curr_row, stacked_img.shape[1], 0, only_border=only_border)
                elif curr_row.shape[1] > stacked_img.shape[1]:
                    stacked_img = resize_ar(stacked_img, curr_row.shape[1], 0, only_border=only_border)

                new_start_col = 0
                for _i in range(n_cols):
                    _start_row, _start_col, _end_row, _end_col = stack_locations[_i - n_cols]
                    _w, _h = _end_col - _start_col, _end_row - _start_row
                    w_resized, h_resized = _w / resize_factor, _h / resize_factor
                    stack_locations[_i - n_cols] = (
                        _start_row, new_start_col, _start_row + h_resized, new_start_col + w_resized)
                    new_start_col += w_resized
            # print('curr_row.shape: ', curr_row.shape)
            # print('stacked_img.shape: ', stacked_img.shape)
            stacked_img = np.concatenate((stacked_img, curr_row), axis=stack_order)

        curr_h, curr_w = curr_row.shape[:2]
        start_row += curr_h

        if list_ended:
            break
    if return_idx:
        return stacked_img, stack_idx, stack_locations
    else:
        return stacked_img


def stack_images_1D(img_list, stack_order=0):
    # stack into a single row or column
    stacked_img = None
    inner_axis = 1 - stack_order
    for img in img_list:
        if stacked_img is None:
            stacked_img = img
        else:
            stacked_img = np.concatenate((stacked_img, img), axis=inner_axis)
    return stacked_img


def remove_sub_folders(dir_name, sub_dir_prefix):
    folders = [linux_path(dir_name, name) for name in os.listdir(dir_name) if
               name.startswith(sub_dir_prefix) and
               os.path.isdir(linux_path(dir_name, name))]
    for folder in folders:
        shutil.rmtree(folder)


def write(str):
    sys.stdout.write(str)
    sys.stdout.flush()


def get_date_time():
    return time.strftime("%y%m%d_%H%M", time.localtime())


def parse_seq_IDs(ids, sample=""):
    """

    :param ids:
    :param str sample:
    :return:
    """
    out_ids = []
    if isinstance(ids, int):
        out_ids.append(ids)
    else:
        for _id in ids:
            if isinstance(_id, list):
                if len(_id) == 1:
                    out_ids.extend(range(_id[0]))
                if len(_id) == 2:
                    out_ids.extend(range(_id[0], _id[1]))
                elif len(_id) == 3:
                    out_ids.extend(range(_id[0], _id[1], _id[2]))
            else:
                out_ids.append(_id)

    if sample:
        if sample == "even":
            out_ids = [k for k in out_ids if k % 2 == 0]
        elif sample == "odd":
            out_ids = [k for k in out_ids if k % 2 == 1]
        else:
            raise AssertionError('invalid sample type: {}'.format(sample))

    return tuple(out_ids)


def help_from_docs(obj, member):
    _help = ''
    doc = inspect.getdoc(obj)
    if doc is None:
        return _help

    doc_lines = doc.splitlines()
    if not doc_lines:
        return _help

    templ = ':param {} {}: '.format(type(getattr(obj, member)).__name__, member)
    relevant_line = [k for k in doc_lines if k.startswith(templ)]

    if relevant_line:
        _help = relevant_line[0].replace(templ, '')

    return _help


def str_to_tuple(val):
    if val.startswith('range('):
        val_list = val[6:].replace(')', '').split(',')
        val_list = [int(x) for x in val_list]
        val_list = tuple(range(*val_list))
        return val_list
    elif ',' not in val:
        val = '{},'.format(val)
    return literal_eval(val)


def add_params_to_parser(parser, obj, root_name='', obj_name=''):
    members = tuple([attr for attr in dir(obj) if not callable(getattr(obj, attr))
                     and not attr.startswith("__")])
    if obj_name:
        if root_name:
            root_name = '{:s}.{:s}'.format(root_name, obj_name)
        else:
            root_name = '{:s}'.format(obj_name)
    for member in members:
        if member == 'help':
            continue
        default_val = getattr(obj, member)
        if isinstance(default_val, (int, bool, float, str, tuple, dict)):
            if root_name:
                member_param_name = '{:s}.{:s}'.format(root_name, member)
            else:
                member_param_name = '{:s}'.format(member)
            if member in obj.help:
                _help = obj.help[member]
            else:
                _help = help_from_docs(obj, member)

            if isinstance(default_val, tuple):
                parser.add_argument('--{:s}'.format(member_param_name), type=str_to_tuple,
                                    default=default_val, help=_help, metavar='')
            elif isinstance(default_val, dict):
                parser.add_argument('--{:s}'.format(member_param_name), type=json.loads, default=default_val,
                                    help=_help, metavar='')
            else:
                parser.add_argument('--{:s}'.format(member_param_name), type=type(default_val), default=default_val,
                                    help=_help, metavar='')
        else:
            # parameter is itself an instance of some other parameter class so its members must
            # be processed recursively
            add_params_to_parser(parser, getattr(obj, member), root_name, member)


def assign_arg(obj, arg, id, val):
    if id >= len(arg):
        print('Invalid arg: ', arg)
        return
    _arg = arg[id]
    obj_attr = getattr(obj, _arg)
    if isinstance(obj_attr, (int, bool, float, str, list, tuple, dict)):
        if val == '#' or val == '__n__':
            if isinstance(obj_attr, str):
                # empty string
                val = ''
            elif isinstance(obj_attr, tuple):
                # empty tuple
                val = ()
            elif isinstance(obj_attr, list):
                # empty list
                val = []
            elif isinstance(obj_attr, dict):
                # empty dict
                val = {}
        setattr(obj, _arg, val)
    else:
        # parameter is itself an instance of some other parameter class so its members must
        # be processed recursively
        assign_arg(obj_attr, arg, id + 1, val)


def process_args_from_parser(obj, args):
    # arg_prefix = ''
    # if hasattr(obj, 'arg_prefix'):
    #     arg_prefix = obj.arg_prefix
    members = vars(args)
    for key in members.keys():
        val = members[key]
        key_parts = key.split('.')
        assign_arg(obj, key_parts, 0, val)


def get_intersection_area(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the dimensions of intersection rectangle
    height = (yB - yA + 1)
    width = (xB - xA + 1)

    if height > 0 and width > 0:
        return height * width
    return 0


def processArguments(args, params):
    # arguments specified as 'arg_name=argv_val'
    no_of_args = len(args)
    for arg_id in range(no_of_args):
        arg_str = args[arg_id]
        if arg_str.startswith('--'):
            arg_str = arg_str[2:]
        arg = arg_str.split('=')
        if len(arg) != 2 or not arg[0] in params.keys():
            raise IOError('Invalid argument provided: {:s}'.format(args[arg_id]))

        if not arg[1] or not arg[0] or arg[1] == '#':
            continue

        if isinstance(params[arg[0]], (list, tuple)):

            if ':' in arg[1]:
                inclusive_start = inclusive_end = 1
                if arg[1].endswith(')'):
                    arg[1] = arg[1][:-1]
                    inclusive_end = 0
                if arg[1].startswith(')'):
                    arg[1] = arg[1][1:]
                    inclusive_start = 0

                _temp = [float(k) for k in arg[1].split(':')]
                if len(_temp) == 3:
                    _step = _temp[2]
                else:
                    _step = 1.0
                if inclusive_end:
                    _temp[1] += _step
                if not inclusive_start:
                    _temp[0] += _step
                arg_vals_parsed = list(np.arange(*_temp))
            else:
                if arg[1] and ',' not in arg[1]:
                    arg[1] = '{},'.format(arg[1])

                arg_vals = [x for x in arg[1].split(',') if x]
                arg_vals_parsed = []
                for _val in arg_vals:
                    if _val == '__n__':
                        _val = ''
                    try:
                        _val_parsed = int(_val)
                    except ValueError:
                        try:
                            _val_parsed = float(_val)
                        except ValueError:
                            _val_parsed = _val
                    arg_vals_parsed.append(_val_parsed)

            params[arg[0]] = type(params[arg[0]])(arg_vals_parsed)
        else:
            params[arg[0]] = type(params[arg[0]])(arg[1])


def sort_key(fname):
    fname = os.path.splitext(os.path.basename(fname))[0]
    # print('fname: ', fname)
    # split_fname = fname.split('_')
    # print('split_fname: ', split_fname)

    # nums = [int(s) for s in fname.split('_') if s.isdigit()]
    # non_nums = [s for s in fname.split('_') if not s.isdigit()]

    split_list = fname.split('_')
    key = ''

    for s in split_list:
        if s.isdigit():
            if not key:
                key = '{:08d}'.format(int(s))
            else:
                key = '{}_{:08d}'.format(key, int(s))
        else:
            if not key:
                key = s
            else:
                key = '{}_{}'.format(key, s)

    # for non_num in non_nums:
    #     if not key:
    #         key = non_num
    #     else:
    #         key = '{}_{}'.format(key, non_num)
    # for num in nums:
    #     if not key:
    #         key = '{:08d}'.format(num)
    #     else:
    #         key = '{}_{:08d}'.format(key, num)

    # try:
    #     key = nums[-1]
    # except IndexError:
    #     return fname

    # print('fname: {}, key: {}'.format(fname, key))
    return key


def linux_path(*args, **kwargs):
    return os.path.join(*args, **kwargs).replace(os.sep, '/')


def nms(score_maxima_loc, dist_sqr_thresh):
    valid_score_maxima_x = []
    valid_score_maxima_y = []

    x, y = np.copy(score_maxima_loc[0]), np.copy(score_maxima_loc[1])

    while True:
        n_score_maxima_loc = len(x)
        if n_score_maxima_loc == 0:
            break

        curr_x, curr_y = x[0], y[0]
        valid_score_maxima_x.append(curr_x)
        valid_score_maxima_y.append(curr_y)

        # for j in range(1, n_score_maxima_loc):
        #     _x, _y = score_maxima_loc[0][j], score_maxima_loc[1][j]

        removed_idx = [0, ]

        removed_idx += [i for i in range(1, n_score_maxima_loc) if
                        (curr_x - x[i]) ** 2 + (curr_y - y[i]) ** 2 < dist_sqr_thresh]

        x, y = np.delete(x, removed_idx), np.delete(y, removed_idx)

    return [valid_score_maxima_x, valid_score_maxima_y]


#
# def df_test():
#     import pandas as pd
#     from policy import PolicyDecision
#
#     _stats_df = pd.DataFrame(
#         np.zeros((len(PolicyDecision.types), len(AnnotationStatus.types))),
#         columns=AnnotationStatus.types,
#         index=PolicyDecision.types,
#     )
#     _stats_df2 = pd.DataFrame(
#         np.zeros((len(PolicyDecision.types),)),
#         index=PolicyDecision.types,
#     )
#
#     _stats_df20 = _stats_df2[0]
#
#     _stats_df11 = _stats_df['fp_background']['unknown_neg']
#     _stats_df21 = _stats_df20['correct']
#
#     _stats_df20['correct'] = 67
#     _stats_df['fp_background']['unknown_neg'] = 92
#
#     _stats_df3 = pd.DataFrame(
#         np.full((len(PolicyDecision.types),), 3),
#         index=PolicyDecision.types,
#     )
#     _stats_df4 = pd.DataFrame(
#         np.full((len(PolicyDecision.types), len(AnnotationStatus.types)), 5),
#         columns=AnnotationStatus.types,
#         index=PolicyDecision.types,
#     )
#     _stats_df4['fp_background'] += _stats_df3[0]


def combined_motmetrics(acc_dict, logger):
    # logger.info(f'Computing overall MOT metrics over {len(acc_dict)} sequences...')
    # start_t = time.time()
    try:
        import evaluation.motmetrics as mm
    except ImportError as excp:
        logger.error('MOT evaluator is not available: {}'.format(excp))
        return False
    seq_names, accs = map(list, zip(*acc_dict.items()))

    # logger.info(f'Merging accumulators...')
    accs = mm.MOTAccumulator.merge_event_dataframes(accs)

    # logger.info(f'Computing metrics...')
    mh = mm.metrics.create()
    summary = mh.compute(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        name='OVERALL',
    )
    # end_t = time.time()
    # logger.info('Time taken: {:.3f}'.format(end_t - start_t))

    summary = summary.rename(columns=mm.io.motchallenge_metric_names)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters
    )
    print(strsummary)

    return summary, strsummary


def print_df(stats, name='', fmt='.3f', return_mode=0):
    # if name:
    #     print(f"\n\n{name}")
    #     print(f"\n\n{name}")
    txt = tabulate(stats, headers='keys', tablefmt="orgtbl", floatfmt=fmt)
    if return_mode:
        return txt
    print(txt)


def write_df(stats, out_path, index_label, title):
    if title:
        with open(out_path, 'a') as fid:
            fid.write(title + '\n')
    stats.to_csv(out_path, sep='\t', index_label=index_label, line_terminator='\n', mode='a')


def mot_metrics_to_file(eval_paths, summary, load_fname, seq_name,
                        mode='a', time_stamp='', verbose=1, devkit=0):
    """

    :param eval_paths:
    :param summary:
    :param load_fname:
    :param seq_name:
    :param mode:
    :param time_stamp:
    :param verbose:
    :return:
    """

    if not time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for eval_path in eval_paths:
        if verbose:
            print(f'{eval_path}')

        write_header = False
        if not os.path.isfile(eval_path):
            write_header = True

        with open(eval_path, mode) as eval_fid:
            if write_header:
                eval_fid.write('{:<50}'.format('timestamp'))
                eval_fid.write('\t{:<50}'.format('file'))
                for _metric, _type in zip(summary.columns.values, summary.dtypes):
                    if _type == np.int64:
                        eval_fid.write('\t{:>6}'.format(_metric))
                    else:
                        eval_fid.write('\t{:>8}'.format(_metric))
                if not devkit:
                    eval_fid.write('\t{:>10}'.format('MT(%)'))
                    eval_fid.write('\t{:>10}'.format('ML(%)'))
                    eval_fid.write('\t{:>10}'.format('PT(%)'))

                eval_fid.write('\n')
            eval_fid.write('{:13s}'.format(time_stamp))
            eval_fid.write('\t{:50s}'.format(load_fname))
            _values = summary.loc[seq_name].values
            # if seq_name == 'OVERALL':
            #     if verbose:
            #         print()

            for _val, _type in zip(_values, summary.dtypes):
                if _type == np.int64:
                    eval_fid.write('\t{:6d}'.format(int(_val)))
                else:
                    eval_fid.write('\t{:.6f}'.format(_val))
            if not devkit:
                try:
                    _gt = float(summary['GT'][seq_name])
                except KeyError:
                    pass
                else:
                    mt_percent = float(summary['MT'][seq_name]) / _gt * 100.0
                    ml_percent = float(summary['ML'][seq_name]) / _gt * 100.0
                    pt_percent = float(summary['PT'][seq_name]) / _gt * 100.0
                    eval_fid.write('\t{:3.6f}\t{:3.6f}\t{:3.6f}'.format(
                        mt_percent, ml_percent, pt_percent))

            eval_fid.write('\n')


def hota_metrics_to_file(eval_paths, summary, load_fname, metric_names,
                         mode='a', time_stamp='', verbose=1):
    """

    :param eval_paths:
    :param dict summary:
    :param load_fname:
    :param metric_names:
    :param mode:
    :param time_stamp:
    :param verbose:
    :return:
    """

    if not time_stamp:
        time_stamp = datetime.now().strftime("%y%m%d_%H%M%S")

    for eval_path in eval_paths:
        if verbose:
            print(f'{eval_path}')

        # header = None
        write_header = False
        if not os.path.isfile(eval_path):
            write_header = True
        # else:
        #     header = open(eval_path, 'r').readline()

        with open(eval_path, mode) as eval_fid:
            if write_header:
                eval_fid.write('{:<50}'.format('timestamp'))
                eval_fid.write('\t{:<50}'.format('file'))

                for _metric in metric_names:
                    eval_fid.write('\t{:>8}'.format(_metric))

                eval_fid.write('\n')
            # else:
            # header_items = [k.strip() for k in header.split('\t')]
            # """exclude time_stamp and load_fname"""
            # header_items = header_items[2:]

            eval_fid.write('{:13s}'.format(time_stamp))
            eval_fid.write('\t{:50s}'.format(load_fname))

            for _metric in metric_names:
                _val = summary[_metric]

                if isinstance(_val, str):
                    eval_fid.write('\t{:s}'.format(_val))
                else:
                    eval_fid.write('\t{:.3f}'.format(_val))

            eval_fid.write('\n')


def combine_hota_devkit_results(devkit_res, hota_res, devkit_metric_names, hota_metric_names):
    n_metrics = len(devkit_metric_names)

    seq_names = {k: os.path.splitext(os.path.basename(k))[0] for k in devkit_res}
    n_seq = len(seq_names)

    rows = ['{} / {}'.format(seq, _type) for _, seq in seq_names.items() for _type in ['devkit', 'HOTA']]
    n_rows = len(rows)

    hota_devkit_df = pd.DataFrame(
        np.empty((n_rows, n_metrics), dtype=np.float32),
        columns=devkit_metric_names, index=rows)

    for seq in devkit_res:
        seq_name = seq_names[seq]
        for devkit_metric_name, hota_metric_name in zip(devkit_metric_names, hota_metric_names):
            hota_devkit_df[devkit_metric_name]['{} / devkit'.format(seq_name)] = devkit_res[seq][
                devkit_metric_name]
            hota_devkit_df[devkit_metric_name]['{} / HOTA'.format(seq_name)] = hota_res[seq][hota_metric_name]

    return hota_devkit_df


def add_suffix(src_path, suffix):
    # abs_src_path = os.path.abspath(src_path)
    src_dir = os.path.dirname(src_path)
    src_name, src_ext = os.path.splitext(os.path.basename(src_path))
    dst_path = os.path.join(src_dir, src_name + '_' + suffix + src_ext)
    return dst_path


def most_recently_modified_dir(prev_results_dir, excluded=()):
    if isinstance(excluded, str):
        excluded = (excluded,)

    subdirs = [linux_path(prev_results_dir, k) for k in os.listdir(prev_results_dir)
               if os.path.isdir(linux_path(prev_results_dir, k)) and k not in excluded]

    subdirs_mtime = [os.path.getmtime(k) for k in subdirs]

    subdirs_sorted = sorted(zip(subdirs_mtime, subdirs))
    load_dir = subdirs_sorted[-1][1]

    return load_dir


def get_neighborhood(_score_map, cx, cy, r, _score_sz, type, thickness=1):
    if type == 0:

        # _max = -np.inf
        neighborhood = []

        x1, y1 = int(cx - r), int(cy - r)
        x2, y2 = int(cx + r), int(cy + r)

        # max_x1 = max_x2 = max_y1 = max_y2 = -np.inf
        incl_x1 = incl_y1 = incl_x2 = incl_y2 = 1

        if x1 < 0:
            x1 = 0
            incl_x1 = 0
        if y1 < 0:
            y1 = 0
            incl_y1 = 0
        if x2 >= _score_sz:
            x2 = _score_sz - 1
            incl_x2 = 0
        if y2 >= _score_sz:
            y2 = _score_sz - 1
            incl_y2 = 0

        if y2 >= y1:
            if incl_x1:
                # max_x1 = np.amax(_score_map[y1:y2 + 1, x1])
                # _max = max(_max, max_x1)
                x1 += 1
                neighborhood += list(_score_map[y1:y2 + 1, x1].flat)

            if incl_x2:
                # max_x2 = np.amax(_score_map[y1:y2 + 1, x2])
                # _max = max(_max, max_x2)
                x2 -= 1
                neighborhood += list(_score_map[y1:y2 + 1, x2].flat)

        if x2 >= x1:
            if incl_y1:
                # max_y1 = np.amax(_score_map[y1, x1:x2 + 1])
                # _max = max(_max, max_y1)
                neighborhood += list(_score_map[y1, x1:x2 + 1].flat)
            if incl_y2:
                # max_y2 = np.amax(_score_map[y2, x1:x2 + 1])
                # _max = max(_max, max_y2)
                neighborhood += list(_score_map[y2, x1:x2 + 1].flat)

        return np.asarray(neighborhood)
    elif type == 1:
        x = np.arange(0, _score_sz)
        y = np.arange(0, _score_sz)
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 >= r ** 2
        # _max = np.amax(_score_map[mask])

        neighborhood = _score_map[mask].flatten()
        return np.asarray(neighborhood)
        # return _max
    elif type == 2:
        mask = np.zeros_like(_score_map)
        cv2.circle(mask, (cx, cy), int(r), color=1, thickness=thickness)
        # _max = np.amax(_score_map[mask.astype(np.bool)])
        neighborhood = _score_map[mask.astype(np.bool)].flatten()

        return neighborhood
    else:
        raise AssertionError('Invalid neighborhood type: {}'.format(type))

    # if _max > 0:
    #     _conf = 1 - math.exp(-1.0 / _max)
    # else:
    #     _conf = 1

    # return _max


def get_patch(img, bbox, to_gs, out_size, context_ratio):
    min_x, min_y, w, h = np.asarray(bbox).squeeze()
    max_x, max_y = min_x + w, min_y + h

    if context_ratio > 0:
        ctx_w, ctx_h = int(w * context_ratio / 2.0), int(h * context_ratio / 2.0)
        min_x, min_y = min_x - ctx_w, min_y - ctx_h
        max_x, max_y = max_x + ctx_w, max_y + ctx_h

    img_h, img_w = img.shape[:2]
    min_x, max_x = clamp([min_x, max_x], 0, img_w)
    min_y, max_y = clamp([min_y, max_y], 0, img_h)

    if max_x < min_x + 1 or max_y < min_y + 1:
        assert out_size is not None, f"out_size must be provided to handle annoying invalid boxes like: {bbox}"

        patch = np.zeros(out_size, dtype=img.dtype)
        resized_patch = patch
    else:
        patch = img[int(min_y):int(max_y), int(min_x):int(max_x), ...]
        if to_gs and len(patch.shape) == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        if out_size is not None:
            resized_patch = cv2.resize(patch, dsize=out_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized_patch = patch

    if len(resized_patch.shape) == 2:
        resized_patch = np.expand_dims(resized_patch, axis=2)

    return patch, resized_patch


def spawn(dst, src):
    src_members = [a for a in dir(src) if not a.startswith('__') and not callable(getattr(src, a))]
    dst_members = [a for a in dir(dst) if not a.startswith('__') and not callable(getattr(dst, a))]

    members_to_spawn = list(filter(lambda a: a not in dst_members, src_members))
    # print(f'members_to_spawn:\n{pformat(members_to_spawn)}')
    for _member in members_to_spawn:
        setattr(dst, _member, getattr(src, _member))


def get_unique_ids(id_list, existing_ids=(), min_id=10000, max_id=99999, n_ids=10000):
    for random_id in random.sample(range(min_id, max_id), n_ids):
        if random_id in existing_ids:
            continue

        id_list.append(random_id)
        yield random_id


def ids_to_member_names(obj, unique_ids, set_none=1):
    id_attr = {getattr(obj, attr): attr for attr in dir(obj) if isinstance(getattr(obj, attr), int)}

    member_str = []
    for _id in unique_ids:
        member_str.append(id_attr[_id])
        if set_none:
            setattr(obj, id_attr[_id], None)

    return member_str


def load_samples_from_file(db_path, load_prev_paths, mem_mapped):
    # if os.path.isdir(db_path):
    #     db_path = linux_path(db_path, 'model.bin.npz')

    start_t = time.time()

    # db_dict = np.load(db_path, mmap_mode='r'
    #                   # if mem_mapped else None
    #                   )
    # features = db_dict['features']  # type: np.ndarray
    # labels = db_dict['labels']  # type: np.ndarray

    if not os.path.isdir(db_path):
        db_dir = os.path.dirname(db_path)
    else:
        db_dir = db_path

    mmap_mode = 'r' if mem_mapped else None

    features_path = linux_path(db_dir, 'features.npy')
    features = np.load(features_path, mmap_mode=mmap_mode)
    labels_path = linux_path(db_dir, 'labels.npy')
    labels = np.load(labels_path, mmap_mode=mmap_mode)
    samples = (features, labels)

    syn_features = syn_labels = None
    n_syn_samples = n_syn_pos_samples = n_syn_neg_samples = 0

    # n_syn_samples = n_syn_pos_samples = n_syn_neg_samples = None
    syn_features_path = linux_path(db_dir, 'synthetic_features.npy')
    if os.path.exists(syn_features_path):
        syn_labels_path = linux_path(db_dir, 'synthetic_labels.npy')
        assert os.path.exists(syn_labels_path), "syn_features_path exists but syn_labels_path does not"

        syn_features = np.load(syn_features_path, mmap_mode=mmap_mode)
        syn_labels = np.load(syn_labels_path, mmap_mode=mmap_mode)

    syn_samples = (syn_features, syn_labels)

    end_t = time.time()
    load_time = end_t - start_t

    n_samples = features.shape[0]

    assert labels.shape[0] == n_samples, f"Mismatch between n_samples in " \
        f"labels: {labels.shape[0]} and features: {n_samples}"

    pos_idx, neg_idx, n_pos_samples, n_neg_samples = get_class_idx(labels)

    # class_idx = (pos_idx, neg_idx)
    sample_counts = (n_samples, n_pos_samples, n_neg_samples)

    if syn_features is not None:
        n_syn_samples = syn_features.shape[0]
        assert syn_labels.shape[0] == n_syn_samples, f"Mismatch between n_syn_samples in " \
            f"labels: {syn_labels.shape[0]} and features: {n_syn_samples}"

        syn_pos_idx, syn_neg_idx, n_syn_pos_samples, n_syn_neg_samples = get_class_idx(syn_labels)
        # syn_class_idx = (syn_pos_idx, syn_neg_idx)

    syn_sample_counts = (n_syn_samples, n_syn_pos_samples, n_syn_neg_samples)

    if not n_samples:
        print(f'no samples found in {db_path}')
    else:
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / 1e9
        load_speed = float(n_samples) / load_time

        print('{}:'.format(db_path))
        print('{} samples ({} pos, {} neg)'.format(*sample_counts))
        if syn_features is not None:
            print('{} synthetic samples ({} pos, {} neg)'.format(*syn_sample_counts))

        print('{:.3f}, {:.3f} samples/sec, {:.3f} GB\n'.format(load_time, load_speed, memory_gb))

    if not load_prev_paths:
        return samples, n_samples, syn_samples, n_syn_samples

    prev_paths_path = linux_path(db_dir, 'prev_paths.npy')
    if os.path.exists(prev_paths_path):
        prev_paths = np.load(prev_paths_path, mmap_mode=mmap_mode)
        prev_paths = [k for k in prev_paths if k != db_path]
    else:
        print(f'No prev_db_paths found')
        prev_paths = []

    return samples, n_samples, syn_samples, n_syn_samples, prev_paths


class SCP:
    class Auth:
        local_url = ""
        global_url = ""
        user = ""
        pwd = ""
        alias = ""

        def __init__(self, _dict):
            self.__dict__.update(_dict)

    def __init__(self):
        self.auth_file = "MtMAz84Mcla1frp.txt"
        self.key_file = "NSlrq8TzxQA5tiO.txt"
        self.auth_dir = ""
        self.global_server = 'grs'

        self.home_dir = "/home/abhineet"
        self.code_path = 'deep_mdp/tracking_module'
        self.auth_data = {}

        self.enable_zipping = 1
        self.remove_zip = 1
        self.exclude_exts = ['pt', 'npz', 'npy', 'jpg', 'png', 'mp4', 'mkv', 'avi', 'zip']

    def read_auth(self):
        if not self.auth_file:
            print("auth_file is not provided")
            return

        if not self.auth_dir:
            self.auth_dir = self.home_dir

        auth_path = linux_path(self.auth_dir, self.auth_file)
        auth_path = os.path.abspath(auth_path)

        # auth_dir_files = os.listdir(self.auth_dir)

        assert os.path.isfile(auth_path), "auth_path does not exist: {}".format(auth_path)

        if self.key_file:
            from cryptography.fernet import Fernet

            key_path = linux_path(self.auth_dir, self.key_file)
            key_path = os.path.abspath(key_path)
            assert os.path.isfile(key_path), "key_path does not exist: {}".format(key_path)

            key = open(key_path, "rb").read()

            f = Fernet(key)
            with open(auth_path, "rb") as file:
                encrypted_data = file.read()
            auth_data_b = f.decrypt(encrypted_data)
            auth_data = auth_data_b.decode("utf-8")
            auth_data_lines = auth_data.splitlines()
        else:
            auth_data = open(auth_path, 'r').readlines()
            auth_data_lines = [k.strip() for k in auth_data]

        for line in auth_data_lines:
            name, alias, global_url, local_url, user, pwd = line.split('\t')
            if alias == '_':
                alias = ''
            if global_url == '_':
                global_url = ''

            self.auth_data[name] = SCP.Auth(dict(
                alias=alias,
                global_url=global_url,
                local_url=local_url,
                user=user,
                pwd=pwd
            ))
            if alias:
                self.auth_data[alias] = SCP.Auth(dict(
                    alias=name,
                    global_url=global_url,
                    local_url=local_url,
                    user=user,
                    pwd=pwd
                ))


def copy_rgb(_frame):
    if len(_frame.shape) == 2:
        frame_disp = cv2.cvtColor(_frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_disp = np.copy(_frame)
    return frame_disp


def get_from_remote(params, server_name, dst_rel_path, is_file, only_txt=0, zip_name='get_from_remote'):
    """

    :param SCP params:
    :param server_name:
    :param dst_rel_path:
    :param is_file:
    :param zip_name:
    :return:
    """

    assert params.auth_data, "remote auth data not provided"
    assert server_name in params.auth_data, "server_name: {} not found in  auth data".format(server_name)

    hostname = socket.gethostname()

    remote_home_dir = params.home_dir

    print('getting {} from {}'.format(dst_rel_path, server_name))

    from sys import platform

    if hostname in params.auth_data:
        """local URL mode"""
        assert server_name not in (hostname, params.auth_data[hostname].alias), \
            "remote server is same as the current host"
        remote = params.auth_data[server_name]  # type: SCP.Auth
        remote_url = remote.local_url
    else:
        """global URL/Samba mode"""
        if server_name not in (params.global_server, params.auth_data[params.global_server].alias):
            remote_home_dir = linux_path(remote_home_dir, "samba_{}".format(server_name))
        remote = params.auth_data[params.global_server]  # type: SCP.Auth
        remote_url = remote.global_url

    if platform in ("linux", "linux2"):
        scp_base_cmd = "sshpass -p {} scp -r -P 22 {}@{}".format(remote.pwd, remote.user, remote_url)
    else:
        # scp_name = 'pscp'
        scp_base_cmd = "pscp -pw {} -r -P 22 {}@{}".format(remote.pwd, remote.user, remote_url)

    src_root_path = linux_path(remote_home_dir, params.code_path)

    if not is_file:
        os.makedirs(dst_rel_path, exist_ok=True)

    if not params.enable_zipping:
        scp_path = linux_path(src_root_path, dst_rel_path + '/*')

        scp_cmd = "{}:{} {}".format(scp_base_cmd, scp_path, dst_rel_path + '/')

        print('Running {}'.format(scp_cmd))
        scp_cmd_list = scp_cmd.split(' ')
        subprocess.check_call(scp_cmd_list)
        return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    zip_fname = "{}_{}_{}.zip".format(zip_name, server_name, timestamp)
    zip_path = linux_path(params.home_dir, zip_fname)

    zip_cmd = "cd {} && zip -r {} {}".format(src_root_path, zip_path, dst_rel_path)

    if only_txt:
        exclude_switches = ' '.join(f'-x "*.{ext}"' for ext in params.exclude_exts)
        zip_cmd = "{} {}".format(zip_cmd, exclude_switches)

    print('Running {}'.format(zip_cmd))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(remote_url, username=remote.user, password=remote.pwd)

    stdin, stdout, stderr = client.exec_command(zip_cmd)

    stdout = list(stdout)
    for line in stdout:
        print(line.strip('\n'))

    stderr = list(stderr)

    if stderr:
        raise AssertionError('remote command did not work' + '\n' + '\n'.join(stderr))

    client.close()

    scp_cmd = "{}:{} ./".format(scp_base_cmd, zip_path)
    print('Running {}'.format(scp_cmd))
    scp_cmd_list = scp_cmd.split(' ')
    subprocess.check_call(scp_cmd_list)

    with ZipFile(zip_fname, 'r') as zipObj:
        zipObj.extractall()

    if params.remove_zip:
        os.remove(zip_fname)


def id_token_to_cmd_args(id_token, params, tee_dir='log/tee', open_tee=0):
    if '::' in id_token:
        _, server_token = id_token.split(' :: ')
    else:
        server_token = id_token

    server_name, tmux_pane, tee_id = server_token.split(':')

    tee_ansi_path = linux_path(tee_dir, tee_id + '.ansi')

    if not os.path.exists(tee_ansi_path):
        params.read_auth()

        tee_zip_path = linux_path(tee_dir, tee_id + '.zip')
        if not os.path.exists(tee_zip_path):
            print(
                'neither tee_ansi_path: {} nor tee_zip_path: {} exist'.format(tee_ansi_path, tee_zip_path))
            try:
                get_from_remote(params, server_name, tee_zip_path, is_file=1)
            except:
                get_from_remote(params, server_name, tee_ansi_path, is_file=1)

        if os.path.exists(tee_zip_path):
            with ZipFile(tee_zip_path, 'r') as zipObj:
                zipObj.extractall(tee_dir)

            if params.remove_zip:
                os.remove(tee_zip_path)

    assert os.path.exists(tee_ansi_path), "tee_ansi_path does not exist: {}".format(tee_ansi_path)

    print('\nreading command–line arguments from {}\n'.format(tee_ansi_path))

    tee_data = open(tee_ansi_path, 'r').readlines()

    while True:
        if tee_data[0].startswith('main.py '):
            break
        del tee_data[0]

    cmd = tee_data[0].strip()

    if open_tee:
        tee_cmd = "start {}".format(tee_ansi_path)
        os.system(tee_cmd)

    cmd_args = cmd.split(' ')[1:-2]

    return cmd_args, tee_id


def check_load_fnames(load_fnames, run_params, _data, _logger):
    """

    :param load_fnames:
    :param run_params:
    :param _data: Data
    :param _logger:
    :return:
    """
    start_id = run_params.start
    run_seq_ids = run_params.seq[start_id:]
    run_seq_set = _data.sets[run_params.seq_set]
    run_seq_names = [_data.sequences[run_seq_set][seq_id][0] for seq_id in run_seq_ids]

    if len(run_seq_ids) != len(load_fnames):
        print("mismatch between the lengths of run sequences and available load filenames")
        return False

    for _id, run_id in enumerate(run_seq_ids):
        if not _data.initialize(run_params.seq_set, run_id, 1, silent=1):
            raise AssertionError('Data module failed to initialize with sequence {:d}'.format(run_id))

        if run_params.subseq_postfix:
            _load_fname_template = '{:s}_{:d}_{:d}'.format(_data.seq_name, _data.start_frame_id + 1,
                                                           _data.end_frame_id + 1)
        else:
            _load_fname_template = '{:s}'.format(_data.seq_name)

        load_fname = [k for k in load_fnames if k.startswith(_load_fname_template)]
        if len(load_fname) != 1:
            print("unique match for the load filename {} not found".format(_load_fname_template))
            return False

    return True


def interpolate_missing_objects(data):
    """sort by object ID"""
    data = data[data[:, 1].argsort(kind='mergesort')]

    out_data = []

    # obj_ids = list(np.unique(data[:, 1], return_counts=False))

    n_data, data_dim = data.shape[:2]

    assert n_data > 0, "too few entries in data"

    if data_dim > 10:
        enable_state_info = 1
    else:
        enable_state_info = 0

    interpolated_ids = []
    for data_id in range(n_data - 1):

        curr_data, next_data = data[data_id, :], data[data_id + 1, :]
        curr_frame_id, curr_obj_id = curr_data[:2].astype(np.int)
        next_frame_id, next_obj_id = next_data[:2].astype(np.int)

        out_data.append(curr_data)

        if curr_obj_id != next_obj_id:
            continue

        assert next_frame_id > curr_frame_id, "multiple objects with the same ID found in the same frame"

        frame_diff = next_frame_id - curr_frame_id

        if frame_diff == 1:
            continue

        curr_location = curr_data[2:6]
        next_location = next_data[2:6]

        curr_score = curr_data[6]
        next_score = next_data[6]

        interpolated_ids.append(curr_obj_id)

        # linear interpolation
        for frame_id in range(curr_frame_id + 1, next_frame_id):
            interp_factor = float(frame_id - curr_frame_id) / float(frame_diff)
            interp_location = curr_location + ((next_location - curr_location) * interp_factor)
            interp_score = curr_score + ((next_score - curr_score) * interp_factor)

            x, y, w, h = interp_location

            interp_data = [frame_id, curr_obj_id, x, y, w, h, interp_score] + list(curr_data[7:])

            if enable_state_info:
                interp_data[10] = MDPStates.lost

            out_data.append(interp_data)

    last_data = data[-1, :]
    out_data.append(last_data)

    out_data = np.asarray(out_data)

    n_interpolations = len(interpolated_ids)

    unique_interpolated_ids = np.unique(interpolated_ids)

    n_unique = len(unique_interpolated_ids)

    n_out_data = out_data.shape[0]

    print('interpolated tracking results: {} --> {} with {} interpolations for {} IDs'.format(
        n_data, n_out_data, n_interpolations, n_unique))

    return out_data


def associate_gt_to_detections(frame, frame_id, annotations, det_data, n_det,
                               ann_to_targets, start_end_frame_ids,
                               ann_assoc_method, ann_iou_threshold,
                               vis, verbose, _logger):
    # vis = 1

    ann_status = [None, ] * n_det
    ann_obj_ids = [None, ] * n_det
    ann_traj_ids = [None, ] * n_det

    if annotations is None:
        return ann_obj_ids, ann_traj_ids, ann_status

    """all annotations in this frame"""
    curr_ann_idx = annotations.idx[frame_id]

    if curr_ann_idx is None:
        """no annotations in this frame"""
        ann_status = ['fp_background', ] * n_det
        return ann_obj_ids, ann_traj_ids, ann_status

    # n_gt = curr_ann_idx.size

    curr_ann_data = annotations.data[curr_ann_idx, :]

    if ann_assoc_method < 2:
        """associate each det to max overlapping GT"""
        associated_gts = []
        det_to_gt = {}
        for _det_id in range(n_det):

            curr_det_data = det_data[_det_id, :]

            # curr_ann_data = _curr_ann_data.squeeze().reshape((-1, 10))
            max_iou, max_iou_idx = get_max_overlap_obj(curr_ann_data, curr_det_data[2:6].reshape((1, 4)),
                                                       _logger)
            if (ann_assoc_method == 0 or max_iou_idx not in associated_gts) and \
                    max_iou >= ann_iou_threshold:
                det_to_gt[_det_id] = max_iou_idx
                associated_gts.append(max_iou_idx)

            if vis == 2:
                frame_disp = np.copy(frame)
                draw_box(frame_disp, curr_det_data[2:6], color='blue')

                for _idx in range(curr_ann_data.shape[0]):
                    col = 'green' if _idx == max_iou_idx else 'red'
                    draw_box(frame_disp, curr_ann_data[_idx, 2:6], color=col)

                annotate_and_show('_get_target_status', frame_disp, text='max_iou: {:.2f}'.format(max_iou))
    else:
        if ann_assoc_method == 3:
            use_hungarian = 1
        else:
            use_hungarian = 0

        curr_ann_data = annotations.data[curr_ann_idx, :]
        det_to_gt, gt_to_det, unassociated_dets, unassociated_gts = find_associations(
            frame, det_data[:, 2:6], curr_ann_data[:, 2:6], ann_iou_threshold, use_hungarian=use_hungarian, vis=vis)

    for _det_id in range(n_det):
        try:
            _gt_id = det_to_gt[_det_id]
        except KeyError:
            ann_status[_det_id] = 'fp_background'
            continue

        ann_obj_ids[_det_id] = int(curr_ann_data[_gt_id, 1])
        ann_traj_ids[_det_id] = annotations.obj_to_traj[ann_obj_ids[_det_id]]

        _ann_obj_id = ann_obj_ids[_det_id]
        _ann_traj_id = ann_traj_ids[_det_id]

        """detection corresponds to a real object"""
        if _ann_obj_id in ann_to_targets:
            _target_id, _target = ann_to_targets[_ann_obj_id]  # type: int, Target
            if _target is None:
                _ann_status = "fp_deleted"
                if verbose:
                    _logger.debug(
                        f'adding fp_deleted target on annotation: {_ann_obj_id} '
                        f'with target {_target_id} '
                        f'previously active in frames {start_end_frame_ids[_target_id]}')
            else:
                if _target.start_frame_id != frame_id:
                    """another target was added in an earlier frame so this is a false positive 
                    which should  be identified by active policy
                     """
                    _ann_status = "fp_apart"
                    if verbose:
                        _logger.debug(
                            f'adding fp_apart target on annotation: {_ann_obj_id} with target '
                            f'{_target.id_} added in frame {_target.start_frame_id}')
                else:
                    """another target was added in the same frame so a failure of NMS
                     """
                    _ann_status = "fp_concurrent"
                    if verbose:
                        _logger.debug(
                            f'adding duplicate target on existing annotation: {_ann_obj_id} '
                            f'with target {_target.id_}')
        else:
            if verbose:
                _logger.debug(f'adding tp target for annotation: {_ann_obj_id}')
            _ann_status = "tp"

        ann_status[_det_id] = _ann_status

    return ann_obj_ids, ann_traj_ids, ann_status
