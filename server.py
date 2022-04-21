import numpy as np
import argparse
import os
import socket
import time
import cv2
import threading
import sys
import xlwt
import pandas as pd

from run import Train, Test
from trainer import Trainer
from tester import Tester
from visualizer import Visualizer, ObjTypes

from trackers.patch_tracker import PatchTracker, PatchTrackerParams
from objects import Annotations, Detections
from Gate import Gate, GateParams
from utilities import add_params_to_parser, process_args_from_parser, get_date_time, linux_path

script_dir = os.path.dirname(os.path.realpath(__file__))
script_parent_dir = os.path.dirname(script_dir)
sys.path.append(script_parent_dir)

print(sys.path)

from libs.frames_readers import get_frames_reader
from libs.netio import bindToPort
# from utils.frames_readers import get_frames_reader
from utils.netio import send_msg_to_connection, recv_from_connection


# from functools import partial

class ServerParams:
    """
    :type mode: int
    :type load_path: str
    :type continue_training: int | bool
    :type gate: GateParams
    :type patch_tracker: PatchTrackerParams
    :type visualizer: Visualizer.Params
    """

    def __init__(self):
        self.mode = 0
        self.load_path = 'trained_target.zip'
        self.save_path = ''

        self.port = 3002
        self.skip_thresh = 10
        self.verbose = 0

        self.train = Train.Params()
        self.test = Test.Params()

        self.gate = GateParams()
        self.patch_tracker = PatchTrackerParams()
        self.visualizer = Visualizer.Params()

        self.help = {
            'mode': 'server mode: '
                    '0: disable '
                    '1: testing '
                    '2: training',
            'load_path': 'location of the zip file from where the pre-trained target is to be loaded',
            'save_path': 'location of the zip file from where the trained target is to be saved',
            'continue_training': 'continue training a previously trained target loaded from trained_target_path',
            'port': 'port on which the server listens for requests',
            'skip_thresh': 'maximum number of attempts to look for the detections for a particular frame '
                           'before skipping to the next one',
            'verbose': 'show detailed diagnostic messages',
            'gate': 'parameters for the Gate module',
            'patch_tracker': 'parameters for the patch tracker module',
        }


class Server:
    """
    :type params: ServerParams
    :type trainer_params: TrainerParams
    :type tester_params: Tester.Params
    :type logger: logging.RootLogger
    :type gates: dict[int:Gate]
    """

    def __init__(self, params, trainer_params, tester_params, _logger):
        """
        :type params: ServerParams
        :type trainer_params: TrainerParams
        :type tester_params: Tester.Params
        :type logger: logging.RootLogger
        :rtype: None
        """

        self.params = params
        self.trainer_params = trainer_params
        self.tester_params = tester_params
        self.logger = _logger

        visualizer_mode = list(self.tester_params.visualizer.mode)
        # no annotations available for visualization
        visualizer_mode[2] = 0
        self.tester_params.visualizer.mode = tuple(visualizer_mode)

        self.request_dict = {}
        self.request_list = []

        self.current_path = None
        self.frames_reader = None
        self.trainer = None
        self.tester = None
        self.visualizer = None
        self.enable_visualization = False
        self.traj_data = []

        self.trained_target = None
        self.tracking_res = None
        self.index_to_name_map = None

        self.tracker_id = 0
        self.max_frame_id = -1
        self.frame_id = -1

        self.request_lock = threading.Lock()

        self.gates = {}
        # pairs of gates that share an intersecting target
        self.gate_pairs = {}
        # last known locations of all targets
        self.target_centers = {}

        # create parsers for real time parameter manipulation
        self.parser = argparse.ArgumentParser()
        add_params_to_parser(self.parser, self.params)

        self.tester_parser = argparse.ArgumentParser()
        add_params_to_parser(self.tester_parser, self.tester_params)

        self.trainer_parser = argparse.ArgumentParser()
        add_params_to_parser(self.trainer_parser, self.trainer_params)

        # self.patch_tracking_results = []

    def parseParams(self, obj, parser, cmd_args, name):
        args_in = []
        # check for a custom cfg file specified at command line
        prefix = '{:s}.'.format(name)
        if len(cmd_args) > 0 and '--cfg' in cmd_args[0]:
            _, arg_val = cmd_args[0].split('=')
            cfg = arg_val
            print('Reading {:s} parameters from {:s}'.format(name, cfg))
            if os.path.isfile(cfg):
                file_args = open(cfg, 'r').readlines()
                # lines starting with # in the cfg file are regarded as comments and thus ignored
                file_args = ['--{:s}'.format(arg.strip()[len(prefix):]) for arg in file_args if
                             arg.startswith(prefix)]
                # print('file_args', file_args)
                args_in += file_args
        # command line arguments override those in the cfg file
        args_in += ['--{:s}'.format(arg[len(prefix) + 2:]) for arg in cmd_args if
                    arg.startswith('--{:s}'.format(prefix))]
        # args_in = [arg[len(prefix):] for arg in args_in if prefix in arg]
        # print('args_in', args_in)
        args, _ = parser.parse_known_args(args_in)
        process_args_from_parser(obj, args)
        return args_in

    def patchTracking(self, request, mtf_args=''):

        request_path = request["path"]
        request_roi = request["roi"]

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path

        id_number = request['id_number']

        tracker = PatchTracker(self.params.patch_tracker, self.logger,
                               id_number, self.tracker_id, mtf_args)
        if not tracker.is_created:
            return

        self.tracker_id += 1

        init_frame_id = request["frame_number"]
        init_frame = self.frames_reader.get_frame(init_frame_id)

        init_bbox = request["bbox"]
        tracker.initialize(init_frame, init_bbox)

        if not tracker.is_initialized:
            raise AssertionError('Tracker initialization was unsuccessful')

        label = request['label']
        request_port = request["port"]

        n_frames = self.frames_reader.num_frames
        # self.logger.info('Tracking target {:d} in sequence with {:d} frames '
        #                  'starting from frame {:d}'.format(
        #     id_numbers[0], n_frames, init_frame_id + 1))

        for frame_id in range(init_frame_id + 1, n_frames):
            curr_frame = self.frames_reader.get_frame(frame_id)

            tracker.update(curr_frame, frame_id)
            if tracker.out_bbox is None:
                raise AssertionError('Tracker update was unsuccessful')

            if len(curr_frame.shape) == 3:
                height, width, channels = curr_frame.shape
            else:
                height, width = curr_frame.shape
                channels = 1

            tracking_result = dict(
                action="add_bboxes",
                path=request_path,
                frame_number=frame_id,
                width=width,
                height=height,
                channel=channels,
                bboxes=[tracker.out_bbox],
                scores=[0],
                labels=[label],
                id_numbers=[id_number],
                bbox_source="single_object_tracker",
                last_frame_number=frame_id - 1,
                trigger_tracking_request=False,
                num_frames=1,
                # port=request_port,
            )
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', request_port))
            send_msg_to_connection(tracking_result, sock)
            sock.close()

            # self.single_object_tracking_results.append(tracking_result)

            if tracker.is_terminated:
                break

    def visualize(self, request):
        request_path = request["path"]
        csv_path = request["csv_path"]
        class_dict = request["class_dict"]
        request_roi = request["roi"]
        init_frame_id = request["frame_number"]

        save_fname_templ = os.path.splitext(os.path.basename(request_path))[0]

        df = pd.read_csv(csv_path)

        if request_path != self.current_path:
            self.frames_reader = get_frames_reader(request_path)
            if request_roi is not None:
                self.frames_reader.setROI(request_roi)
            self.current_path = request_path

        # print('self.params.visualizer.save: ', self.params.visualizer.save)

        visualizer = Visualizer(self.params.visualizer, self.logger)
        init_frame = self.frames_reader.get_frame(init_frame_id)

        height, width, _ = init_frame.shape
        frame_size = width, height
        visualizer.initialize(save_fname_templ, frame_size)

        n_frames = self.frames_reader.num_frames
        for frame_id in range(init_frame_id, n_frames):
            try:
                curr_frame = self.frames_reader.get_frame(frame_id)
            except IOError as e:
                print('{}'.format(e))
                break

            multiple_instance = df.loc[df['frame_id'] == frame_id]
            # Total # of object instances in a file
            no_instances = len(multiple_instance.index)
            # Remove from df (avoids duplication)
            df = df.drop(multiple_instance.index[:no_instances])

            frame_data = []

            for instance in range(0, len(multiple_instance.index)):
                target_id = multiple_instance.iloc[instance].loc['target_id']
                xmin = multiple_instance.iloc[instance].loc['xmin']
                ymin = multiple_instance.iloc[instance].loc['ymin']
                xmax = multiple_instance.iloc[instance].loc['xmax']
                ymax = multiple_instance.iloc[instance].loc['ymax']
                class_name = multiple_instance.iloc[instance].loc['class']
                class_id = class_dict[class_name]

                width = xmax - xmin
                height = ymax - ymin

                frame_data.append([frame_id, target_id, xmin, ymin, width, height, class_id])

            frame_data = {ObjTypes.tracking_result: np.asarray(frame_data)}
            if not visualizer.update(frame_id, curr_frame, [None, None, frame_data]):
                break

        visualizer.close()

    def request_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bindToPort(sock, self.params.port, 'tracking')
        sock.listen(1)
        self.logger.info('Started tracking server')
        while True:
            try:
                connection, addr = sock.accept()
                connection.settimeout(None)
                msg = recv_from_connection(connection)
                connection.close()
                if isinstance(msg, list):
                    raw_requests = msg
                else:
                    raw_requests = [msg]
                for request in raw_requests:
                    # print('request: ', request)
                    request_type = request['request_type']
                    if request_type == 'patch_tracking':
                        cmd_args = request['cmd_args']
                        _args = self.parseParams(self.params, self.parser, cmd_args, 'server')

                        # sys.stdout.write('_args:\n {}\n'.format(_args))
                        # sys.stdout.flush()

                        mtf_prefix = '--patch_tracker.mtf.'
                        mtf_args_list = [k.replace(mtf_prefix, '') for k in _args if k.startswith(mtf_prefix)]
                        mtf_args = ''
                        for _arg in mtf_args_list:
                            mtf_args += ' {}'.format(_arg)

                        mtf_args = mtf_args.strip().replace('=', ' ')
                        # if mtf_args:
                        #     sys.stdout.write('MTF parameters:\n {}\n'.format(mtf_args))
                        #     sys.stdout.flush()

                        self.patchTracking(request, mtf_args)
                        # threading.Thread(target=partial(
                        #     self.singleObjectTRacking, request)).start()
                    elif request_type == 'start_tracking':
                        cmd_args = request['cmd_args']
                        self.frame_id = request['start_frame_id']
                        self.parseParams(self.params, self.parser, cmd_args, 'server')
                        self.parseParams(self.tester_params, self.tester_parser, cmd_args, 'tester')
                        self.parseParams(self.trainer_params, self.trainer_parser, cmd_args, 'trainer')

                        if self.params.save_path:
                            self.tester_params.visualizer.save_prefix = \
                                os.path.splitext(os.path.basename(self.params.save_path))[0]

                        self.tester = None
                        self.tracking_res = None
                        self.trained_target = None

                        self.logger.info('Started multi object tracker')
                        print('self.frame_id', self.frame_id)
                    elif request_type == 'visualize':
                        self.parseParams(self.params, self.parser, '', 'server')
                        self.visualize(request)
                    else:
                        if self.frame_id < 0:
                            self.logger.error('Tracker must be started before add_bboxes requests can be processed')
                            continue

                        frame_number = request['frame_number']
                        new_request = dict(
                            request_type="add_bboxes",
                            path=request['path'],
                            frame_number=request['frame_number'],
                            port=request['port'],
                            last_frame_number=request['last_frame_number'],
                            bboxes=request['bboxes'],
                            labels=request['labels'],
                            scores=request['scores'],
                            roi=request['roi'],
                            id_numbers=request['id_numbers'],
                            gates=request['gates'],
                            num_frames=request['num_frames'],
                        )
                        if frame_number > self.max_frame_id:
                            self.max_frame_id = frame_number

                        # frame_id = request['frame_number']
                        # if frame_id == self.frame_id:
                        #     with self.request_lock:
                        #         self.request_list.append(new_request)
                        #         self.frame_id += 1
                        # elif self.frame_id in self.request_dict:
                        #     with self.request_lock:
                        #         self.request_list.append(self.request_dict[self.frame_id].copy())
                        #         # del self.request_dict[self.frame_id]
                        #         self.frame_id += 1
                        # else:
                        #     self.request_dict[frame_id] = new_request

                        self.request_dict[frame_number] = new_request
                        # with self.request_lock:
                        # print('Received request for frame {:d} with {:d} boxes'.format(
                        #     request['frame_number'], len(request['bboxes'])))
                        # print(msg)
                        # self.request_dict[request['frame_number']] = new_request

            except (KeyboardInterrupt, SystemExit):
                return

    def processing_loop(self):
        skip_count = 0
        while True:
            try:
                # if len(self.single_object_tracking_results) > 0:
                #     tracking_result = self.single_object_tracking_results.pop(0)
                #     request_port = tracking_result['port']
                #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #     sock.connect(('localhost', request_port))
                #     send_msg_to_connection(tracking_result, sock)
                #     sock.close()

                if self.frame_id >= 0 and len(self.request_dict) > 0:
                    # with self.request_lock:
                    #     request = self.request_list.pop(0)

                    try:
                        # with self.request_lock:
                        request = self.request_dict[self.frame_id]
                        self.frame_id += 1
                        skip_count = 0
                        # del self.request_dict[self.frame_id]
                    except KeyError:
                        # self.logger.info('Request for frame {:d} not found'.format(self.frame_id))
                        skip_count += 1
                        if self.frame_id < self.max_frame_id and skip_count > self.params.skip_thresh:
                            # write('Skipping frame {}\n'.format(self.frame_id))
                            self.frame_id += 1
                            skip_count = 0
                        time.sleep(0.01)
                        continue
                else:
                    time.sleep(0.01)
                    continue

                if request["request_type"] != "add_bboxes":
                    continue

                request_path = request["path"]
                request_frame_number = request["frame_number"]

                # print('Processing request for frame {:d}'.format(request['frame_number']))

                request_port = request["port"]
                request_roi = request["roi"]

                if request_path != self.current_path:
                    self.frames_reader = get_frames_reader(request_path)
                    if request_roi is not None:
                        self.frames_reader.setROI(request_roi)
                    self.current_path = request_path
                current_frame = self.frames_reader.get_frame(request_frame_number)
                if len(current_frame.shape) == 3:
                    height, width, channels = current_frame.shape
                else:
                    height, width = current_frame.shape
                    channels = 1

                # print('width: ', width)
                # print('height: ', height)
                # print('current_frame.shape: ', current_frame.shape)

                bboxes = request['bboxes']
                scores = request['scores']
                gates = request['gates']

                # multi object tracking
                last_frame_number = request['last_frame_number']
                if request_frame_number >= last_frame_number:
                    print('All frames processed')
                    self.close(gates)
                    continue

                frame_size = (width, height)
                frame_id = request_frame_number

                for gate in gates:
                    _id = gate['id']
                    pt1, pt2 = gate['pts']
                    if _id in self.gates and self.gates[_id].isSame(pt1, pt2):
                        continue
                    self.gates[_id] = Gate(self.params.gate, _id, pt1, pt2, self.logger)

                n_det = len(bboxes)
                if self.params.verbose:
                    print("Handling request for frame {:d} with {:d} detections".format(
                        request_frame_number + 1, n_det))
                # print('n_det: ', n_det)
                # print('bboxes: ', bboxes)
                det_data = np.zeros((n_det, 10))
                for idx in range(n_det):
                    box = bboxes[idx]
                    det_data[idx, 0] = request_frame_number
                    det_data[idx, 1] = -1
                    det_data[idx, 2] = box['xmin']
                    det_data[idx, 3] = box['ymin']
                    det_data[idx, 4] = box['xmax'] - det_data[idx, 2] + 1
                    det_data[idx, 5] = box['ymax'] - det_data[idx, 3] + 1
                    det_data[idx, 6] = scores[idx]
                    det_data[idx, 7:9] = -1
                # print('det_data: ', det_data)
                _gates = self.gates if gates else None

                # print('gates', gates)
                # print('_gates', _gates)

                # multi object tracking
                if self.params.test.load:
                    if self.tracking_res is None:
                        num_frames = request['num_frames']
                        self.logger.info('Reading tracking results from {:s}'.format(self.params.load_path))
                        self.tester_params.input.tracking_res.path = self.params.load_path
                        self.tracking_res = Annotations(self.tester_params.input.tracking_res,
                                                        self.logger, obj_type='Tracking Results')
                        self.tracking_res.initialize(num_frames)
                        if not self.tracking_res.read(build_index=True):
                            self.logger.error('Tracking results could not be read')
                            return False
                        self.visualizer = Visualizer(self.tester_params.visualizer, self.logger)
                        self.enable_visualization = np.array(self.tester_params.visualizer.mode).any() and \
                                                    (self.tester_params.visualizer.save or
                                                     self.tester_params.visualizer.show)
                        # initialize visualizer
                        self.traj_data = []
                        save_fname_templ = 'server_loaded'
                        if self.enable_visualization and not self.visualizer.initialize(
                                save_fname_templ, frame_size):
                            self.logger.error('Visualizer could not be initialized')
                            return False

                    # if self.frame_id >= 0 and frame_id < self.frame_id:
                    #     self.logger.error('Frame {:d} has already been processed (latest frame_id: {:d})'.format(
                    #         frame_id, self.frame_id))
                    #     continue
                    # self.frame_id += 1

                    try:
                        curr_res_idx = self.tracking_res.idx[request_frame_number]
                    except KeyError:
                        curr_res_idx = []
                    if curr_res_idx is None:
                        curr_res_idx = []
                    n_boxes = len(curr_res_idx)
                    out_bboxes = [None] * n_boxes
                    out_ids = [None] * n_boxes
                    out_scores = [0] * n_boxes
                    out_labels = ['Vehicle'] * n_boxes
                    res_data = self.tracking_res.data
                    for idx, res_id in enumerate(curr_res_idx):
                        out_ids[idx] = int(res_data[res_id, 1])
                        xmin = int(res_data[res_id, 2])
                        ymin = int(res_data[res_id, 3])
                        width = int(res_data[res_id, 4])
                        height = int(res_data[res_id, 5])
                        out_scores[idx] = res_data[res_id, 6]

                        xmax = int(xmin + width - 1)
                        ymax = int(ymin + height - 1)

                        out_bboxes[idx] = dict(
                            xmin=xmin,
                            ymin=ymin,
                            xmax=xmax,
                            ymax=ymax,
                        )
                    if self.enable_visualization:
                        frame_data = {}
                        if self.tester_params.visualizer.mode[0]:
                            tracked_data = res_data[curr_res_idx, :]
                            frame_data[ObjTypes.tracking_result] = tracked_data
                        if self.tester_params.visualizer.mode[1]:
                            frame_data[ObjTypes.detection] = det_data
                        if not self.visualizer.update(frame_id, current_frame, frame_data, _gates):
                            self.logger.error('Visualizer update failed')
                            self.visualizer.close()
                            self.tracking_res = None
                else:
                    if self.tester is None:
                        if self.trained_target is None:
                            trainer = Trainer(self.trainer_params, self.logger)
                            filename, file_extension = os.path.splitext(self.params.load_path)
                            trainer.load(filename)
                            self.trained_target = trainer.target
                            if self.trained_target is None:
                                self.logger.error('Trained target could not be loaded')
                                return
                            if not self.trained_target.add_sequence(frame_size):
                                self.logger.error('Adding sequence to trained target failed')
                                return
                        self.tester = Tester(self.trained_target, self.tester_params, self.logger)
                        # initialize tester
                        if not self.tester.initialize(save_fname_templ='server',
                                                      frame_size=frame_size):
                            self.logger.error('Tester could not be initialized')
                            return
                    # if self.frame_id >= 0 and frame_id < self.frame_id:
                    #     self.logger.error('Frame {:d} has already been processed (latest frame_id: {:d})'.format(
                    #         frame_id, self.frame_id))
                    #     continue
                    # self.frame_id += 1

                    result = []
                    if not self.tester.update(request_frame_number, current_frame,
                                              det_data, n_det, _gates, result):
                        self.logger.error('Tester update failed')
                        self.close(gates)
                        return
                    n_boxes = len(result)
                    if self.params.verbose:
                        print("Obtained {:d} tracked boxes for frame {:d}".format(
                            n_boxes, request_frame_number + 1))

                    out_bboxes = [None] * n_boxes
                    out_ids = [None] * n_boxes
                    out_scores = [0] * n_boxes
                    out_labels = ['Vehicle'] * n_boxes
                    for idx, res_data in enumerate(result):
                        out_ids[idx] = int(res_data[0])
                        location = res_data[2]
                        out_bboxes[idx] = dict(
                            xmin=int(location[0, 0]),
                            ymin=int(location[0, 1]),
                            xmax=int(location[0, 0] + location[0, 2] - 1),
                            ymax=int(location[0, 1] + location[0, 3] - 1),
                        )
                        out_scores[idx] = res_data[3]

                gate_intersections = []
                if gates and n_boxes:
                    gate_intersections = self.updateGates(out_bboxes, out_ids, frame_id)

                num_frames = 1
                tracking_result = dict(
                    action="add_bboxes",
                    path=request_path,
                    frame_number=frame_id,
                    width=width,
                    height=height,
                    channel=channels,
                    bboxes=out_bboxes,
                    scores=out_scores,
                    labels=out_labels,
                    bbox_source="tracker",
                    id_numbers=out_ids,
                    last_frame_number=last_frame_number,
                    trigger_tracking_request=False,
                    num_frames=num_frames,
                    gates=gates,
                    gate_intersections=gate_intersections
                )
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(('localhost', request_port))
                send_msg_to_connection(tracking_result, sock)
                sock.close()
            except (KeyboardInterrupt, SystemExit):
                print('Exiting tracking server due to exception')
                return

    def updateGates(self, out_bboxes, out_ids, frame_id):
        gate_boxes = {}
        for box, _id in zip(out_bboxes, out_ids):
            box_xmin = box['xmin']
            box_xmax = box['xmax']
            box_ymin = box['ymin']
            box_ymax = box['ymax']
            if self.params.gate.intersection_method == 2:
                # center of the bottom edge of the box
                box_xcenter = float(box_xmin + box_xmax) / 2.0
                box_ycenter = float(box_ymax)
                if _id in self.target_centers:
                    gate_boxes[_id] = (self.target_centers[_id], (box_xcenter, box_ycenter))
                    self.target_centers[_id] = (box_xcenter, box_ycenter)
                else:
                    self.target_centers[_id] = (box_xcenter, box_ycenter)
            else:
                gate_boxes[_id] = ((box_xmin, box_ymin), (box_xmax, box_ymax))

        gate_intersections = []
        for _id, gate in self.gates.viewitems():
            gate.updateIntersections(gate_boxes, frame_id)
            gate_intersections.extend(
                [(_id, target_id) for target_id in gate.intersections])

        for id1, gate1 in self.gates.viewitems():
            for id2, gate2 in self.gates.viewitems():
                if id1 == id2:
                    continue
                common_targets = [k for k in gate1.intersections if k in gate2.intersections]
                for _id in common_targets:
                    if gate1.intersections[_id] < gate2.intersections[_id]:
                        first_id, second_id = id1, id2
                    else:
                        first_id, second_id = id2, id1

                    gate_pair = (first_id, second_id)
                    if gate_pair not in self.gate_pairs:
                        self.gate_pairs[gate_pair] = set()
                        self.gate_pairs[gate_pair].add(_id)
                    elif _id not in self.gate_pairs[gate_pair]:
                        self.gate_pairs[gate_pair].add(_id)
                    else:
                        continue
                    self.logger.info('Target {:d} went from gate {:d} to gate {:d}'.format(
                        _id, first_id, second_id))
        return gate_intersections

    def gatesToExcel(self):
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Statistics")
        sheet1.write(0, 0, "Gate")
        sheet1.write(0, 1, "Vehicles")

        sheet1.write(0, 3, "Entry Gate")
        sheet1.write(0, 4, "Exit Gate")
        sheet1.write(0, 5, "Vehicles")
        i = 1
        for _id in self.gates:
            sheet1.write(i, 0, str(_id))
            sheet1.write(i, 1, str(len(self.gates[_id].intersections)))
            i += 1
        i = 1
        for gate_pair in self.gate_pairs:
            sheet1.write(i, 3, str(gate_pair[0]))
            sheet1.write(i, 4, str(gate_pair[1]))
            sheet1.write(i, 5, str(len(self.gate_pairs[gate_pair])))
            i += 1
        if self.params.save_path:
            dir_name = os.path.dirname(self.params.save_path)
            fname = os.path.splitext(os.path.basename(self.params.save_path))[0]
            fname = "{:s}_gate_statistics_{:s}.xls".format(fname, get_date_time())
            file_path = linux_path(dir_name, fname)
        else:
            file_path = "gate_statistics_{:s}.xls".format(get_date_time())

        self.logger.info('Saving gate statistics to {:s}'.format(file_path))
        book.save(file_path)

    def close(self, gates):
        self.max_frame_id = -1
        self.frame_id = -1
        if self.params.test.load:
            if self.enable_visualization:
                self.visualizer.close()
        else:
            if self.tester is not None:
                self.tester.close()
                if self.params.test.save:
                    if not self.params.save_path:
                        save_path = 'server_res_{:s}.txt'.format(get_date_time())
                    else:
                        save_path = self.params.save_path
                    self.tester.save(save_path)
                self.tester = None
        if gates:
            self.gatesToExcel()

        self.gates = {}
        self.gate_pairs = {}
        self.target_centers = {}

    def train(self):
        input_path = self.trainer_params.input.path
        xmin, ymin, xmax, ymax = self.trainer_params.input.roi
        if input_path != self.current_path:
            self.frames_reader = get_frames_reader(input_path)
            self.current_path = input_path

        if xmax > xmin and ymax > ymin:
            roi = dict(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax
            )
            print('Setting ROI to: ', roi)
            self.frames_reader.setROI(roi)
        else:
            print('Invalid ROI provided: ', [xmin, ymin, xmax, ymax])

        start_frame_id, end_frame_id = self.trainer_params.input.frame_ids
        if start_frame_id < 0:
            start_frame_id = 0
        if end_frame_id <= start_frame_id:
            end_frame_id = self.frames_reader.num_frames - 1
        n_frames = end_frame_id - start_frame_id + 1
        frames = []
        print('Reading frames from ID {:d} to {:d}'.format(start_frame_id, end_frame_id))
        start_t = time.time()
        for frame_id in range(start_frame_id, end_frame_id + 1):
            curr_frame = self.frames_reader.get_frame(frame_id)
            if len(curr_frame.shape) == 3:
                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # print('curr_frame.shape: ', curr_frame.shape)
            frames.append(curr_frame)
            if (frame_id + 1) % 100 == 0:
                sys.stdout.write('Done {:d}/{:d} frames\n'.format(frame_id - start_frame_id + 1, n_frames))
                sys.stdout.flush()
        end_t = time.time()
        sys.stdout.write('Time taken: {:f} secs\n'.format(end_t - start_t))
        sys.stdout.flush()

        annotations = Annotations(self.trainer_params.input.annotations, self.logger)
        annotations.initialize(self.frames_reader.num_frames, start_frame_id, end_frame_id)
        if not annotations.read(build_index=True, build_trajectory_index=True):
            self.logger.error('Annotations could not be read')
            return

        detections = Detections(self.trainer_params.input.detections, self.logger)
        detections.initialize(self.frames_reader.num_frames, start_frame_id, end_frame_id)
        if not detections.read(build_index=True):
            self.logger.error('Detections could not be read')
            return

        seq_name = os.path.splitext(os.path.basename(input_path))[0]

        self.trainer = Trainer(self.trainer_params, self.logger)
        if self.params.continue_training:
            filename, file_extension = os.path.splitext(self.params.load_path)
            if not self.trainer.load(filename):
                self.logger.error('Trained target could not be loaded')
                return
        self.trainer.initialize(frames=frames, annotations=annotations, detections=detections)
        self.trainer.update()
        self.trainer.save(self.params.save_path)

    def run(self):
        if self.params.mode == 1:
            # testing mode
            threading.Thread(target=self.request_loop).start()
            threading.Thread(target=self.processing_loop).start()
        else:
            self.train()
