import os
import sys
import subprocess
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(__file__))

try:
    from Tkinter import Tk
except ImportError:
    from tkinter import Tk

from zipfile import ZipFile
from datetime import datetime

import logging
import shutil
import paramiko
import cv2

# import difflib

logging.getLogger("paramiko").setLevel(logging.WARNING)

import paramparse

from objects import Objects, TrackingResults
from input import Input
from data import Data
from visualizer import Visualizer, ObjTypes
from params import Params
from utilities import CustomLogger, SIIF, linux_path, stack_images, annotate_and_show, print_df, profile, \
    check_load_fnames, SCP


class VisParams:
    """
    :ivar res_type: the type of tracking results to use for  visualization:
        0: standard tracking results
        1: MOT compatible results
        2: raw tracking results
    """

    def __init__(self):
        self.in_file = ""
        # in_file = linux_path('log', 'multi_vis.txt')
        self.cmd_in_file = linux_path('log', 'multi_vis_cmd.txt')
        self.force_download = 0
        self.allow_missing_hota = 0
        self.grid_size = ()
        self.max_width = 3600
        self.max_height = 2100
        self.batch_input = 0
        self.annotations = 1
        self.detections = 1
        self.res_type = 0
        self.open_tee = 1
        self.disp_metrics = ['HOTA', 'MTR', 'IDSW', 'IDF1', 'DetA', 'AssA', 'Frag', 'MLR']
        self.disp_stats = ['total_num_frames', 'GT_IDs', 'IDs', 'GT_Dets', 'Dets']

        self.scp = SCP()


def run_unzip(tee_zip_path):
    unzip_cmd = 'unzip {}'.format(tee_zip_path)
    unzip_cmd_list = unzip_cmd.split(' ')
    print('Running {}'.format(unzip_cmd))
    subprocess.check_call(unzip_cmd_list)


def run_scp(params, server_name, dst_rel_path, is_file, only_txt=0):
    """

    :param SCP params:
    :param server_name:
    :param dst_rel_path:
    :param is_file:
    :return:
    """

    assert params.auth_data, "auth data not provided"
    assert server_name in params.auth_data, "server_name: {} not found in auth data".format(server_name)

    auth_data = params.auth_data[params.global_server]  # type: SCP.Auth

    remote_home_dir = params.home_dir
    if server_name not in (params.global_server, params.auth_data[params.global_server].alias):
        remote_home_dir = linux_path(remote_home_dir, "samba_{}".format(server_name))

    # if server_name == 'grs':
    #     src_root_path = linux_path(params.home_dir, params.code_path)
    # elif server_name == 'x99':
    #     src_root_path = linux_path(params.home_dir, "samba_x99", params.code_path)
    # elif server_name == 'orca':
    #     src_root_path = linux_path(params.home_dir, "samba_orca", params.code_path)
    # else:
    #     raise AssertionError('invalid server name: {}'.format(server_name))

    src_root_path = linux_path(remote_home_dir, params.code_path)

    if not is_file:
        os.makedirs(dst_rel_path, exist_ok=True)

    if not params.enable_zipping:
        scp_path = linux_path(src_root_path, dst_rel_path + '/*')

        scp_cmd = "pscp -pw {} -r -P 22 {}@{}:{} {}".format(auth_data.pwd, auth_data.user, auth_data.global_url,
                                                            scp_path,
                                                            dst_rel_path + '/')

        print('Running {}'.format(scp_cmd))
        scp_cmd_list = scp_cmd.split(' ')
        subprocess.check_call(scp_cmd_list)

        return

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    zip_fname = "multi_vis_{}_{}.zip".format(server_name, timestamp)
    zip_path = linux_path(params.home_dir, zip_fname)

    zip_cmd = "cd {} && zip -r {} {}".format(src_root_path, zip_path, dst_rel_path)

    if only_txt:
        # zip_multi_path = linux_path(params.home_dir, 'PTF', 'zipMulti.py')
        # z2txt = 'python3 {} relative=0 exclude_ext=pt,npz,npy,jpg,png,mp4,mkv,avi,zip ' \
        #         'dir_names={} ' \
        #         'out_name={} '.format(zip_multi_path, dst_rel_path, zip_fname)
        # zip_cmd = "cd {} && {}".format(src_root_path, z2txt)

        exclude_switches = ' '.join(f'-x "*.{ext}"' for ext in params.exclude_exts)
        zip_cmd = "{} {}".format(zip_cmd, exclude_switches)

    print('Running {}'.format(zip_cmd))

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(auth_data.global_url, username=auth_data.user, password=auth_data.pwd)

    stdin, stdout, stderr = client.exec_command(zip_cmd)

    stdout = list(stdout)
    for line in stdout:
        print(line.strip('\n'))

    stderr = list(stderr)

    if stderr:
        raise AssertionError('remote command did not work' + '\n' + '\n'.join(stderr))

    client.close()

    # ssh = subprocess.Popen(["ssh", scp_dst, zip_cmd],
    #                        shell=False,
    #                        stdout=subprocess.PIPE,
    #                        stderr=subprocess.PIPE)
    # result = ssh.stdout.readlines()
    # if not result:
    #     error = ssh.stderr.readlines()
    #     raise AssertionError("ERROR: %s" % error)
    # else:
    #     print(result)

    scp_cmd = "pscp -pw {} -r -P 22 {}@{}:{} ./".format(auth_data.pwd, auth_data.user, auth_data.global_url, zip_path)
    print('Running {}'.format(scp_cmd))
    scp_cmd_list = scp_cmd.split(' ')
    subprocess.check_call(scp_cmd_list)

    with ZipFile(zip_fname, 'r') as zipObj:
        zipObj.extractall()

    if params.remove_zip:
        os.remove(zip_fname)


def main():
    SIIF.setup()
    """setup logger"""
    _logger = CustomLogger.setup(__name__)

    params = VisParams()

    paramparse.process(params, allow_unknown=1)

    params.scp.read_auth()

    if params.in_file and os.path.exists(params.in_file):
        in_txt = open(params.in_file, 'r').read()
    else:
        try:
            in_txt = Tk().clipboard_get()
        except BaseException as e:
            print('Tk().clipboard_get() failed: {}'.format(e))
            return

    print('in_txt: {}'.format(in_txt))

    in_lines = in_txt.split('\n')
    in_lines = [in_line.strip() for in_line in in_lines if in_line.strip()]
    n_lines = len(in_lines)

    labels = {_line_id: '{}'.format(_line_id) for _line_id in range(n_lines)}

    cmd = None
    cmd_args = None

    if params.cmd_in_file and os.path.exists(params.cmd_in_file):
        cmd = open(params.cmd_in_file, 'r').read().strip()

    load_dirs = []
    log_ids = []
    disp_metrics_list = []
    disp_stats_list = []

    tee_dir = linux_path('log', 'tee')
    os.makedirs(tee_dir, exist_ok=True)

    rpip_dir = linux_path('log', 'rpip')
    os.makedirs(rpip_dir, exist_ok=True)

    load_dir_txt = ''

    disp_metrics = ['file', ] + params.disp_metrics
    disp_stats = ['file', ] + params.disp_stats

    tee_only = 0

    load_fnames = None

    # run_on_train = 0

    for _line_id, _line in enumerate(in_lines):
        tokens = _line.split('\t')

        if len(tokens) == 1:
            id_token = tokens[0]
            load_dir = ''
            tee_only = 1
        else:
            id_token, load_dir = tokens[:2]

        id_time, server_token = id_token.split(' :: ')

        if '-' in id_time:
            _label, _time = id_time.split('-')
            labels[_line_id] = _label

        if ':' in server_token:
            server_name, tmux_pane, tee_id = server_token.split(':')
            log_id = tee_id
            log_dir = tee_dir
        else:
            server_name = server_token.split('_')[-1]
            log_id = server_token
            log_dir = rpip_dir

        log_ids.append(log_id)
        if cmd is None or params.open_tee:
            log_ansi_path = linux_path(log_dir, log_id + '.ansi')

            if not os.path.exists(log_ansi_path):
                log_zip_path = linux_path(log_dir, log_id + '.zip')
                if not os.path.exists(log_zip_path):
                    print(
                        'neither log_ansi_path: {} nor log_zip_path: {} exist'.format(log_ansi_path, log_zip_path))

                    try:
                        run_scp(params.scp, server_name, log_zip_path, is_file=1)
                    except:
                        run_scp(params.scp, server_name, log_ansi_path, is_file=1)

                # run_unzip(log_zip_path)

                if os.path.exists(log_zip_path):
                    with ZipFile(log_zip_path, 'r') as zipObj:
                        zipObj.extractall(log_dir)

                    if params.scp.remove_zip:
                        os.remove(log_zip_path)

            if cmd is None:
                tee_data = open(log_ansi_path, 'r').readlines()

                while True:
                    if tee_data[0].startswith('main.py '):
                        break
                    del tee_data[0]

                cmd = tee_data[0].strip()

            if params.open_tee:
                tee_cmd = "start {}".format(log_ansi_path)
                os.system(tee_cmd)

        if cmd_args is None:
            cmd_args = cmd.split(' ')[1:-2]

        if tee_only:
            continue

        if params.force_download and os.path.exists(load_dir):
            shutil.rmtree(load_dir)

        hota_path = linux_path(load_dir, 'mot_metrics_hota.log')
        hota_mot_path = linux_path(load_dir, 'mot_metrics_hota_mot.log')

        if not os.path.exists(load_dir) or not os.listdir(load_dir) or \
                not (params.allow_missing_hota or os.path.exists(hota_path)):
            run_scp(params.scp, server_name, load_dir, is_file=0, only_txt=1)
        # print()

        if os.path.exists(hota_path):
            hota_data = pd.read_csv(hota_path, sep='\t')

            hota_data_cols = hota_data.columns.tolist()
            if os.path.exists(hota_mot_path):
                hota_mot_data = pd.read_csv(hota_mot_path, sep='\t')
                hota_mot_cols = hota_mot_data.columns.tolist()

                hota_mot_metric_cols = [col for col in hota_mot_cols if col not in hota_data_cols]

                if hota_mot_metric_cols:
                    hota_mot_data_clipped = hota_mot_data[hota_mot_metric_cols]
                    hota_combined_data = pd.concat((hota_data, hota_mot_data_clipped), axis=1)
                else:
                    hota_combined_data = hota_data
            else:
                hota_combined_data = hota_data

            hota_combined_cols = hota_combined_data.columns.tolist()
            hota_combined_cols_stripped = [col.strip() for col in hota_combined_cols]

            invalid_metrics = [col for col in params.disp_metrics if col not in hota_combined_cols_stripped]
            assert not invalid_metrics, 'invalid_metrics: {}'.format(invalid_metrics)

            invalid_stats = [col for col in params.disp_stats if col not in hota_combined_cols_stripped]
            assert not invalid_stats, 'invalid_stats: {}'.format(invalid_stats)

            rename_mapper = dict(zip(hota_combined_cols, hota_combined_cols_stripped))
            hota_combined_data.rename(mapper=rename_mapper, axis=1, inplace=True)

            hota_disp_metrics = hota_combined_data[disp_metrics]
            hota_disp_stats = hota_combined_data[disp_stats]

            disp_metrics_list.append(hota_disp_metrics)
            disp_stats_list.append(hota_disp_stats)
        else:
            msg = 'hota_path does not exist: {}'.format(hota_path)
            if params.allow_missing_hota:
                print(msg)
            else:
                raise AssertionError(msg)

        load_dir_txt += '{} : {}\n'.format(_line_id, load_dir)

        all_fnames = [k for k in os.listdir(load_dir)]

        if params.res_type == 0:
            _load_fnames = [k for k in all_fnames if k.endswith('.txt') and 'mot_compat' not in k]
        elif params.res_type == 1:
            _load_fnames = [k for k in all_fnames if k.endswith('mot_compat.txt')]
        elif params.res_type == 2:
            _load_fnames = [k for k in all_fnames if k.endswith('raw')]
        else:
            raise AssertionError('invalid tracking result type: {}'.format(params.res_type))

        _load_fnames.sort()

        # _load_fnames = set(_load_fnames)

        if load_fnames is None:
            load_fnames = _load_fnames
        else:
            assert load_fnames == _load_fnames, "mismatch in load filenames found"

        load_dirs.append(load_dir)

        # if 'dtest' in load_dir:
        #     run_on_train = 1

    if tee_only:
        return

    all_args = list(cmd_args) + list(sys.argv[1:])

    main_params = Params()
    paramparse.process(main_params, cmd_args=all_args, allow_unknown=1)

    main_params.process()

    visualizer_params = main_params.tester.visualizer  # type: Visualizer.Params
    # visualizer_params.disp_size = ()

    input_params = main_params.tester.input  # type: Input.Params
    input_params.batch_mode = params.batch_input

    _input = Input(input_params, _logger)
    _data = Data(main_params.data, _logger)

    # if run_on_train:
    #     print('running on training sequences')
    #     run_params = main_params.train
    # else:
    #     run_params = main_params.test

    run_params = main_params.test

    if not check_load_fnames(load_fnames, run_params, _data, _logger):
        run_params.synchronize(main_params.train, force=True)
        assert check_load_fnames(load_fnames, run_params, _data, _logger), \
            "neither training not testing parameters match with the available load filenames"
        print('running on training sequences')

    start_id = run_params.start
    n_seq = len(run_params.seq)
    run_seq_ids = run_params.seq[start_id:]

    pause = 1
    _fps = {}
    _times = {}

    # for _id, load_fname in enumerate(load_fnames):
    for _id, run_id in enumerate(run_seq_ids):

        # load_fname = load_fnames[_id]

        _logger.info('Running tester on sequence {:d} in set {:d} ({:d} / {:d} )'.format(
            run_id, run_params.seq_set, _id + run_params.start + 1, n_seq))

        if not _data.initialize(run_params.seq_set, run_id, 1, logger=_logger):
            raise AssertionError('Data module failed to initialize with sequence {:d}'.format(run_id))

        if not _input.initialize(_data, read_img=1, logger=_logger):
            raise IOError('Input pipeline could not be initialized')

        if run_params.subseq_postfix:
            _load_fname_template = '{:s}_{:d}_{:d}'.format(_data.seq_name, _data.start_frame_id + 1,
                                                           _data.end_frame_id + 1)
        else:
            _load_fname_template = '{:s}'.format(_data.seq_name)

        load_fname = [k for k in load_fnames if k.startswith(_load_fname_template)]
        assert len(load_fname) == 1, \
            "unique match for the load filename not found"
        load_fname = load_fname[0]

        tracking_results = []
        visualizers = []

        seq_hotta_metrics = []
        seq_hotta_stats = []
        for _line_id, load_dir in enumerate(load_dirs):
            load_path = linux_path(load_dir, load_fname).replace('\\', '/')

            if not _input.read_tracking_results(load_path):
                raise IOError('Tracking results could not be loaded')
            tracking_results.append(_input.track_res)
            _visualizer = Visualizer(visualizer_params, _logger)
            visualizers.append(_visualizer)

            hota_id = start_id + _id

            hotta_metrics = disp_metrics_list[_line_id].iloc[[hota_id, ]].rename(index={hota_id: _line_id})
            hotta_stats = disp_stats_list[_line_id].iloc[[hota_id, ]].rename(index={hota_id: _line_id})

            hota_file = hotta_metrics['file'].values.item().replace('\\', '/')
            hota_file_no_ext = os.path.splitext(hota_file)[0]
            load_path_no_ext = os.path.splitext(load_path)[0]

            assert hota_file_no_ext.startswith(load_path_no_ext), \
                "mismatch between hota_file:\n{}\n and load_path:\n{}\n".format(hota_file, load_path)

            seq_hotta_metrics.append(hotta_metrics[params.disp_metrics])
            seq_hotta_stats.append(hotta_stats[params.disp_stats])

        seq_hotta_metrics = pd.concat(seq_hotta_metrics, axis=0)
        seq_hotta_stats = pd.concat(seq_hotta_stats, axis=0)

        seq_hotta_metrics_txt = print_df(seq_hotta_metrics, name='seq_hotta_metrics', fmt='.3f', return_mode=1)
        seq_hotta_stats_txt = print_df(seq_hotta_stats, name='seq_hotta_stats', fmt='.3f', return_mode=1)

        _logger.warning('\n' + load_dir_txt)
        _logger.warning('\n' + seq_hotta_metrics_txt)
        _logger.warning('\n' + seq_hotta_stats_txt)

        ann_visualizer = det_visualizer = None

        if params.annotations:
            if not _input.read_annotations():
                raise IOError('annotations could not be loaded')
            ann_visualizer = Visualizer(visualizer_params, _logger)

        if params.detections:
            if not _input.read_detections():
                raise IOError('detections could not be loaded')
            det_visualizer = Visualizer(visualizer_params, _logger)

        for frame_id in range(_input.n_frames):
            with profile('_input', _times=_times, _fps=_fps, show=0):
                if _input.params.batch_mode:
                    frame = _input.all_frames[frame_id]
                else:
                    # first frame was read during pipeline initialization
                    if frame_id > 0 and not _input.update():
                        raise AssertionError('Input image {:d} could not be read'.format(frame_id))
                    frame = _input.curr_frame

            with profile('vis', _times=_times, _fps=_fps, show=0):
                tracking_res_images = []
                for _line_id, tracking_res in enumerate(tracking_results):
                    visualizer = visualizers[_line_id]
                    frame_data = {obj_type: None for obj_type in visualizer.obj_types}

                    _obj = tracking_res  # type: TrackingResults
                    # all objects in the current frame
                    ids = _obj.idx[frame_id]
                    if ids is None:
                        # no objects in this frame
                        tracking_res_img = np.copy(frame)
                    else:
                        _frame_data = _obj.data[ids, :]
                        if _obj.states is not None:
                            _states = _obj.states[ids].reshape((_frame_data.shape[0], 1))
                            _frame_data = np.concatenate((_frame_data, _states), axis=1)
                        if _obj.is_valid is not None:
                            _is_valid = _obj.is_valid[ids].reshape((_frame_data.shape[0], 1))
                            _frame_data = np.concatenate((_frame_data, _is_valid), axis=1)
                        frame_data[ObjTypes.tracking_result] = _frame_data

                        vis_imgs = visualizer.update(frame_id, frame, frame_data, return_mode=1,
                                                     label=labels[_line_id])

                        tracking_res_img = vis_imgs[ObjTypes.tracking_result]
                    tracking_res_images.append(tracking_res_img)

                if params.annotations:
                    frame_data = {obj_type: None for obj_type in ann_visualizer.obj_types}
                    ids = _input.annotations.idx[frame_id]
                    if ids is not None:
                        _frame_data = _input.annotations.data[ids, :]
                        frame_data[ObjTypes.annotation] = _frame_data
                        vis_imgs = ann_visualizer.update(frame_id, frame, frame_data, return_mode=1,
                                                         label='annotation')
                        tracking_res_images.append(vis_imgs[ObjTypes.annotation])

                if params.detections:
                    frame_data = {obj_type: None for obj_type in det_visualizer.obj_types}
                    ids = _input.detections.idx[frame_id]
                    if ids is not None:
                        _frame_data = _input.detections.data[ids, :]
                        frame_data[ObjTypes.detection] = _frame_data
                        vis_imgs = det_visualizer.update(frame_id, frame, frame_data, return_mode=1,
                                                         label='detection')
                        tracking_res_images.append(vis_imgs[ObjTypes.detection])

            # with profile('stack', _times=_times, _fps=_fps, show=0):
            # tracking_res_image = stack_images(tracking_res_images)

            total_time = sum(list(_times.values()))
            if total_time > 0:
                overall_fps = 1.0 / total_time
            else:
                overall_fps = 0

            title = 'tracking_res_imgage'
            _fps_txt = 'fps: {:.3f}'.format(overall_fps) + ' '.join('{}: {:.3f}'.format(k, v) for k, v in _fps.items())

            with profile('annotate', _times=_times, _fps=_fps, show=0):
                img_stacked_txt = annotate_and_show(title, tracking_res_images, text=_fps_txt, n_modules=0,
                                                    grid_size=params.grid_size,
                                                    max_width=params.max_width,
                                                    max_height=params.max_height, only_annotate=1)

            _siif = SIIF.imshow(title, img_stacked_txt)
            k = cv2.waitKey(1 - pause)
            if k == 27:
                cv2.destroyWindow(title)
                exit()
            elif k == 32:
                pause = 1 - pause
            elif k == ord('q'):
                break


if __name__ == '__main__':
    main()
