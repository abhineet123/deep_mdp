#!/usr/bin/env python

import subprocess
import sys
import time
from ast import literal_eval

from . import utilities as utils

start_id = 0
end_id = 0
use_hardcoded_options = 0
verbose = 0
log_fname = 'pymdp'

arg_id = 1
if len(sys.argv) > arg_id:
    start_id = literal_eval(sys.argv[arg_id])
    arg_id += 1

if len(sys.argv) > arg_id:
    end_id = literal_eval(sys.argv[arg_id])
    arg_id += 1

# if len(sys.argv) > arg_id:
#     use_hardcoded_options = int(sys.argv[arg_id])
#     arg_id += 1
#
# if len(sys.argv) > arg_id:
#     verbose = int(sys.argv[arg_id])
#     arg_id += 1

if len(sys.argv) > arg_id:
    log_fname = sys.argv[arg_id]
    arg_id += 1

if use_hardcoded_options:
    options = [
        'seq_set_id=7',
        'data.ratios=1,1',
        'data.offsets=0,0',
        'trainer.load=0',
        'trainer.debug.write_state_info=0',
        'trainer.debug.write_thresh=1,1',
        'trainer.debug.write_to_bin=1',
        'trainer.verbose={:d}'.format(verbose),
        'trainer.input.read_from_bin=1',
        'trainer.input.batch_mode=1',
        'trainer.input.annotations.sort_by_frame_ids=0',
        'trainer.input.detections.sort_by_frame_ids=0',
        'trainer.visualizer.mode=0,0,0',
        'trainer.visualizer.pause_after_frame=0',
        'tester.load=1',
        'tester.evaluate=1',
        'tester.debug.write_state_info=0',
        'tester.debug.write_thresh=1,1',
        'tester.verbose={:d}'.format(verbose),
        'tester.hungarian=0',
        'tester.save_prefix=#',
        'tester.visualizer.mode=0,0,0',
        'tester.visualizer.pause_after_frame=0',
        'tester.input.read_from_bin=1',
        'tester.input.batch_mode=0'
    ]
else:
    options = []

if len(sys.argv) > arg_id:
    options.extend(list(sys.argv[arg_id:]))

command = 'python main.py'
for option in options:
    command = '{:s} --{:s}'.format(command, option)

if log_fname == 'n':
    log_fname = ''
else:
    curr_date_time = time.strftime("%y%m%d_%H%M", time.localtime())
    log_fname = '{:s}_{:s}.log'.format(log_fname, curr_date_time)
    print(('Writing log to {:s}'.format(log_fname)))

if isinstance(start_id, tuple):
    # piecewise training mode
    train_seq_ids = utils.parse_seq_IDs(start_id)
    print(('Running piecewise training on sequences: ', train_seq_ids))
    n_ids = len(train_seq_ids)
    for i in range(n_ids):
        if i == 0:
            if end_id >= 0:
                continue
            full_command = '{:s} --train_seq_ids={:d}'.format(command, train_seq_ids[i])
        else:
            full_command = '{:s} --train_seq_ids={:d},{:d} --continue_training=1'.format(
                command, train_seq_ids[i - 1], train_seq_ids[i])
        if log_fname:
            full_command = '{:s} 2>&1 | tee -a {:s}'.format(full_command, log_fname)
        print('\nRunning: {:s}\n'.format(full_command))
        subprocess.call(full_command, shell=True)
elif isinstance(end_id, tuple):
    # piecewise testing mode
    test_seq_ids = utils.parse_seq_IDs(end_id)
    print(('Running piecewise testing on sequences: ', test_seq_ids))
    n_ids = len(test_seq_ids)
    for i in range(n_ids):
        full_command = '{:s} --test_seq_ids={:d}'.format(command, test_seq_ids[i])
        if log_fname:
            full_command = '{:s} 2>&1 | tee -a {:s}'.format(full_command, log_fname)
        print('\nRunning: {:s}\n'.format(full_command))
        subprocess.call(full_command, shell=True)
else:
    # piecewise training-testing mode
    if end_id < start_id:
        end_id = start_id

    if start_id >= 0:
        for seq_id in range(start_id, end_id + 1):
            full_command = '{:s} --train_seq_ids={:d} --test_seq_ids={:d}'.format(
                command, seq_id, seq_id)
            if log_fname:
                full_command = '{:s} 2>&1 | tee -a {:s}'.format(full_command, log_fname)
            print('\nRunning: {:s}\n'.format(full_command))
            subprocess.call(full_command, shell=True)
    else:
        full_command = '{:s}'.format(command)
        if log_fname:
            full_command = '{:s} 2>&1 | tee -a {:s}'.format(full_command, log_fname)
        print('\nRunning: {:s}\n'.format(full_command))
        subprocess.call(full_command, shell=True)
