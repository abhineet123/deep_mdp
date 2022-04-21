import os
import copy
import json

import torch
import gc

import paramparse
from paramparse import MultiCFG, MultiPath

from data import Data
from trainer import Trainer
from tester import Tester
from target import Target
from policy_base import PolicyBase
from active import Active
from lost import Lost
from tracked import Tracked
from models.model_base import ModelBase
from models.svm import SVM
from models.xgb import XGB
from models.mlp import MLP
from models.cnn import CNN
from models.dummy import Oracle, Positive, Negative, Random
from run import Train, Test

from utilities import CustomLogger, linux_path, SaveModes
# from Server import ServerParams

from utilities import parse_seq_IDs, x11_available, disable_vis, set_recursive, BaseParams


class MainParams(BaseParams):
    """
    root parameters - has to be defined here instead of main or separate module to prevent circular dependency between
     IBT and that module since Params needs IBT.Params and IBT needs Params for intellisense

    :type cfg: str
    :type data: Data.Params
    :type train: Train.Params
    :type test: Test.Params
    :type tester: Tester.Params
    :type trainer: Trainer.Params
    :type ibt: IBT.Params


    :ivar mode: 0: standard training and testing; 1: IBT

    :ivar vis: set to 0 to disable visualization globally

    :ivar train: training settings

    :ivar test: testing settings

    :ivar cfg: optional ASCII text file from where parameter values can be
    read;command line parameter values will override the values in this file

    :ivar log_dir: directory where log files are created; leaving it empty
    disables logging to file

    :ivar data: parameters for Data module

    :ivar tester: Tester parameters

    :ivar trainer: Trainer parameters

    :ivar ibt: Iterative Batch Training parameters

    """

    def __init__(self):

        self.gpu = ""
        self.vis = 1
        self.mode = 0

        self.data = Data.Params()
        self.train = Train.Params()
        self.test = Test.Params()

        self.ibt = IBT.Params()

        self.cfg_root = 'cfg'
        self.cfg_ext = 'cfg'
        self.cfg = ()

        self.log_dir = ''

        self.tester = Tester.Params()
        self.trainer = Trainer.Params()
        # self.server = ServerParams()

    def process(self, args_in=None):
        # self.train.process()
        # self.test.process()
        self.train.seq = parse_seq_IDs(self.train.seq, self.train.sample)
        self.test.seq = parse_seq_IDs(self.test.seq, self.test.sample)

        self.test.synchronize(self.train, force=False)

        if self.tee_log:
            set_recursive(self, 'tee_log', self.tee_log, check_existence=1)

        if self.vis != 2:
            """disable vis if GUI not available"""
            from sys import platform
            if not self.vis or (platform in ("linux", "linux2") and not x11_available()):
                disable_vis(self, args_in)

                import matplotlib as mpl

                mpl.use('Agg')

            # print('train_seq_ids: ', self.train_seq_ids)
            # print('test_seq_ids: ', self.test_seq_ids)


class IBT:
    class Params(BaseParams):
        """
        Iterative Batch Train Parameters

        :type cfgs: MultiCFG
        :type test_cfgs: MultiCFG
        :type async_dir: MultiPath
        :type states: MultiPath


        :ivar async_dir: 'Directory for saving the asynchronous training data',
        :ivar test_cfgs: 'cfg files and sections from which to read iteration specific configuration data '
                     'for testing and evaluation phases; '
                     'cfg files for different iterations must be separated by double colons followed by the '
                     'iteration id and a single colon; cfg files for any iteration can be provided in multiple '
                     'non contiguous units in which case they would be concatenated; '
                     'commas separate different cfg files for the same iteration and '
                     'single colons separate different sections for the same cfg file as usual; '
                     'configuration in the last provided iteration would be used for all subsequent '
                     'iterations as well unless an underscore (_) is used to revert to the global '
                     '(non iteration-specific) parameters; ',
        :ivar cfgs: 'same as test_cfgs except for the data generation and training phases '
                'which are specific to each '
                'state so that the iteration ID here includes both the iteration itself as well as the state; '
                'e.g. with 2 states:  iter 0, state 1  -> id = 01; iter 2, state 0 -> id = 20',
        :ivar start_iter: 'Iteration at which the start the training process',
        :ivar load: '0: Train from scratch '
                '1: load previously saved weights from the last iteration and continue training;'
                'Only applies if iter_id>0',
        :ivar states: 'states to train: one or more of [active, tracked, lost]',
        :ivar load_weights: '0: Train from scratch; '
                        '1: load previously saved weights and test; '
                        '2: load previously saved weights and continue training; ',
        :ivar min_samples: 'minimum number of samples generated in data_from_tester '
                       'for the policy to be considered trainable',
        :ivar accumulative: 'decides if training data from all previous iterations is added to that from '
                        'the current iteration for training',
        :ivar start_phase: 'Phase at which the start the training process in the iteration specified by start_id:'
                       '0: data generation / evaluation of previous iter'
                       '1: batch training '
                       '2: testing / evaluation of policy classifier '
                       '3: testing / evaluation of tracker ',
        :ivar ips: 'triplet of integers specifying start iter,phase,state (optionally separated by commas)',
        :ivar start: 'single string specifying both start_id and start_phase by simple concatenation;'
                 'e.g  start=12 means start_id=1 and start_phase=2; '
                 'overrides both if provided',

        :ivar load_prev: continue training in the start iteration by loading weights from the same iteration
        saved in a previous run instead of loading them from previous iteration (if start iter > 0)

        """

        def __init__(self):
            self.ips = ''
            self.start = ''
            self.start_iter = 0
            self.start_phase = 0
            self.start_state = 0
            self.start_seq = -1
            self.data_from_tester = 0
            self.load = 0
            self.states = []
            self.skip_states = []
            self.n_iters = 5
            self.min_samples = 100
            self.allow_too_few_samples = 0
            self.accumulative = 0
            self.load_weights = 2
            self.save_suffix = ''
            self.load_prev = 0
            self.phases = ()
            self.test_iters = ()
            self.async_dir = MultiPath()
            self.cfgs = MultiCFG()
            self.test_cfgs = MultiCFG()

        def process(self):
            # self.async_dir = '_'.join(self.async_dir)
            if self.ips:
                self.start = self.ips

            if self.start:
                if ',' in self.start:
                    start = list(map(int, self.start.split(',')))
                else:
                    start = list(map(int, [*self.start]))

                if len(start) > 4:
                    self.start_iter, self.start_phase, self.start_state = start[:3]
                    self.start_seq = int(''.join(map(str, start[3:])))
                elif len(start) == 4:
                    self.start_iter, self.start_phase, self.start_state, self.start_seq = start
                elif len(start) == 3:
                    self.start_iter, self.start_phase, self.start_state = start
                elif len(start) == 2:
                    self.start_iter, self.start_phase = start
                else:
                    raise AssertionError(f'Invalid start IDs: {self.start}')

        def get_cfgs(self):
            n_states = len(self.states)

            valid_cfgs = [f'{iter_id}{state_id}' for iter_id in range(self.n_iters) for state_id in range(n_states)]
            return MultiCFG.to_dict(self.cfgs, valid_cfgs)

        def get_test_cfgs(self):
            valid_test_cfgs = list(map(str, range(self.n_iters)))
            return MultiCFG.to_dict(self.test_cfgs, valid_test_cfgs)

    class Phases:
        data_generation, training, evaluation, testing = range(4)

    @staticmethod
    def _add_header(logger, phase, iter_id, states=None):
        """

        :param CustomLogger logger:
        :param str phase:
        :param int iter_id:
        :param list[str] | tuple[str] | str states:
        :return:
        """

        if states is not None:
            if isinstance(states, str):
                states_str = states
            else:
                states_str = '_'.join(states)

            header = '{}:{}:{}'.format(iter_id, phase, states_str)
        else:
            header = '{}:{}'.format(iter_id, phase)

        logger = CustomLogger(logger, names=(header,), key='custom_header')

        return logger

    @staticmethod
    def get_state_params(target_params, state):
        """

        :param Target.Params target_params:
        :param str state:
        :rtype: Policy.Params, Model.Params, int
        :return:
        """
        if state == 'active':
            state_params = target_params.active  # type: Active.Params
            async_mode = 3
        elif state == 'tracked':
            state_params = target_params.tracked  # type: Tracked.Params
            async_mode = 2
        elif state == 'lost':
            state_params = target_params.lost  # type: Lost.Params
            async_mode = 1
        else:
            raise AssertionError('Invalid state: {}'.format(state))

        if state_params.model == 'svm':
            model_params = state_params.svm  # type: SVM.Params
        elif state_params.model == 'xgb':
            model_params = state_params.xgb  # type: XGB.Params
        elif state_params.model == 'mlp':
            model_params = state_params.mlp  # type: MLP.Params
        elif state_params.model == 'cnn':
            model_params = state_params.cnn  # type: CNN.Params
        elif state_params.model == 'oracle':
            model_params = state_params.oracle  # type: Oracle.Params
        elif state_params.model == 'pos':
            model_params = state_params.pos  # type: Positive.Params
        elif state_params.model == 'neg':
            model_params = state_params.neg  # type: Negative.Params
        elif state_params.model == 'random':
            model_params = state_params.random  # type: Random.Params
        elif state_params.model in ('abs', 'none'):
            model_params = None
        else:
            raise RuntimeError('Invalid model type provided: {}'.format(
                state_params.model))

        return state_params, model_params, async_mode

    @staticmethod
    def save_status(ibt_status_file, update_dict):
        ibt_status_dict = {}

        if os.path.exists(ibt_status_file):
            with open(ibt_status_file, "r") as ibt_status_fid:
                ibt_status_dict = json.load(ibt_status_fid)

        ibt_status_dict.update(update_dict)

        with open(ibt_status_file, "w") as ibt_status_fid:
            json.dump(ibt_status_dict, ibt_status_fid)

    @staticmethod
    def _load_status(ibt_status_file, key):
        with open(ibt_status_file, "r") as ibt_status_fid:
            ibt_status_dict = json.load(ibt_status_fid)

        return ibt_status_dict[key]

    @staticmethod
    def pretrain_active(train_params, trainer_params, data_params, log_dir, logger, args_in):
        """

        pretrain active policy before training any other stuff
        annoying hack necessitated by the patchy nature of the whole training and testing setup

        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type data_params Data.Params
        :type log_dir str
        :type logger CustomLogger
        :type args_in list[str]

        :return:
        """

        curr_state = 'active'

        """data generation phase
        """

        active_params = trainer_params.target.active  # type: Active.Params

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params

        target_params = _trainer_params.target  # type: Target.Params

        state_params, model_params, async_mode = IBT.get_state_params(
            target_params, curr_state)  # type: PolicyBase.Params, ModelBase.Params, int
        batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

        active_pt_dir = _train_params.active_pt_dir
        if not active_pt_dir:
            active_pt_dir = state_params.model

        pretrain_dir = linux_path('log', 'pretrain_active', active_pt_dir)
        pretrain_mode = _train_params.active_pt

        pretrain_info_path = linux_path(pretrain_dir, 'pretrain_info.txt')

        logger.info('pretrain_info_path: {}'.format(pretrain_info_path))

        if os.path.isfile(pretrain_info_path):
            pretrain_load_path = open(pretrain_info_path, 'r').readlines()[0]
            if pretrain_mode == 2:
                logger.info('overwriting existing active pre-trained model: {}'.format(pretrain_load_path))
            else:
                logger.info('skipping active pre-training for existing model: {}'.format(pretrain_load_path))

                active_params.pretrain_path = pretrain_load_path
                return True
        elif pretrain_mode == 3:
            raise IOError('pre-trained active model info not found: {}'.format(pretrain_info_path))

        _train_params.results_dir = pretrain_dir
        # batch_params.save_path = pretrain_dir

        """save generated data"""
        _train_params.save = 1
        batch_params.save_samples = 1

        """single pass through training trajectories"""
        _trainer_params.max_count = 1
        _trainer_params.max_pass = 1

        """Asynchronous data generation on all samples whether or not they cause model failure"""
        _train_params.load = Train.Modes.train_from_scratch
        state_params.always_train = 1

        _trainer_params.mode = async_mode

        """load all states being trained and save only the current one"""
        # target_params.load_states = states
        target_params.save_states = [curr_state, ]

        _train_params.results_dir_root = ''

        _logger = CustomLogger(logger, names=('active:pretrain:data',), key='custom_header')

        data = Data(data_params, _logger)
        trained_target = Train.run(data, _trainer_params, _train_params, _logger, log_dir, args_in)

        if trained_target is None:
            raise AssertionError('active pretraining data generation phase failed')
        """training phase
        """

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params

        target_params = _trainer_params.target  # type: Target.Params

        _, model_params, async_mode = IBT.get_state_params(target_params, curr_state)
        batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

        _trainer_params.mode = -async_mode

        batch_params.db_path = linux_path(trained_target.save_dir, curr_state)
        batch_params.save_samples = 0

        _train_params.load = Train.Modes.train_from_scratch
        _train_params.results_dir = ''
        _train_params.results_dir_root = ''

        batch_params.save_dir = pretrain_dir

        batch_params.load_weights = 0

        _logger = CustomLogger(logger, names=('active:pretrain:train',), key='custom_header')

        data = Data(data_params, _logger)

        trained_target = Train.run(data, _trainer_params, _train_params, _logger, log_dir, args_in)

        if trained_target is None:
            raise AssertionError('active pretraining training phase failed')

        """enable preloading by setting path in original params"""
        active_params.pretrain_path = trained_target.active.batch_save_path

        open(pretrain_info_path, 'w').write(trained_target.active.batch_save_path)

        return True

    @staticmethod
    def test(iter_id, states, load_dir,
             accumulative,
             train_params, trainer_params,
             test_params, tester_params,
             data_params,
             results_dir_root, log_dir, logger, args_in):
        """

        :param int iter_id:
        :param list[str] states:
        :param str load_dir:
        :type accumulative int
        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type test_params Test.Params
        :type tester_params Tester.Params
        :type data_params Data.Params
        :param str results_dir_root:
        :param str log_dir:
        :param CustomLogger logger:
        :param list args_in:
        :return:

        """

        # assert iter_id > 0, "cannot test in iter 0"

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params
        _test_params = paramparse.copy_recursive(test_params)  # type: Test.Params
        _tester_params = paramparse.copy_recursive(tester_params)  # type: Tester.Params

        target_params = _trainer_params.target  # type: Target.Params

        _train_params.load = Train.Modes.test_only
        _train_params.load_dir = load_dir

        for state in states:
            _, model_params, _ = IBT.get_state_params(target_params, state)
            if model_params is None:
                logger.warning(f'none model specified for state: {state} which differs from training')
                continue

            batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

            _load_dir = '{}_batch_{}'.format(state, iter_id - 1)
            if not accumulative and iter_id > 1:
                _load_dir = '{}_acc0'.format(_load_dir)
            load_path = linux_path(results_dir_root, _load_dir)
            batch_params._load_dir = load_path

        save_dir = '{}_batch_{}'.format('_'.join(states), iter_id - 1)
        if not accumulative and iter_id > 1:
            save_dir = '{}_acc0'.format(save_dir)

        _test_params.results_dir = linux_path(results_dir_root, save_dir, 'test')

        # _test_params.eval_dir = _test_params.results_dir

        print(f'\nrunning iteration {iter_id - 1} testing phase\n')
        _logger = IBT._add_header(logger, 'test', iter_id - 1, states)
        test_data = Data(data_params, _logger)

        _trained_target = None
        if iter_id == 0:
            _train_params.load = Train.Modes.no_train

        if not _test_params.load:
            _trained_target = Train.run(test_data, _trainer_params, _train_params, _logger, log_dir, args_in)
            if not _trained_target:
                return False

        success = Test.run(_trained_target, test_data, _tester_params, _test_params,
                           _logger, log_dir, args_in)
        return success

    @staticmethod
    def train(iter_id, db_path, results_dir_root,
              state, load_weights, load_prev,
              accumulative,
              save_suffix,
              train_params, trainer_params,
              data_params,
              log_dir, logger, args_in):
        """

        :param int iter_id:
        :param str results_dir_root:
        :param str state:
        :param int load_weights:
        :param int load_prev:
        :type accumulative int
        :type save_suffix str
        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type data_params Data.Params
        :param str log_dir:
        :param CustomLogger logger:
        :param list args_in:
        :return:
        """

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params

        target_params = _trainer_params.target  # type: Target.Params

        _, model_params, async_mode = IBT.get_state_params(target_params, state)
        batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

        _trainer_params.mode = -async_mode

        batch_params.db_path = db_path
        batch_params.save_samples = 0

        _train_params.load = Train.Modes.train_from_scratch
        _train_params.results_dir = ''

        save_dir = '{}_batch_{}'.format(state, iter_id)
        load_path = ''

        if iter_id > 0:
            # if data_from_tester:
            #     load_dir = 'test_batch_{}'.format(iter_id - 1)
            # else:

            if load_prev:
                """load from same iter of a previous run and continue training"""

                """restore optimizer state too"""
                batch_params.load_opt = 1
                batch_params.load_stats = 1
                batch_params.load_weights = 2
            else:
                """load from previous iter of same or previous run and train from scratch"""
                load_iter = iter_id - 1
                batch_params.load_opt = 0
                """training on a different dataset with new samples so stats from previous iter are not relevant
                """
                batch_params.load_stats = 0

                load_dir = '{}_batch_{}'.format(state, load_iter)
                if not accumulative and load_iter > 0:
                    load_dir = '{}_acc0'.format(load_dir)

                load_path = linux_path(results_dir_root, load_dir)

                batch_params.load_weights = load_weights

            if not load_weights:
                save_dir = '{}_load0'.format(save_dir)

            if not accumulative:
                save_dir = '{}_acc0'.format(save_dir)

        else:
            batch_params.load_weights = load_prev
            batch_params.load_stats = 1

            if load_prev:
                batch_params.load_opt = 1

        if save_suffix:
            save_dir = '{}_{}'.format(save_dir, save_suffix)

        save_path = linux_path(results_dir_root, save_dir)

        batch_params.save_dir = save_path
        batch_params.load_dir = load_path

        _logger = IBT._add_header(logger, 'train', iter_id, state)
        data = Data(data_params, _logger)
        trained_target = Train.run(data, _trainer_params, _train_params, _logger, log_dir, args_in)
        if not trained_target:
            return False

        """hack to free GPU memory"""
        del trained_target
        gc.collect()
        torch.cuda.empty_cache()

        return True

    @staticmethod
    def data_from_trainer(
            _status,
            iter_id, states, state_id, load_dir, results_dir, results_dir_root,
            load_weights,
            accumulative,
            train_params, trainer_params,
            data_params,
            log_dir, logger, args_in):
        """

        :param dict _status:
        :param int iter_id:
        :param list[str] states:
        :param int state_id:
        :param str load_dir:
        :param str results_dir:
        :param str results_dir_root:
        :param int load_weights:
        :type accumulative int
        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type data_params Data.Params
        :param str log_dir:
        :param CustomLogger logger:
        :param list args_in:

        :rtype dict
        """
        assert state_id >= 0, "invalid state_id for data generation mode"

        curr_state = states[state_id]

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params
        target_params = _trainer_params.target  # type: Target.Params

        state_params, model_params, async_mode = IBT.get_state_params(
            target_params, curr_state)  # type: PolicyBase.Params, ModelBase.Params, int
        batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

        # _train_params.results_dir = results_dir
        _train_params.results_dir = results_dir

        """save generated data"""
        _train_params.save = 1
        batch_params.save_samples = 1

        """single pass through training trajectories"""
        _trainer_params.max_count = 1
        _trainer_params.max_pass = 1

        _trainer_params.mode = async_mode

        if iter_id == 0:
            """Asynchronous data generation on all samples whether or not they cause model failure"""

            state_params.always_train = 1
            if state_id == 0:
                """first state in first iter --> train_from_scratch
                """
                _train_params.load = Train.Modes.train_from_scratch
            else:
                """load all previously trained states in first iter and save only the current one
                """
                _train_params.load = Train.Modes.continue_training

                target_params.load_states = states[:state_id]
                target_params.save_states = [states[state_id], ]

                _train_params.load_dir = load_dir

                """don't train active as it has already been trained while generating data for the 
                first state"""
                # _trainer_params.train_active = 0

        else:
            """Data generation with trained model"""

            _train_params.load = Train.Modes.continue_training
            _train_params.load_dir = load_dir

            """save only hard samples where model fails
            """
            state_params.always_train = 0

            """load latest trained model for current state from the previous iter"""
            _load_dir = '{}_batch_{}'.format(curr_state, iter_id - 1)
            if not accumulative and iter_id > 1:
                _load_dir = '{}_acc0'.format(_load_dir)
            batch_params._load_dir = linux_path(results_dir_root, _load_dir)

            """load latest trained model for all subsequent states from the previous iter"""
            for _state in states[state_id + 1:]:
                """load latest trained model for all subsequent states from the previous iter"""
                _, _model_params, _ = IBT.get_state_params(target_params, _state)  # type: ModelBase.Params
                _batch_params = _model_params.batch  # type: ModelBase.Params.BatchTrain

                _load_dir = '{}_batch_{}'.format(_state, iter_id - 1)
                if not accumulative and iter_id > 1:
                    _load_dir = '{}_acc0'.format(_load_dir)
                _batch_params._load_dir = linux_path(results_dir_root, _load_dir)
                _batch_params.load_weights = load_weights

        for _state in states[:state_id]:
            """load latest trained model for all previous states from the same iter"""
            _, _model_params, _ = IBT.get_state_params(target_params, _state)  # type: ModelBase.Params
            _batch_params = _model_params.batch  # type: ModelBase.Params.BatchTrain

            _load_dir = '{}_batch_{}'.format(_state, iter_id)
            if not accumulative and iter_id > 0:
                _load_dir = '{}_acc0'.format(_load_dir)
            _batch_params._load_dir = linux_path(results_dir_root, _load_dir)
            _batch_params.load_weights = load_weights

        """load all states being trained and save only the current one"""
        # target_params.load_states = states
        target_params.save_states = [states[state_id], ]

        _logger = IBT._add_header(logger, 'data', iter_id, curr_state)
        data = Data(data_params, _logger)
        trained_target = Train.run(data, _trainer_params, _train_params, _logger, log_dir, args_in)

        _status = {
            curr_state: 1,
            'save_dir': trained_target.save_dir
        }

        return _status

    @staticmethod
    def data_from_tester(status,
                         iter_id, states,
                         load_dir, results_dir, results_dir_root,
                         min_samples,
                         allow_too_few_samples,
                         accumulative,
                         train_params, trainer_params,
                         test_params, tester_params,
                         data_params,
                         log_dir, logger, args_in):
        """

        :param dict | None status:
        :param int iter_id:
        :param list[str] states:
        :param str load_dir:
        :param str results_dir:
        :param str results_dir_root:
        :type min_samples int
        :type accumulative int
        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type test_params Test.Params
        :type tester_params Tester.Params
        :type data_params Data.Params
        :param str log_dir:
        :param CustomLogger logger:
        :param list args_in:
        :param int eval_mode:

        :rtype dict
        """

        if status is None:
            status = {_state: 0 for _state in states}

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params
        _test_params = paramparse.copy_recursive(test_params)  # type: Test.Params
        _tester_params = paramparse.copy_recursive(tester_params)  # type: Tester.Params
        _data_params = paramparse.copy_recursive(data_params)  # type: Data.Params

        target_params = _trainer_params.target  # type: Target.Params

        rgb_input = not _tester_params.input.convert_to_gs

        _test_params.results_dir = results_dir
        # _test_params.evaluate = 0

        """run tester on training set"""
        _test_params.synchronize(_train_params, force=True)

        _data_params.synchronize()

        _logger = IBT._add_header(logger, 'data_test', iter_id)

        """repetitive code for intellisense"""

        if 'active' in states:
            if not status['active']:
                target_params.active.model = "oracle"
                target_params.active.oracle.type = Oracle.Types.relative
            #     target_params.active.save_mode = SaveModes.all
            # else:
            #     target_params.active.save_mode = SaveModes.error

            target_params.active.syn_samples = 2
            target_params.active.save_mode = SaveModes.all
            # target_params.active.save_mode = SaveModes.none

        if 'lost' in states:
            if not status['lost']:
                target_params.lost.model = "oracle"
                target_params.lost.oracle.type = Oracle.Types.relative
            #     target_params.lost.save_mode = SaveModes.all
            # else:
            #     target_params.lost.save_mode = SaveModes.error

            target_params.lost.syn_samples = 2
            target_params.lost.save_mode = SaveModes.all
            # target_params.lost.save_mode = SaveModes.none

        if 'tracked' in states:
            if not status['tracked']:
                target_params.tracked.model = "oracle"
                target_params.tracked.oracle.type = Oracle.Types.relative
            #     target_params.tracked.save_mode = SaveModes.all
            # else:
            #     target_params.tracked.save_mode = SaveModes.error
            target_params.tracked.save_mode = SaveModes.all
            # target_params.tracked.save_mode = SaveModes.none

        target_params.save_states = states[:]
        _data = Data(_data_params, _logger)

        target = None

        if not _test_params.load:
            if iter_id == 0:
                # _tester_params.mode = TestModes.save_all_samples
                target = Target(target_params, rgb_input, _logger)
            else:
                # _tester_params.mode = TestModes.save_error_samples

                _train_params.load = Train.Modes.test_only
                _train_params.load_dir = load_dir

                for state in states:
                    if not status[state]:
                        """using relative oracle for states with too few samples to train so no loading"""
                        continue

                    _, model_params, _ = IBT.get_state_params(target_params, state)
                    if model_params is None:
                        _logger.warning(f'none model specified for state: {state} which differs from training')
                        continue

                    batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain

                    _load_dir = '{}_batch_{}'.format(state, iter_id - 1)
                    if not accumulative and iter_id > 1:
                        _load_dir = '{}_acc0'.format(_load_dir)
                    load_path = linux_path(results_dir_root, _load_dir)
                    batch_params._load_dir = load_path

                target = Train.run(_data, _trainer_params, _train_params, _logger, log_dir, args_in)
                if not target:
                    return None

        success = Test.run(target, _data, _tester_params, _test_params, _logger, log_dir, args_in)

        if _test_params.replace.target is not None:
            test_params.replace.target = _test_params.replace.target
            return None

        if not success:
            return None

        if _test_params.load:
            status['save'] = 0
            return status

        status['save'] = 1

        status_change = 0
        invalid_exists = 0

        if 'active' in states:
            n_samples = target.active.samples.count
            if not status['active'] and n_samples >= min_samples:
                status_change = 1
                status['active'] = 1
                _logger.info(f'found: {n_samples} samples for active')
            else:
                msg = f'found too few samples: {n_samples} / {min_samples} for active '
                if not status['active']:
                    invalid_exists = 1
                if n_samples < min_samples:
                    if not allow_too_few_samples:
                        raise AssertionError(msg)

                    _logger.warning(msg)

        if 'lost' in states:
            n_samples = target.lost.samples.count
            if not status['lost'] and n_samples >= min_samples:
                status['lost'] = 1
                status_change = 1
                _logger.info(f'found: {n_samples} samples for lost')
            else:
                msg = f'found too few samples: {n_samples} / {min_samples} for lost '
                if not status['lost']:
                    invalid_exists = 1
                if n_samples < min_samples:
                    if not allow_too_few_samples:
                        raise AssertionError(msg)
                    _logger.warning(msg)

        if 'tracked' in states:
            n_samples = target.tracked.samples.count
            if not status['tracked'] and n_samples >= min_samples:
                status['tracked'] = 1
                status_change = 1
                _logger.info(f'found: {n_samples} samples for tracked')
            else:
                msg = f'found too few samples: {n_samples} / {min_samples} for tracked '
                if not status['tracked']:
                    invalid_exists = 1
                if n_samples < min_samples:
                    if not allow_too_few_samples:
                        raise AssertionError(msg)
                    _logger.warning(msg)

        if not status_change and invalid_exists:
            raise AssertionError(f'no change in status for any states while one or more invalid states exist')

        status['save_dir'] = target.save_dir

        return status

    @staticmethod
    def eval(iter_id, states, load_dir, results_dir, results_dir_root,
             accumulative, train_params, trainer_params, data_params,
             log_dir, logger, args_in):
        """

        :param int iter_id:
        :param list[str] states:
        :param str load_dir:
        :param str results_dir:
        :param str results_dir_root:
        :type accumulative int
        :type train_params Train.Params
        :type trainer_params Trainer.Params
        :type data_params Data.Params
        :param str log_dir:
        :param CustomLogger logger:
        :param list args_in:
        :return:
        """
        assert iter_id > 0, "invalid iter_id for evaluation mode"

        _train_params = paramparse.copy_recursive(train_params)  # type: Train.Params
        _trainer_params = paramparse.copy_recursive(trainer_params)  # type: Trainer.Params
        target_params = _trainer_params.target  # type: Target.Params

        """
        evaluating all trained states simultaneously
        load latest trained model for all states from previous iteration ID since eval IDs are 1-based
        """
        for _state in states:
            state_params, model_params, async_mode = IBT.get_state_params(
                target_params, _state)  # type: PolicyBase.Params, ModelBase.Params, int
            batch_params = model_params.batch  # type: ModelBase.Params.BatchTrain
            batch_params.save_samples = 1

            _load_dir = '{}_batch_{}'.format(_state, iter_id - 1)
            if not accumulative and iter_id > 1:
                _load_dir = '{}_acc0'.format(_load_dir)
            batch_params._load_dir = linux_path(results_dir_root, _load_dir)

        results_dir = linux_path(results_dir, 'eval')

        """save generated data"""
        _train_params.save = 1
        # _train_params.results_dir = results_dir
        _train_params.results_dir = results_dir

        """single pass through training trajectories"""
        _trainer_params.max_count = 1
        _trainer_params.max_pass = 1

        """Data generation with trained model"""
        _train_params.load = Train.Modes.continue_training
        _train_params.load_dir = load_dir

        _logger = IBT._add_header(logger, 'eval', iter_id - 1, states)
        data = Data(data_params, _logger)
        trained_target = Train.run(data, _trainer_params, _train_params, _logger, log_dir, args_in)

        return trained_target

    @staticmethod
    def _skip_train(iter_id, ibt, state_id, state):
        """

        :param int iter_id:
        :param IBT.Params ibt:
        :param int state_id:
        :param state:
        :return:
        """
        skip = (
                   # not all phases are enabled and training is not included in the enabled phases
                       ibt.phases and IBT.Phases.training not in ibt.phases) or (
                   # current iteration is disabled
                       iter_id < ibt.start_iter or
                       # current iteration is the first one enabled but
                       (iter_id == ibt.start_iter and
                        # state is marked for skipping or
                        state in ibt.skip_states or
                        # either training is only enabled for all subsequent iterations
                        (ibt.start_phase > IBT.Phases.training or
                         # or for later (testing) states
                         ibt.start_state > state_id)

                        )
               )
        return skip

    @staticmethod
    def _skip_data_from_trainer(iter_id, ibt, state_id):

        not_skip = (
                       # either all phases are enabled or data generation phase is enabled
                           not ibt.phases or IBT.Phases.data_generation in ibt.phases
                   ) and (
                       # this is not the start iteration in which case all phases and states qualify
                           iter_id > ibt.start_iter or
                           (
                               # this is the start iteration in which case either
                                   iter_id == ibt.start_iter and
                                   (
                                       # start_state is less than the current one whereupon both training and
                                       # data_generation phases of current state  will be run as long as
                                       # start_phase is either of these or
                                           (
                                                   ibt.start_state < state_id and ibt.start_phase <=
                                                   IBT.Phases.training) or
                                           # start_state is same as the current one in which case start_phase
                                           #  must be data_generation too
                                           (
                                                   ibt.start_state == state_id and ibt.start_phase ==
                                                   IBT.Phases.data_generation)
                                   )
                           )
                   )
        return not not_skip

    @staticmethod
    def _skip_eval(iter_id, ibt, test_mode):
        """

        :param int iter_id:
        :param IBT.Params ibt:
        :param int test_mode:
        :return:
        """

        not_skip = (not ibt.phases or IBT.Phases.evaluation in ibt.phases) and \
                   (iter_id - 1 > ibt.start_iter or
                    (iter_id - 1 == ibt.start_iter and ibt.start_phase <= IBT.Phases.evaluation)) and (
                           test_mode == 1 or test_mode == 2)
        return not not_skip

    @staticmethod
    def _skip_test(iter_id, ibt, test_mode):
        """

        :param int iter_id:
        :param IBT.Params ibt:
        :param int test_mode:
        :return:
        """

        not_skip = (not ibt.phases or IBT.Phases.testing in ibt.phases) and \
                   (iter_id - 1 > ibt.start_iter or (
                           iter_id - 1 == ibt.start_iter and ibt.start_phase <= IBT.Phases.testing)) and \
                   (test_mode == 0 or test_mode == 2)

        return not not_skip

    @staticmethod
    def _skip_data_from_tester(iter_id, ibt):

        not_skip = (
                       # either all phases are enabled or data generation phase is enabled
                           not ibt.phases or IBT.Phases.data_generation in ibt.phases
                   ) and (
                       # this is not the start iteration in which case all phases and states qualify
                           iter_id > ibt.start_iter or
                           (
                               # this is the start iteration in which case start_phase must be data_generation
                                   iter_id == ibt.start_iter and ibt.start_phase == IBT.Phases.data_generation
                           )
                   )
        return not not_skip

    @staticmethod
    def is_start(ibt, iter_id, phase, state_id=None):
        _is_start = ibt.start_iter == iter_id \
                    and ibt.start_phase == phase

        if not _is_start or state_id is None:
            return _is_start

        return ibt.start_state == state_id

    @staticmethod
    def run(params, logger, args_in):
        """

        :param MainParams params:
        :param logger:
        :param args_in:
        :return:
        """

        ibt = params.ibt  # type: IBT.Params
        ibt.process()

        assert ibt.states, 'ibt state must be provided'
        assert ibt.async_dir, 'ibt async_dir must be provided'

        if ibt.accumulative and ibt.data_from_tester:
            logger.warning('turning off accumulative training for data_from_tester mode')
            ibt.accumulative = 0

        # if ',' in ibt.state:
        #     ibt_states = ibt.state.split(',')
        # else:
        #     ibt_states = [ibt.state, ]

        ibt_states = ibt.states  # type: list

        assert all(
            x in ('lost', 'tracked', 'active') for x in ibt_states), f"one or more invalid IBT states: {ibt_states}"

        # ibt.state = None

        # default_params = paramparse.copy_recursive(params)  # type: ModelBase.Params.BatchTrain
        default_params = paramparse.copy_recursive(params)

        async_dir = ibt.async_dir

        if not async_dir.startswith('log'):
            async_dir = linux_path('log', async_dir)

        results_dir_root = params.train.results_dir_root
        params.train.results_dir_root = ''

        if not results_dir_root:
            results_dir_root = async_dir

        results_dir_root = linux_path(results_dir_root, params.train.results_dir)

        if not results_dir_root.startswith('log'):
            results_dir_root = linux_path('log', results_dir_root)

        # iter_params = IterativeBatchTrainParams.IterationParams()

        ibt_status_file = linux_path(results_dir_root, 'ibt_status_file.txt')
        os.makedirs(results_dir_root, exist_ok=True)

        load_dir = None
        results_dir = None
        default_test_params = None
        iter_params = paramparse.copy_recursive(default_params)  # type: MainParams
        test_params = paramparse.copy_recursive(default_params)  # type: MainParams

        n_ibt_states = len(ibt_states)
        db_path = dict(zip(ibt_states, ('',) * n_ibt_states))

        states_str = '_'.join(ibt_states)
        ibt_test_cfgs = ibt.get_test_cfgs()  # type: dict
        ibt_cfgs = ibt.get_cfgs()  # type: dict

        _status = None

        for iter_id in range(ibt.n_iters + 1):

            """
            Testing / Evaluation Phase
            Performed jointly for all trained policies
            """
            if iter_id > 0:
                try:
                    test_cfg = ibt_test_cfgs[str(iter_id - 1)]
                except KeyError:
                    test_cfg = ''
                else:
                    if test_cfg and test_cfg != '_':
                        default_test_params = paramparse.copy_recursive(iter_params)
                        paramparse.process(default_test_params, cfg=test_cfg, cmd=False)
                        default_test_params.process()

                if test_cfg == '_' or default_test_params is None:
                    test_params = iter_params
                else:
                    test_params = default_test_params

            if iter_id > 0 and iter_id > ibt.start_iter and iter_id - 1 in ibt.test_iters:
                test_mode = test_params.test.mode

                if IBT._skip_eval(iter_id, ibt, test_mode):
                    print(f'skipping iteration {iter_id - 1} policy evaluation phase')
                else:
                    print(f'\nrunning iteration {iter_id - 1} policy evaluation phase\n')
                    if not IBT.eval(iter_id, ibt_states, load_dir, results_dir,
                                    results_dir_root,
                                    test_params.ibt.accumulative,
                                    test_params.train, test_params.trainer,
                                    test_params.data,
                                    test_params.log_dir,
                                    logger, args_in):
                        raise AssertionError('policy evaluation phase failed')

                if IBT._skip_test(iter_id, ibt, test_mode):
                    print(f'skipping iteration {iter_id - 1} testing phase')
                else:
                    logger.warning(f'\nrunning iteration {iter_id - 1} testing phase\n')
                    success = IBT.test(iter_id, ibt_states, load_dir,
                                       test_params.ibt.accumulative,
                                       test_params.train, test_params.trainer,
                                       test_params.test, test_params.tester,
                                       test_params.data,
                                       results_dir_root,
                                       test_params.log_dir, logger,
                                       args_in)
                    if test_params.test.replace.target is not None:
                        params.test.replace.target = test_params.test.replace.target
                        return True

                    if not test_params.test.load and not success:
                        raise AssertionError('testing phase failed')

            if iter_id >= ibt.n_iters:
                break

            if ibt.data_from_tester:
                ibt_key = '{}:data_tester'.format(iter_id)
                phase = 'tester data generation'

                if IBT._skip_data_from_tester(iter_id, ibt):
                    if not os.path.isfile(ibt_status_file):
                        raise AssertionError(f'ibt_status_file required to skip iteration {iter_id} {phase} '
                                             f'phase not found: {ibt_status_file}')
                    print(f'skipping iteration {iter_id} {phase} phase')
                    _status = IBT._load_status(ibt_status_file, ibt_key)
                else:
                    """
                    Data Generation from Tester Phase
                    """
                    save_dir = 'dtest_batch_{}'.format(iter_id)
                    if not ibt.accumulative and iter_id > 0:
                        save_dir = '{}_acc0'.format(save_dir)

                    results_dir = linux_path(results_dir_root, save_dir)

                    if ibt.start_seq >= 0 and IBT.is_start(ibt, iter_id, IBT.Phases.data_generation):
                        iter_params.train.start = ibt.start_seq
                        pass

                    print(f'\nrunning iteration {iter_id} {phase} phase\n')
                    _status = IBT.data_from_tester(
                        _status,
                        iter_id, ibt_states,
                        load_dir, results_dir, results_dir_root,
                        ibt.min_samples,
                        ibt.allow_too_few_samples,
                        ibt.accumulative,
                        iter_params.train, iter_params.trainer,
                        iter_params.test, iter_params.tester,
                        iter_params.data,
                        iter_params.log_dir,
                        logger, args_in)

                    if iter_params.test.replace.target is not None:
                        params.test.replace.target = iter_params.test.replace.target
                        return True

                    if _status is None:
                        raise AssertionError('data generation from tester phase failed')

                    if _status['save']:
                        IBT.save_status(ibt_status_file, update_dict={ibt_key: _status})
                    else:
                        logger.warning('skipping saving IBT status')

            for ibt_state_id, ibt_state in enumerate(ibt_states):
                cfg_id = f'{iter_id}{ibt_state_id}'
                try:
                    cfg = ibt_cfgs[cfg_id]
                except KeyError:
                    pass
                else:
                    if cfg == '_':
                        iter_params = paramparse.copy_recursive(default_params)  # type: MainParams
                    elif cfg:
                        paramparse.process(iter_params, cfg=cfg, cmd=False)
                        iter_params.process()

                if not ibt.data_from_tester:
                    if iter_id == 0:
                        """Asynchronous data generation"""
                        results_dir = linux_path(async_dir, '{}_async'.format(states_str))
                    else:
                        """Data generation with trained model"""
                        _save_dir = '{}_batch_{}'.format(states_str, iter_id)
                        if not params.ibt.accumulative and iter_id > 0:
                            _save_dir = '{}_acc0'.format(_save_dir)
                        results_dir = linux_path(results_dir_root, _save_dir)

                    """
                    Data Generation from Trainer Phase
                    """
                    ibt_key = '{}:{}:data'.format(iter_id, ibt_state)
                    phase = 'trainer data generation'
                    if IBT._skip_data_from_trainer(iter_id, ibt, ibt_state_id):
                        if not os.path.isfile(ibt_status_file):
                            raise AssertionError(f'ibt_status_file required to skip iteration {iter_id} {phase} phase '
                                                 f'not found: {ibt_status_file}')
                        print(f'skipping iteration {iter_id} {ibt_state} {phase} phase')
                        try:
                            _status = IBT._load_status(ibt_status_file, ibt_key)
                        except KeyError:
                            """old style status"""
                            __save_dir = IBT._load_status(ibt_status_file, ibt_key + ':save_dir')
                            _status = {
                                'save_dir': __save_dir
                            }
                    else:

                        print(f'\nrunning iteration {iter_id} {ibt_state} trainer {phase} phase\n')
                        _status = IBT.data_from_trainer(
                            _status,
                            iter_id, ibt_states, ibt_state_id, load_dir, results_dir,
                            results_dir_root, ibt.load_weights,
                            iter_params.ibt.accumulative,
                            iter_params.train, iter_params.trainer,
                            iter_params.data,
                            iter_params.log_dir,
                            logger, args_in)
                        if _status is None:
                            raise AssertionError('data generation phase failed')
                        IBT.save_status(ibt_status_file, update_dict={ibt_key: _status})

                """
                Training Phase
                """

                try:
                    load_dir = _status['save_dir']
                except KeyError:
                    logger.warning('save_dir not found in _status')

                if ibt.phases and IBT.Phases.training not in ibt.phases:
                    pass
                else:

                    curr_db_path = linux_path(load_dir, ibt_state)
                    if ibt.accumulative and db_path[ibt_state]:
                        db_path[ibt_state] = '{},{}'.format(db_path[ibt_state], curr_db_path)
                    else:
                        db_path[ibt_state] = curr_db_path

                # ibt_key = '{}:train:save_dir'.format(iter_id)
                if IBT._skip_train(iter_id, ibt, ibt_state_id, ibt_state):
                    print(f'skipping iteration {iter_id} {ibt_state} training phase')
                    # load_dir = load_ibt_status(ibt_status_file, ibt_key)
                    continue

                if not _status[ibt_state]:
                    print(f'skipping iteration {iter_id} {ibt_state} training phase due to insufficient generated data')
                    continue

                print(f'\nrunning iteration {iter_id} {ibt_state} training phase\n')
                success = IBT.train(iter_id, db_path[ibt_state], results_dir_root,
                                    ibt_state, ibt.load_weights, ibt.load_prev,
                                    iter_params.ibt.accumulative,
                                    iter_params.ibt.save_suffix,
                                    iter_params.train, iter_params.trainer,
                                    iter_params.data,
                                    iter_params.log_dir, logger, args_in)
                if not success:
                    raise AssertionError('training phase failed')

        return True
