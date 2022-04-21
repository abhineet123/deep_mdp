import time
import os
import logging

from paramparse import MultiPath

from data import Data
from trainer import Trainer
from tester import Tester
from target import Target
from utilities import SCP, CustomLogger, BaseParams, linux_path


class RunParams(BaseParams):
    """
    Iterative Batch Train Parameters
    :type save_prefix: MultiPath
    :type results_dir_root: MultiPath
    :type results_dir: MultiPath

    :ivar seq_set: 'Numeric ID of the data set from which the training sequences '
               'have to be taken as defined in the Data.get_sequences(); '
               'at present only sequences from a single data set can be trained on '
               'in a single run',
            0: 'MOT2015',
            1: 'MOT2017',
            2: 'MOT2017_SDP',
            3: 'MOT2017_DPM',
            4: 'KITTI',
            5: 'GRAM_ONLY',
            6: 'IDOT',
            7: 'DETRAC',
            8: 'LOST',
            9: 'ISL',
            10: 'GRAM',  # combined sequence set; named GRAM for convenience
            11: 'MNIST_MOT',

    :ivar  seq: Numeric IDs of the sequences on which
    training has to be performed as defined in the Data.get_sequences()

    """

    def __init__(self):
        self.seq_set = -1
        self.seq = ()
        self.seq_set_info = MultiPath()
        self.sample = ""

        self.load = 0
        self.save = 1
        self.start = 0

        self.save_prefix = MultiPath()
        self.results_dir_root = MultiPath()
        self.results_dir = MultiPath()

    def _synchronize(self, src):
        """
        :type src: RunParams
        """
        if self.seq_set < 0:
            self.seq_set = src.seq_set
        if not self.seq:
            self.seq = src.seq
        if not self.results_dir_root:
            self.results_dir_root = src.results_dir_root
        if not self.results_dir:
            self.results_dir = src.results_dir


class Train:
    class Params(RunParams):
        """
        :type seq_set: int
        :type seq: (int, )

        :ivar load:
                1: Load a previously trained tracker and test;
                2: Load a previously trained tracker and continue training;
                0: train from scratch,
                -1: skip loading target and test - ,

        :ivar load_id: ID of the sequence for which to load trained target if
        load_dir is not provided;

        :ivar load_dir: directory from where to load trained target; overrides
        load_id

        :ivar start: ID of the sequence from which to start training; if load_id
        and load_dir are not provided, training is continued after loading the
        target corresponding to the sequence preceding start_id; this ID is specified
        relative to the IDs provided in seq_ids

        :ivar save: Save the trained tracker to disk so it can be loaded later;
        only matters if load is disabled

        :ivar save_prefix: Prefix in the name of the file into which the trained
        tracker is to be saved

        :ivar load_prefix: prefix in the name of the file from which the previously
        trained tracker has to be loaded for testing

        :ivar results_dir: Directory where training results files are written to

        :ivar active_pt: set active policy pre-training mode:
            0: disable
            1: enable and load previously trained model if it exists, train otherwise
            2: enable and overwrite previously trained model if it exists
            3: enable and load previously trained model if it exists, raise error otherwise
        """

        def __init__(self):
            RunParams.__init__(self)

            self.load = 0

            self.load_dir = ''
            self.load_id = -1

            self.seq_set = 4
            self.seq = (5,)

            self.active_pt = 0
            self.active_pt_dir = MultiPath()

            self.load_prefix = MultiPath('trained')
            self.save_prefix = MultiPath('trained')
            self.results_dir_root = MultiPath()
            self.results_dir = MultiPath('log')

    class Modes:
        no_train, train_from_scratch, test_only, continue_training = range(-1, 3)

    @staticmethod
    def run(data, trainer_params, train_params, logger, log_dir, args_in):
        """
        :type data: Data
        :type trainer_params: Trainer.Params
        :type train_params: Train.Params
        :type logger: logging.RootLogger | logging.logger
        :type log_dir: str
        :type args_in: list
        :rtype: Target
        """

        n_handlers = len(logger.handlers)

        trainer = Trainer(trainer_params, logger, args_in)

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        no_load = (train_params.load == Train.Modes.no_train)

        log_file = ''
        logging_handler = None

        results_dir = train_params.results_dir
        if train_params.results_dir_root:
            results_dir = linux_path(train_params.results_dir_root, results_dir)

        if train_params.load != Train.Modes.train_from_scratch:
            if log_dir:
                log_file, logging_handler = CustomLogger.add_file_handler(log_dir, 'trainer_load', logger)
                logger.info('Saving trainer loading log to {}'.format(log_file))

            load_dir = train_params.load_dir

            if load_dir:
                logger.info('Loading trainer from custom dir: {:s}'.format(load_dir))
            else:
                load_seq_id = train_params.load_id
                if load_seq_id < 0:
                    """load from the last training sequence if trainer.start_id is 0 
                    and from the one preceding it otherwise"""
                    load_seq_id = train_params.seq[train_params.start - 1]

                logger.info('Loading trainer from sequence {:d} in set {:d}'.format(
                    load_seq_id, train_params.seq_set))

                if not data.initialize(train_params.seq_set, load_seq_id, 0, logger):
                    raise AssertionError('Data module could not be initialized')

                load_fname = '{:s}_{:s}_{:d}_{:d}'.format(
                    data.seq_set, data.seq_name, data.start_frame_id + 1, data.end_frame_id + 1)
                load_prefix = train_params.load_prefix
                if load_prefix:
                    load_fname = '{:s}_{:s}'.format(load_prefix, load_fname)

                load_dir = linux_path(results_dir, load_fname)

            if not trainer.load(load_dir, no_load=no_load):
                raise AssertionError('Trained target could not be loaded')

            logger.info('Trained target loaded successfully')

        if no_load:
            return trainer.target

        if train_params.load == Train.Modes.test_only:
            """return trained model for testing
            """
            logger.info('Not continuing training')
        else:
            train_seq_ids = train_params.seq[train_params.start:]
            n_train_seq_ids = len(train_seq_ids)
            n_seq = len(train_params.seq)
            for _id, train_id in enumerate(train_seq_ids):
                if logging_handler is not None:
                    CustomLogger.remove_file_handler(logging_handler, logger)
                    logging_handler = None

                if log_dir:
                    log_file, logging_handler = CustomLogger.add_file_handler(log_dir, 'training_{}'.format(train_id),
                                                                              logger)
                    logger.info('Saving training {} log to {}'.format(train_id, log_file))

                logger.info('Running trainer on sequence {:d} in set {:d} ({:d} / {:d})'.format(
                    train_id, train_params.seq_set, _id + train_params.start + 1, n_seq))
                if not data.initialize(train_params.seq_set, train_id, 0, logger):
                    raise AssertionError('Data module failed to initialize with sequence {:d}'.format(train_id))

                seq_logger = CustomLogger(logger, names=(data.seq_name,), key='custom_header')

                # header = _header + ':{}'.format(data.seq_name)

                start_t = time.time()
                if not trainer.run(data, seq_logger):
                    raise AssertionError('Trainer failed on sequence {:d} : {:s}'.format(train_id, data.seq_name))
                end_t = time.time()

                if trainer_params.mode in Trainer.Modes.batch:
                    """batch training mode"""
                    seq_logger.info(f'Done batch training. Time Taken: {end_t - start_t} seconds.')
                    break

                seq_logger.info(f'Done training on sequence {_id + 1} / {n_train_seq_ids}. '
                                f'Time Taken: {end_t - start_t} seconds.')

                if train_params.save:
                    save_fname = '{:s}_{:s}_{:d}_{:d}'.format(
                        data.seq_set, data.seq_name, data.start_frame_id + 1, data.end_frame_id + 1)
                    save_prefix = train_params.save_prefix

                    if save_prefix:
                        save_fname = '{:s}_{:s}'.format(save_prefix, save_fname)

                    save_dir = linux_path(results_dir, save_fname)
                    trainer.save(save_dir, log_file)

        if logging_handler is not None:
            CustomLogger.remove_file_handler(logging_handler, logger)

        assert len(logger.handlers) == n_handlers, f'Unexpected number of logging_handlers: {len(logger.handlers)} ' \
            f'Expected: {n_handlers}'

        # if train_paramser.mode > 0:
        #     """async training mode"""
        #     return None

        return trainer.target


class Test:
    class Params(RunParams):
        class Replace:
            def __init__(self):
                self.scp = SCP()
                self.token = ''
                self.modules = []
                self._target = None
                self._copy_excluded = ['_target']

            @property
            def target(self): return self._target

            @target.setter
            def target(self, target): self._target = target

            def reset(self):
                self.token = ''
                self.modules = []
                self._target = None

        """
        :type seq_set: int
        :type seq: (int, )
        :type load: int

        :ivar load: 'Load previously saved tracking results from file for evaluation or visualization'
                ' instead of running the tracker to generate new results;'
                'load=2 will load raw results collected online for each frame instead of the post-processed '
                'ones generated at the end of each target',

        :ivar save: Save tracking results to file;only matters if load is disabled

        :ivar save_prefix: Prefix in the name of the file into which the tracking
        results are to be saved

        :ivar results_dir: Directory where the tracking results file is saved
        in

        :ivar evaluate: 'Enable evaluation of the tracking result; '
                    'only works if the ground truth for the tested sequence is available; '
                    '1: evaluate each sequence and all combined; '
                    '2: evaluate sequences incrementally as well(i.e. seq (1,) (1,2), (1,2,3) and so on); ',
        :ivar eval_dist_type: 'Type of distance measure between tracking result and ground truth '
                          'bounding boxes to use for evaluation:'
                          '0: intersection over union (IoU) distance'
                          '1: squared Euclidean distance; '
                          'only matters if evaluate is set to 1',

        :ivar eval_dir: Name of the Directory into which a summary of the evaluation
        result will be saved; defaults to results_dir if not provided

        :ivar enable_tb: enable tensorboard logging

        :ivar eval_file: Name of the file into which a summary of the evaluation
        result will be written if evaluation is enabled

        """

        def __init__(self):
            RunParams.__init__(self)

            self.eval_dir = MultiPath()

            self.mode = 1
            self.evaluate = 1
            self.eval_dist_type = 0
            self.eval_file = 'mot_metrics.log'

            self.subseq_postfix = 1
            self.enable_tb = 0

            self.replace = Test.Params.Replace()

        def synchronize(self, src, force):
            """

            :param Train.Params src:
            :param bool force:
            :return:
            """
            if not force:
                self._synchronize(src)
            else:
                self.seq_set_info = src.seq_set_info
                self.seq_set = src.seq_set
                self.seq = src.seq
                self.start = src.start
                self.sample = src.sample

    @staticmethod
    def run(trained_target, data, tester_params, test_params, logger, logging_dir, args_in):
        """
        test a trained target
        :type trained_target: Target
        :type data: Data
        :type tester_params: Tester.Params
        :type test_params: Test.Params
        :type logger: logging.RootLogger | logging.logger | CustomLogger
        :type logging_dir: str
        :type args_in: list
        :rtype: bool
        """

        if test_params.replace.modules:
            logger.warning('returning target with replacement modules: {}'.format(test_params.replace.modules))

            """Repeat code for better intellisence """
            if 'active' not in test_params.replace.modules:
                trained_target.active = None

            if 'tracked' not in test_params.replace.modules:
                trained_target.tracked = None

            if 'lost' not in test_params.replace.modules:
                trained_target.lost = None

            if 'tracker' not in test_params.replace.modules and 'templates' not in test_params.replace.modules:
                trained_target.templates = None

            if 'history' not in test_params.replace.modules:
                trained_target.history = None

            test_params.replace.target = trained_target

            return
        global_logger = logger

        assert test_params.start < len(test_params.seq), f"Invalid start_id: {test_params.start} " \
            f"for {len(test_params.seq)} sequences"

        n_handlers = len(global_logger.handlers)

        tester = Tester(trained_target, tester_params, global_logger, args_in)

        if logging_dir:
            os.makedirs(logging_dir, exist_ok=True)

        log_file = ''
        logging_handler = None
        success = True
        eval_path = load_dir = None

        evaluate = test_params.evaluate
        eval_dist_type = test_params.eval_dist_type
        # if evaluate > 1:
        #     """only combined evaluation
        #     """
        #     eval_dist_type = -1

        results_dir = test_params.results_dir
        results_dir_root = test_params.results_dir_root
        if results_dir_root:
            results_dir = linux_path(results_dir_root, results_dir)

        if test_params.seq_set_info:
            results_dir = linux_path(results_dir, test_params.seq_set_info)
        # combined_stats = {
        #     _state: pd.DataFrame(
        #         np.zeros((len(PolicyDecision.types), len(AnnotationStatus.types))),
        #         columns=AnnotationStatus.types,
        #         index=PolicyDecision.types,
        #     )
        #     for _state in ('active', 'lost', 'tracked')
        # }

        save_txt = f'saving results to: {results_dir}'
        if trained_target is not None and trained_target.rep_prefix:
            rep_prefix_str = '_'.join(trained_target.rep_prefix)
            test_params.save_prefix = '{}_{}'.format(test_params.save_prefix,
                                                     rep_prefix_str) if test_params.save_prefix else rep_prefix_str

        save_prefix = test_params.save_prefix
        if save_prefix:
            save_txt += f' with save_prefix: {save_prefix}'

        global_logger.info(save_txt)

        n_seq = len(test_params.seq)
        for _id, test_id in enumerate(test_params.seq[test_params.start:]):
            if logging_handler is not None:
                CustomLogger.remove_file_handler(logging_handler, global_logger)
                logging_handler = None

            if logging_dir:
                log_file, logging_handler = CustomLogger.add_file_handler(logging_dir, 'testing', global_logger)
                global_logger.info('Saving testing {} log to {}'.format(test_id, log_file))

            global_logger.info('Running tester on sequence {:d} in set {:d} ({:d} / {:d} )'.format(
                test_id, test_params.seq_set, _id + test_params.start + 1, n_seq))

            if not data.initialize(test_params.seq_set, test_id, 1, logger=global_logger):
                raise AssertionError('Data module failed to initialize with sequence {:d}'.format(test_id))

            save_dir = results_dir

            os.makedirs(save_dir, exist_ok=True)

            save_prefix = test_params.save_prefix
            if save_prefix:
                # save_fname = '{:s}_{:s}'.format(save_prefix, save_fname)
                save_dir = linux_path(save_dir, save_prefix)

            load_dir = save_dir

            eval_dir = test_params.eval_dir
            if not eval_dir:
                eval_dir = save_dir
            eval_path = linux_path(eval_dir, test_params.eval_file)

            seq_logger = CustomLogger(global_logger, names=(data.seq_name,), key='custom_header')
            tb_path = None
            if test_params.enable_tb:
                tb_path = linux_path(save_dir, "tb")

            if not tester.initialize(data, test_params.load, evaluate, tb_path, logger=seq_logger):
                raise AssertionError('Tester initialization failed on sequence {:d} : {:s}'.format(
                    test_id, data.seq_name))

            if tester.annotations is None:
                seq_logger.warning('Tester annotations unavailable so disabling evaluation')
                evaluate = 0

            if test_params.load:
                """load existing tracking results and optionally visualize or evaluate"""

                if test_params.subseq_postfix:
                    load_fname = '{:s}_{:d}_{:d}.txt'.format(data.seq_name, data.start_frame_id + 1,
                                                             data.end_frame_id + 1)
                else:
                    load_fname = '{:s}.txt'.format(data.seq_name)

                load_path = linux_path(load_dir, load_fname)
                if evaluate or tester_params.visualizer.mode[0]:
                    if test_params.load == 2:
                        seq_logger.warning('Loading raw tracking results')
                        load_path = load_path.replace('.txt', '.raw')
                    if not tester.load(load_path):
                        raise AssertionError('Tester loading failed on sequence {:d} : {:s}'.format(
                            test_id, data.seq_name))

                if tester.vis:
                    tester.visualizer.run(tester.input)

                if evaluate:
                    eval_dir = test_params.eval_dir
                    if not eval_dir:
                        eval_dir = load_dir
                    eval_path = linux_path(eval_dir, test_params.eval_file)
                    acc = tester.eval(load_path, eval_path, eval_dist_type)
                    if not acc:
                        raise AssertionError('Tester evaluation failed on sequence {:d} : {:s}'.format(
                            test_id, data.seq_name))
                    if evaluate == 2:
                        tester.accumulative_eval(load_dir, eval_path, seq_logger)

                continue

            """run the tester"""
            start_t = time.time()
            if not tester.run():
                raise AssertionError('Tester failed on sequence {:d} : {:s}'.format(test_id, data.seq_name))
            end_t = time.time()
            time_taken = end_t - start_t

            if test_params.save or evaluate:
                """save tracking results to file and optionally evaluate"""
                if test_params.subseq_postfix:
                    save_fname = '{:s}_{:d}_{:d}.txt'.format(data.seq_name, data.start_frame_id + 1,
                                                             data.end_frame_id + 1)
                else:
                    save_fname = '{:s}.txt'.format(data.seq_name)

                save_path = linux_path(save_dir, save_fname)
                tester.save(save_path, log_file)

                if evaluate:
                    if not tester.load(save_path):
                        success = False
                        break
                    acc = tester.eval(save_path, eval_path, eval_dist_type)
                    if not acc:
                        raise AssertionError('Tester evaluation failed on sequence {:d} : {:s}'.format(
                            test_id, data.seq_name))
                    if evaluate == 2:
                        tester.accumulative_eval(load_dir, eval_path, seq_logger)

            seq_logger.info('Done testing on sequence {:d} / {:d}. Time Taken: {:f} seconds.'.format(
                _id + test_params.start + 1, n_seq, time_taken))

            if not test_params.load and not evaluate and tester.input.n_frames > 0:
                seq_logger.info('Average speed: {:f} fps'.format(float(tester.input.n_frames) / time_taken))

        if evaluate:
            tester.accumulative_eval(load_dir, eval_path, global_logger)

        if logging_handler is not None:
            CustomLogger.remove_file_handler(logging_handler, global_logger)

        assert len(global_logger.handlers) == n_handlers, f'Unexpected number of logging_handlers: ' \
            f'{len(global_logger.handlers)} Expected: {n_handlers}'

        return success
