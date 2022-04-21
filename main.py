try:
    """has to be imported before any other package to avoid annoying 
    bad_alloc issues due to obscure conflicts with one or more other 
    packages including pytorch"""
    import matlab.engine
except ImportError:
    pass

import os
import sys
import time

# import paramparse
from _paramparse_ import paramparse

from utilities import CustomLogger, SIIF, profile, id_token_to_cmd_args

from data import Data
from trainer import Trainer

from ibt import IBT, MainParams
from run import Train, Test


def main(params):
    """

    :param MainParams params:
    :return:
    """

    if not params.test.load and not params.tester.res_from_gt and params.train.active_pt:

        if params.train.load:
            """load previously trained model only"""
            params.train.active_pt = 3

        """active state does not depend on the other states for its training data generation
        """
        _logger.info('starting active policy pre-training')
        if not IBT.pretrain_active(params.train, params.trainer, params.data,
                                   params.log_dir, _logger, args_in):
            raise AssertionError('active policy pre-training failed')
        _logger.info('completed active policy pre-training')
    else:
        _logger.warning('active policy pre-training is disabled')

    if params.mode == 1:
        """run iterative batch training"""
        success = IBT.run(params, _logger, args_in)
        if not success:
            raise AssertionError('iterative batch training failed')
    else:
        _data = Data(params.data, _logger)
        """run training and testing"""
        _trained_target = None
        if not params.test.load and not params.tester.res_from_gt:
            train_logger = CustomLogger(_logger, names=('train',), key='custom_header')
            _trained_target = Train.run(_data, params.trainer, params.train, train_logger, params.log_dir, args_in)
            if params.trainer.mode != Trainer.Modes.standard:
                return
            if _trained_target is None:
                return
        test_logger = CustomLogger(_logger, names=('test',), key='custom_header')
        Test.run(_trained_target, _data, params.tester, params.test, test_logger, params.log_dir, args_in)


if __name__ == '__main__':
    """print command for debugging of batch jobs"""
    command = ' '.join(sys.argv)
    print(command)

    """get parameters"""
    _params = MainParams()
    with profile('paramparse', enable=1):
        args_in = paramparse.process(_params)
    _params.process(args_in)

    """check if external image viewer SIIF (https://github.com/abhineet123/PTF/blob/master/show_images_in_folder.py)
    is running and setup environment variables to dump images to its monitoring folder instead of showing them;
    helps with debugging since the image window, running in a separate process does not freeze during a breakpoint
    """
    SIIF.setup()

    """setup logger"""
    _logger = CustomLogger.setup(__name__)

    if _params.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = _params.gpu

    if _params.tee_log:
        """disable buffering to ensure in-order output in the tee logging fine"""
        os.environ["PYTHONUNBUFFERED"] = '1'

    # # check if server mode is enabled
    # if params.server.mode:
    #     server = Server(params.server, params.trainer, params.tester, _logger)
    #     server.run()
    #     sys.exit(0)

    replace = _params.test.replace
    if replace.modules:
        cmd_args, tee_id = id_token_to_cmd_args(replace.token, replace.scp)

        cmd_args = list(cmd_args)

        main_params = MainParams()
        paramparse.process(main_params, cmd_args=cmd_args, allow_unknown=1)

        main_params.test.replace = replace

        main(main_params)

        assert replace.target is not None, "replacement target not found"

        _params.trainer.target.replacement = replace.target

        _params.test.save_prefix = '{}_rep_{}'.format(_params.test.save_prefix, tee_id)

        replace.reset()


    main(_params)
