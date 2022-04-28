from data import Data
from trainer import Trainer
from tester import Tester
from run import Train, Test

from ibt_params import IBTParams

from utilities import parse_seq_IDs, x11_available, disable_vis, set_recursive, BaseParams


class Params(BaseParams):
    """
    root parameters - has to be defined here instead of main to prevent circular dependency between
     IBT and that module since Params needs IBT and IBT needs Params for intellisense

    :type cfg: str
    :type data: Data.Params
    :type train: Train.Params
    :type test: Test.Params
    :type tester: Tester.Params
    :type trainer: Trainer.Params
    :type ibt: IBTParams


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

        self.ibt = IBTParams()

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
