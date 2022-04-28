from paramparse import MultiCFG, MultiPath

from utilities import BaseParams


class IBTParams(BaseParams):
    """
    Iterative Batch Train Parameters

    has to be defined here instead of IBT to prevent circular dependency between
    IBT and that module since Params needs IBTParams and IBT needs Params for intellisense

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
