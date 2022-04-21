from __future__ import division
from collections import OrderedDict, Iterable
import pandas as pd
import numpy as np
import pickle
import math
import os


class MOTMetrics(object):
    def __init__(self, seqName=None):
        self.metrics = OrderedDict()
        self.cache_dict = OrderedDict()

        if seqName:
            self.seqName = seqName
        else:
            self.seqName = 0

        self.IDF1 = None
        self.IDP = None
        self.IDR = None
        self.recall = None
        self.precision = None
        self.n_gt_trajectories = None
        self.MT = None
        self.PT = None
        self.ML = None
        self.total_num_frames = None
        self.PTR = None
        self.FAR = None
        self.tp = None
        self.fp = None
        self.fn = None
        self.id_switches_rel = None
        self.fragments_rel = None
        self.MOTP = None
        self.MOTA = None
        self.MTR = None
        self.id_switches = None
        self.FM = None
        self.MLR = None
        self.n_gt = None
        self.n_tr = None
        self.MOTAL = None
        self.IDTP = None
        self.IDFP = None
        self.IDFN = None
        self.F1 = None
        self.total_cost = None
        self.n_tr_trajectories = None

        # Evaluation metrics
        self.register(name="IDF1", formatter='{:.2f}'.format)
        self.register(name="IDP", formatter='{:.2f}'.format)
        self.register(name="IDR", formatter='{:.2f}'.format)

        self.register(name="recall", display_name="Rcll", formatter='{:.2f}'.format)
        self.register(name="precision", display_name="Prcn", formatter='{:.2f}'.format)

        self.register(name="n_gt_trajectories", display_name="GT", formatter='{:.0f}'.format, write_mail=True)
        self.register(name="MT", formatter='{:.0f}'.format)
        self.register(name="PT", formatter='{:.0f}'.format)
        self.register(name="ML", formatter='{:.0f}'.format)

        self.register(name="total_num_frames", display_name="NUM", formatter='{:.0f}'.format, write_mail=True,
                      write_db=True)

        self.register(name="PTR", display_name="PT(%)", formatter='{:.2f}'.format)
        self.register(name="FAR", formatter='{:.2f}'.format)

        self.register(name="tp", display_name="TP", formatter='{:.0f}'.format)  # number of true positives
        self.register(name="fp", display_name="FP", formatter='{:.0f}'.format)  # number of false positives
        self.register(name="fn", display_name="FN", formatter='{:.0f}'.format)  # number of false negatives

        self.register(name="id_switches_rel", display_name="IDSR", formatter='{:.2f}'.format)
        self.register(name="fragments_rel", display_name="FMR", formatter='{:.2f}'.format)

        self.register(name="MOTP", formatter='{:.2f}'.format)

        self.register(name="MOTA", formatter='{:.2f}'.format)
        self.register(name="MTR", display_name="MT(%)", formatter='{:.2f}'.format)

        self.register(name="id_switches", display_name="IDS", formatter='{:.0f}'.format)
        self.register(name="FM", formatter='{:.0f}'.format)
        self.register(name="MLR", display_name="ML(%)", formatter='{:.2f}'.format)

        self.register(name="n_gt", display_name="GTO", formatter='{:.0f}'.format, write_mail=False,
                      write_db=True)  # number of ground truth detections
        self.register(name="n_tr", display_name="TRO", formatter='{:.0f}'.format, write_mail=False,
                      write_db=True)  # number of tracker detections minus ignored tracker detections

        self.register(name="MOTAL", formatter='{:.2f}'.format, write_mail=False)
        self.register(name="IDTP", formatter='{:.0f}'.format, write_mail=False)
        self.register(name="IDFP", formatter='{:.0f}'.format, write_mail=False)
        self.register(name="IDFN", formatter='{:.0f}'.format, write_mail=False)

        self.register(name="F1", display_name="F1", formatter='{:.2f}'.format, write_mail=False)
        self.register(name="total_cost", display_name="COST", formatter='{:.0f}'.format, write_mail=False)
        self.register(name="n_tr_trajectories", display_name="TR", formatter='{:.0f}'.format, write_db=True,
                      write_mail=False)

    def register(self, name=None, value=None, formatter=None,
                 display_name=None, write_db=True, write_mail=True):
        """Register a new metric.
        Params
        ------
        name: str
            Name of the metric. Name is used for computation and set as attribute.
        display_name: str or None
            Disoplay name of variable written in db and mail
        value:
        formatter:
            Formatter to present value of metric. E.g. `'{:.2f}'.format`
        write_db: boolean, default = True
            Write value into db
        write_mail: boolean, default = True
            Write metric in result mail to user
        """
        assert not name is None, 'No name specified'.format(name)

        if not value:
            value = 0

        self.__setattr__(name, value)

        if not display_name:
            display_name = name

        self.metrics[name] = {
            'name': name,
            'write_db': write_db,
            'formatter': formatter,
            'write_mail': write_mail,
            'display_name': display_name
        }

    def cache(self, name=None, value=None, func=None):
        assert not name is None, 'No name specified'.format(name)

        self.__setattr__(name, value)

        self.cache_dict[name] = {
            'name': name,
            'func': func
        }

    def __call__(self, name):
        return self.metrics[name]

    @property
    def names(self):
        """Returns the name identifiers of all registered metrics."""
        return [v['name'] for v in self.metrics.values()]

    @property
    def display_names(self):
        """Returns the display name identifiers of all registered metrics."""
        return [v['display_name'] for v in self.metrics.values()]

    @property
    def formatters(self):
        """Returns the formatters for all metrics that have associated formatters."""
        return dict(
            [(v['display_name'], v['formatter']) for k, v in self.metrics.items() if not v['formatter'] is None])

    # @property
    def val_dict(self, display_name=False, object="metrics"):
        """Returns dictionary of all registered values of object name or display_name as key.
        Params
        ------

       display_name: boolean, default = False
            If True, display_name of keys in dict. (default names)
        object: "cache" or "metrics", default = "metrics"
        """
        if display_name:
            key_string = "display_name"
        else:
            key_string = "name"
        print("object dict: ", object)
        val_dict = dict([(self.__getattribute__(object)[key][key_string], self.__getattribute__(key)) for key in
                         self.__getattribute__(object).keys()])
        return val_dict

    def val_db(self, display_name=True):
        """Returns dictionary of all registered values metrics to write in db."""
        if display_name:
            key_string = "display_name"
        else:
            key_string = "name"
        val_dict = dict([(self.metrics[key][key_string], self.__getattribute__(key))
                         for key in self.metrics.keys() if self.metrics[key]["write_db"]])
        return val_dict

    def val_mail(self, display_name=True):
        """Returns dictionary of all registered values metrics to write in mail."""
        if display_name:
            key_string = "display_name"
        else:
            key_string = "name"
        val_dict = dict([(self.metrics[key][key_string], self.__getattribute__(key)) for key in self.metrics.keys() if
                         self.metrics[key]["write_mail"]])
        return val_dict

    def to_dataframe(self, display_name=False, type=None):
        """Returns pandas dataframe of all registered values metrics. """
        if type == "mail":
            self.df = pd.DataFrame(self.val_mail(display_name=display_name), index=[self.seqName])
        else:
            self.df = pd.DataFrame(self.val_dict(display_name=display_name), index=[self.seqName])

    def update_values(self, value_dict=None):
        """Updates registered metrics with new values in value_dict. """
        if value_dict:
            for key, value in value_dict.items():
                if hasattr(self, key):
                    self.__setattr__(key, value)

    def print_type(self, object="metrics"):
        """Prints  variable type of registered metrics or caches. """
        print("OBJECT ", object)
        val_dict = self.val_dict(object=object)
        for key, item in val_dict.items():
            print("%s: %s; Shape: %s" % (key, type(item), np.shape(item)))

    def print_results(self):
        """Prints metrics. """
        result_dict = self.val_dict()
        for key, item in result_dict.items():
            print(key)
            print("%s: %s" % (key, self.metrics[key]["formatter"](item)))

    def save_dict(self, path):
        """Save value dict to path as pickle file."""
        with open(path, 'wb') as handle:
            pickle.dump(self.__dict__, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # def compute_metrics_per_sequence(self):
    #     raise NotImplementedError

    def compute_clearmot(self):
        # precision/recall etc.
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.
            self.precision = 0.
        else:
            self.recall = (self.tp / float(self.tp + self.fn)) * 100.
            self.precision = (self.tp / float(self.fp + self.tp)) * 100.
        if (self.recall + self.precision) == 0:
            self.F1 = 0.
        else:
            self.F1 = 2. * (self.precision * self.recall) / (self.precision + self.recall)
        if self.total_num_frames == 0:
            self.FAR = "n/a"
        else:
            self.FAR = (self.fp / float(self.total_num_frames))
        # compute CLEARMOT
        if self.n_gt == 0:
            self.MOTA = -float("inf")
        else:
            self.MOTA = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)) * 100.

        if self.tp == 0:
            self.MOTP = 0
        else:
            self.MOTP = (1 - self.total_cost / float(self.tp)) * 100.
        if self.n_gt != 0:
            if self.id_switches == 0:
                self.MOTAL = (1 - (self.fn + self.fp + self.id_switches) / float(self.n_gt)) * 100.0
            else:
                self.MOTAL = (1 - (self.fn + self.fp + math.log10(self.id_switches)) / float(self.n_gt)) * 100.

        # calculate relative IDSW and FM
        if self.recall != 0:
            self.id_switches_rel = self.id_switches / self.recall
            self.fragments_rel = self.FM / self.recall

        else:
            self.id_switches_rel = 0
            self.fragments_rel = 0

        # ID measures
        try:
            IDPrecision = self.IDTP / (self.IDTP + self.IDFP)
        except ZeroDivisionError:
            IDPrecision = 0

        try:
            IDRecall = self.IDTP / (self.IDTP + self.IDFN)
        except ZeroDivisionError:
            IDRecall = 0

        self.IDF1 = 2 * self.IDTP / (self.n_gt + self.n_tr)
        if self.n_tr == 0:
            IDPrecision = 0

        self.IDP = IDPrecision * 100
        self.IDR = IDRecall * 100
        self.IDF1 = self.IDF1 * 100

        if self.n_gt_trajectories == 0:
            self.MTR = 0.
            self.PTR = 0.
            self.MLR = 0.
        else:
            self.MTR = self.MT * 100. / float(self.n_gt_trajectories)
            self.PTR = self.PT * 100. / float(self.n_gt_trajectories)
            self.MLR = self.ML * 100. / float(self.n_gt_trajectories)

    def compute_metrics_per_sequence(self,
                                     sequence, pred_file, gt_file, gtDataDir,
                                     benchmark_name):
        import matlab.engine

        try:
            eng = matlab.engine.start_matlab()
            print("MATLAB successfully connected for {}".format(sequence))
        except:
            raise Exception("MATLAB could not connect!")

        curr_path = os.path.dirname(os.path.realpath(__file__))

        matlab_devkit_path = os.path.join(curr_path, 'matlab_devkit')
        matlab_devkit_utils = os.path.join(matlab_devkit_path, 'utils')

        eng.addpath(matlab_devkit_path, nargout=0)
        eng.addpath(matlab_devkit_utils, nargout=0)

        results = eng.evaluateTracking(sequence, pred_file, gt_file, gtDataDir, benchmark_name, nargout=5)
        eng.quit()
        update_dict = results[4]
        self.update_values(update_dict)
