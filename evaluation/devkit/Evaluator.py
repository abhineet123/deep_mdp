import os
import sys

sys.path.append(os.path.abspath(os.getcwd()))

import multiprocessing
import time
import numpy as np

# from collections import defaultdict
# from evaluation.devkit.MOT.MOT_metrics import MOTMetrics
import multiprocessing as mp
import pandas as pd

from evaluation.devkit.Metrics import MOTMetrics


class MotEvaluator(object):
    """ The `MotEvaluator` class runs evaluation per sequence and computes the overall performance on the benchmark"""

    def __init__(self):
        self.results = None
        self.overall_results = None
        self.type = "MOT"

    def run(self, gtfiles, tsfiles, datadir, sequences, benchmark_name):
        """
        Params
        -----
        benchmark_name: Name of benchmark, e.g. MOT17
        gt_dir: directory of folders with gt data, including the c-files with sequences
        res_dir: directory with result files
            <seq1>.txt
            <seq2>.txt
            ...
            <seq3>.txt
        eval_mode:
        seqmaps_dir:
        seq_file: File name of file containing sequences, e.g. 'c10-train.txt'
        save_pkl: path to output directory for final results
        """

        print('Evaluating on {} ground truth files and {} test files.'.format(len(gtfiles), len(tsfiles)))
        print('\n'.join(gtfiles))
        print('\n'.join(tsfiles))

        start_time = time.time()

        self.gtfiles, self.tsfiles = gtfiles, tsfiles
        self.datadir = datadir
        self.sequences = sequences
        self.benchmark_name = benchmark_name

        # error_traceback = ""

        self.MULTIPROCESSING = 1
        MAX_NR_CORES = multiprocessing.cpu_count()
        # self.NR_CORES = MAX_NR_CORES
        # set number of core for mutliprocessing
        if self.MULTIPROCESSING:
            self.NR_CORES = np.minimum(MAX_NR_CORES, len(self.tsfiles))
        # try:

        """ run evaluation """
        self.eval()

        # calculate overall results
        results_attributes = self.overall_results.metrics.keys()

        for attr in results_attributes:
            """ accumulate evaluation values over all sequences """
            try:
                self.overall_results.__dict__[attr] = sum(obj.__dict__[attr] for _, obj in self.results.items())
            except:
                pass
        cache_attributes = self.overall_results.cache_dict.keys()
        for attr in cache_attributes:
            """ accumulate cache values over all sequences """
            try:
                self.overall_results.__dict__[attr] = self.overall_results.cache_dict[attr]['func'](
                    [obj.__dict__[attr] for _, obj in self.results.items()])
            except:
                pass
        print("evaluation successful")

        # Compute clearmot metrics for overall and all sequences
        for _, res in self.results.items():
            res.compute_clearmot()

        self.overall_results.compute_clearmot()

        self.accumulate_df(type="mail")
        self.failed = False
        # error = None

        # except Exception as e:
        #     print(str(traceback.format_exc()))
        #     print("<br> Evaluation failed! <br>")
        #
        #     error_traceback += str(traceback.format_exc())
        #     self.failed = True
        #     self.summary = None

        end_time = time.time()

        self.duration = (end_time - start_time) / 60.

        # ======================================================
        # Collect evaluation errors
        # ======================================================
        # if self.failed:
        #
        #     startExc = error_traceback.split("<exc>")
        #     error_traceback = [m.split("<!exc>")[0] for m in startExc[1:]]
        #
        #     error = ""
        #
        #     for err in error_traceback:
        #         error += "Error: %s" % err
        #
        #     print("Error Message", error)
        #     self.error = error
        #     print("ERROR %s" % error)

        print("Evaluation Finished in {} sec".format(self.duration))
        print("Your Results")
        str_summary = self.render_summary()
        print(str_summary)
        # save results if path set
        # if save_pkl:
        #
        #     self.Overall_Results.save_dict(
        #         os.path.join(save_pkl, "%s-%s-overall.pkl" % (self.benchmark_name, self.mode)))
        #     for res in self.results:
        #         res.save_dict(os.path.join(save_pkl, "%s-%s-%s.pkl" % (self.benchmark_name, self.mode, res.seqName)))
        #     print("Successfully save results")

        return self.overall_results, self.results, self.summary, str_summary

    # def eval(self):
    #     raise NotImplementedError

    def eval(self):

        # print("Check prediction files")
        # error_message = ""
        for pred_file in self.tsfiles:
            # print(pred_file)
            # check if file is comma separated
            try:
                df = pd.read_csv(pred_file, header=None, sep=",")
            except pd.errors.EmptyDataError:
                print('empty file found: {}'.format(pred_file))
                continue

            if len(df.columns) == 1:
                f = open(pred_file, "r")
                error_message = "Submission %s not in correct form. Values in file must be comma separated." \
                                "Current form:<br>%s<br>%s<br>.........<br>" % (
                                    pred_file.split("/")[-1], f.readline(), f.readline())
                raise Exception(error_message)

            df.groupby([0, 1]).size().head()
            count = df.groupby([0, 1]).size().reset_index(name='count')

            # check if any duplicate IDs
            if any(count["count"] > 1):
                doubleIDs = count.loc[count["count"] > 1][[0, 1]].values
                error_message = "Found duplicate ID/Frame pairs in sequence %s." % pred_file.split("/")[-1]
                for id in doubleIDs:
                    double_values = df[((df[0] == id[0]) & (df[1] == id[1]))]
                    for row in double_values.values:
                        error_message += "\n%s" % row
                raise Exception(error_message)

                # error_message += "<br> <!exc> "
        # if error_message != "":
        #     raise Exception(error_message)

        print("Files are ok!")
        arguments = []

        for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):
            arguments.append({"metricObject": MOTMetrics(seq), "args": {
                "gtDataDir": os.path.join(self.datadir, seq),
                "sequence": str(seq),
                "pred_file": res,
                "gt_file": gt,
                "benchmark_name": self.benchmark_name}})
        # try:
        if self.MULTIPROCESSING:
            p = mp.Pool(self.NR_CORES)
            print("Evaluating on {} cpu cores".format(self.NR_CORES))
            processes = [p.apply_async(run_metrics, kwds=inp) for inp in arguments]
            self.results = {inp["args"]['gt_file']: p.get() for inp, p in zip(arguments, processes)}
            p.close()
            p.join()

        else:
            self.results = {inp["args"]['gt_file']: run_metrics(**inp) for inp in arguments}

        # self.failed = False
        # except:
        #     self.failed = True
        #     raise Exception("<exc> MATLAB evalutation failed <!exc>")
        self.overall_results = MOTMetrics("OVERALL")

    def accumulate_df(self, type=None):
        """ create accumulated dataframe with all sequences """
        for k, seq in enumerate(self.results):
            res = self.results[seq]

            res.to_dataframe(display_name=True, type=type)
            if k == 0:
                summary = res.df
            else:
                summary = summary.append(res.df)
        summary = summary.sort_index()

        self.overall_results.to_dataframe(display_name=True, type=type)

        self.summary = summary.append(self.overall_results.df)

    def render_summary(self, buf=None):
        """Render metrics summary to console friendly tabular output.

        Params
        ------
        summary : pd.DataFrame
            Dataframe containing summaries in rows.

        Kwargs
        ------
        buf : StringIO-like, optional
            Buffer to write to
        formatters : dict, optional
            Dicionary defining custom formatters for individual metrics.
            I.e `{'mota': '{:.2%}'.format}`. You can get preset formatters
            from MetricsHost.formatters
        namemap : dict, optional
            Dictionary defining new metric names for display. I.e
            `{'num_false_positives': 'FP'}`.

        Returns
        -------
        string
            Formatted string
        """

        output = self.summary.to_string(
            buf=buf,
            formatters=self.overall_results.formatters,
            justify="left"
        )

        return output


def run_metrics(metricObject, args):
    """ Runs metric for individual sequences

    :param MOTMetrics metricObject: metricObject that has computer_compute_metrics_per_sequence function
    :param dict args: dictionary with args for evaluation function
    :return:
    """

    metricObject.compute_metrics_per_sequence(**args)
    return metricObject

# if __name__ == "__main__":
#     Evaluator()
