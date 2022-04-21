import sys, os

sys.path.append(os.path.abspath(os.getcwd()))

from evaluation.devkit.Evaluator import MotEvaluator, run_metrics


from os import path
import numpy as np

#
# class MOT_evaluator(Evaluator):
#     def __init__(self):
#         Evaluator.__init__(self)


def get_file_lists(benchmark_name=None, gt_dir=None, res_dir=None, save_pkl=None, eval_mode="train",
                   seqmaps_dir="seqmaps"):
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

    benchmark_gt_dir = gt_dir
    seq_file = "{}-{}.txt".format(benchmark_name, eval_mode)

    res_dir = res_dir
    benchmark_name = benchmark_name
    seqmaps_dir = seqmaps_dir

    mode = eval_mode

    datadir = os.path.join(gt_dir, mode)

    # getting names of sequences to evaluate
    assert mode in ["train", "test", "all"], "mode: %s not valid " % mode

    print("Evaluating Benchmark: %s" % benchmark_name)

    # ======================================================
    # Handle evaluation
    # ======================================================

    # load list of all sequences
    seq_path = os.path.join(seqmaps_dir, seq_file)
    seq_path = os.path.abspath(seq_path)
    sequences = np.genfromtxt(seq_path, dtype='str', skip_header=True)

    gtfiles = []
    tsfiles = []
    for seq in sequences:
        # gtf = os.path.join(benchmark_gt_dir, mode, seq, 'gt/gt.txt')
        gtf = os.path.join(benchmark_gt_dir, '{}.txt'.format(seq))
        if path.exists(gtf):
            gtfiles.append(gtf)
        else:
            raise Exception("Ground Truth %s missing" % gtf)
        tsf = os.path.join(res_dir, "%s.txt" % seq)
        if path.exists(gtf):
            tsfiles.append(tsf)
        else:
            raise Exception("Result file %s missing" % tsf)

    return gtfiles, tsfiles, datadir, sequences


def main():
    eval = MotEvaluator()

    benchmark_name = "2D_MOT_2015"
    # gt_dir = "C:/Datasets/MOT2015/2DMOT2015Labels"
    gt_dir = "C:/Datasets/MOT2015/Annotations"
    res_root_dir = 'C:/UofA/PhD/Code/deep_mdp/tracking_module/log'
    res_dir = "no_ibt_mot15_0_10_100_100/lk_wrapper_tmpls2_svm_min10_active_pt_svm/MOT15_0_10_100_100/max_lost0_trd34"
    res_dir = os.path.join(res_root_dir, res_dir)

    eval_mode = "train"
    seqmaps_dir = "../seqmaps"
    gtfiles, tsfiles, datadir, sequences = get_file_lists(
        benchmark_name=benchmark_name,
        gt_dir=gt_dir,
        res_dir=res_dir,
        seqmaps_dir=seqmaps_dir,
        eval_mode=eval_mode)

    eval.run(gtfiles, tsfiles, datadir, sequences, benchmark_name)


if __name__ == "__main__":
    main()
