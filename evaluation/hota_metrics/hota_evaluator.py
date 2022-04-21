from multiprocessing.pool import Pool
from functools import partial
import time

from evaluation.hota_metrics.hota_metrics.metrics import Count
from evaluation.hota_metrics import hota_metrics as hm  # noqa: E402
from evaluation.hota_metrics.hota_metrics import utils


def eval_sequence(files, class_list, metrics_list, metric_names):
    dataset = hm.datasets.MotChallenge2DBox
    """Function for evaluating a single sequence"""
    gt_file, tracked_file, num_timesteps = files
    raw_data = dataset.get_raw_seq_data_mdp(gt_file, tracked_file, num_timesteps)
    seq_res = {}
    for cls in class_list:
        seq_res[cls] = {}
        data = dataset.get_preprocessed_seq_data(raw_data, cls)
        for metric, met_name in zip(metrics_list, metric_names):
            seq_res[cls][met_name] = metric.eval_sequence(data)
    return seq_res


class HOTA_evaluator:

    def __init__(self):

        self.metrics_list = [hm.metrics.HOTA(), hm.metrics.CLEAR(), hm.metrics.Identity()]

    def metric_names(self):
        devkit_metric_names = [
            'MOTA',
            'MTR',
            'id_switches',
            'IDF1',
            'n_gt_trajectories',
            'n_tr_trajectories',
            'n_gt',
            'n_tr',
            'total_num_frames',
            'FM',
            'MLR',
            'F1',
            'IDP',
            'IDR',
            'IDTP',
            'IDFN',
            'IDFP',
            'MOTAL',
            'recall',
            'precision',
            'MT',
            'PT',
            'ML',
            'PTR',
            'FAR',
            'tp',
            'fp',
            'fn',
            'id_switches_rel',
            'fragments_rel',
            'MOTP',

        ]
        hota_mot_metric_names = [
            'MOTA',
            'MTR',
            'IDSW',
            'IDF1',
            'GT_IDs',
            'IDs',
            'GT_Dets',
            'Dets',
            'total_num_frames',
            'Frag',
            'MLR',
            'CLR_F1',
            'IDP',
            'IDR',
            'IDTP',
            'IDFN',
            'IDFP',
            'MOTAL',
            'CLR_Re',
            'CLR_Pr',
            'MT',
            'PT',
            'ML',
            'PTR',
            'FAR',
            'CLR_TP',
            'CLR_FP',
            'CLR_FN',
            'IDSR',
            'FMR',
            'MOTP',

        ]

        assert len(devkit_metric_names) == len(hota_mot_metric_names), \
            "mismatch between devkit_metric_names and hota_metric_names"

        hota_metric_names = [
            'HOTA',
            'DetA',
            'AssA',
            'DetRe',
            'DetPr',
            'AssRe',
            'AssPr',
            'LocA',
            'RHOTA',
            'HOTA(0)',
            'LocA(0)',
            'HOTALocA(0)',

        ]

        combined_metric_names = [
            'HOTA',
            'DetA',
            'AssA',

            'MOTA',
            'MTR',
            'IDSW',
            'IDF1',
            'GT_IDs',
            'IDs',
            'GT_Dets',
            'Dets',
            'total_num_frames',

            'DetRe',
            'DetPr',
            'AssRe',
            'AssPr',
            'LocA',
            'RHOTA',
            'HOTA(0)',
            'LocA(0)',
            'HOTALocA(0)',

            'Frag',
            'MLR',
            'CLR_F1',
            'IDP',
            'IDR',
            'IDTP',
            'IDFN',
            'IDFP',
            'MOTAL',
            'CLR_Re',
            'CLR_Pr',
            'MT',
            'PT',
            'ML',
            'PTR',
            'FAR',
            'CLR_TP',
            'CLR_FP',
            'CLR_FN',
            'IDSR',
            'FMR',
            'MOTP',
        ]

        return devkit_metric_names, hota_mot_metric_names, hota_metric_names, combined_metric_names

    def run(self, gtfiles, tsfiles, datadir, sequences, benchmark_name, num_timesteps,
            output_fol, config=None, tracker_display_name='mdp', class_list=None, should_classes_combine=0):

        if config is None:
            config = hm.Evaluator.get_default_eval_config()

        if class_list is None:
            class_list = ['pedestrian', ]

        """Evaluate a set of metrics on a set of datasets"""
        metrics_list = self.metrics_list + [Count()]  # Count metrics are always run
        metric_names = utils.validate_metrics_list(metrics_list)

        n_gt_files, n_ts_files = len(gtfiles), len(tsfiles)

        assert n_gt_files == n_ts_files, "mismatch between n_gt_files and n_ts_files"

        # Evaluate each sequence in parallel or in series.
        # returns a nested dict (res), indexed like: res[seq][class][metric_name][sub_metric field]
        # e.g. res[seq_0001][pedestrian][hota][DetA]

        seq_files = tuple(zip(gtfiles, tsfiles, num_timesteps))

        time_start = time.time()
        if config['USE_PARALLEL']:
            n_cores = config['NUM_PARALLEL_CORES']
            print('running in parallel over {} cores'.format(n_cores))
            with Pool(n_cores) as pool:
                _eval_sequence = partial(eval_sequence, class_list=class_list,
                                         metrics_list=metrics_list, metric_names=metric_names)
                results = pool.map(_eval_sequence, seq_files)
                res = dict(zip(gtfiles, results))
        else:
            res = {}
            for seq_file in seq_files:
                res[seq_file[0]] = eval_sequence(seq_file, class_list=class_list,
                                                 metrics_list=metrics_list, metric_names=metric_names)

        # Combine results over all sequences and then over all classes
        res['COMBINED_SEQ'] = {}
        for c_cls in class_list:
            res['COMBINED_SEQ'][c_cls] = {}
            for metric, metric_name in zip(metrics_list, metric_names):
                curr_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in res.items() if
                            seq_key is not 'COMBINED_SEQ'}
                res['COMBINED_SEQ'][c_cls][metric_name] = metric.combine_sequences(curr_res)
        if should_classes_combine:
            for metric, metric_name in zip(metrics_list, metric_names):
                cls_res = {cls_key: cls_value[metric_name] for cls_key, cls_value in
                           res['COMBINED_SEQ'].items()}
                res['COMBINED_SEQ']['COMBINED_CLS'] = metric.combine_classes(cls_res)

        summaries = []
        summaries_dict = {}
        for c_cls in res['COMBINED_SEQ'].keys():  # class_list + 'COMBINED_CLS' if calculated
            summaries = []
            details = []
            summaries_dict = {}
            num_dets = res['COMBINED_SEQ'][c_cls]['Count']['Dets']
            if config['OUTPUT_EMPTY_CLASSES'] or num_dets > 0:
                for metric, metric_name in zip(metrics_list, metric_names):
                    table_res = {seq_key: seq_value[c_cls][metric_name] for seq_key, seq_value in
                                 res.items()}

                    _summaries_dict = {}
                    for seq in res:
                        _summaries_dict[seq] = metric.summary_results(table_res, seq)

                    summaries_dict[metric_name] = _summaries_dict

                    if config['PRINT_RESULTS'] and config['PRINT_ONLY_COMBINED']:
                        metric.print_table({'COMBINED_SEQ': table_res['COMBINED_SEQ']},
                                           tracker_display_name, c_cls)
                    elif config['PRINT_RESULTS']:
                        metric.print_table(table_res, tracker_display_name, c_cls)
                    if config['OUTPUT_SUMMARY']:
                        summaries.append(metric.summary_results(table_res))
                    if config['OUTPUT_DETAILED']:
                        details.append(metric.detailed_results(table_res))
                    if config['PLOT_CURVES']:
                        metric.plot_single_tracker_results(table_res, tracker_display_name, c_cls,
                                                           output_fol)
                if config['OUTPUT_SUMMARY']:
                    utils.write_summary_results(summaries, c_cls, output_fol)
                if config['OUTPUT_DETAILED']:
                    utils.write_detailed_results(details, c_cls, output_fol)

        # Output for returning from function
        output_res = res
        output_msg = 'Success'

        cmb_res_dict_all = {}

        for seq in output_res:
            res_dict = output_res[seq]['pedestrian']
            cmb_res_dict = res_dict['CLEAR'].copy()
            cmb_count_dict = res_dict['Count']
            cmb_id_dict = res_dict['Identity']

            for k in cmb_count_dict:
                cmb_res_dict[k] = cmb_count_dict[k]

            for k in cmb_id_dict:
                cmb_res_dict[k] = cmb_id_dict[k]

            percent_metrics = [
                'CLR_F1',
                'IDF1',
                'IDP',
                'IDR',
                'MOTAL',
                'CLR_Re',
                'CLR_Pr',
                'PTR',
                'MOTP',
                'MOTA',
                'MTR',
                'MLR',
            ]
            # for _i in range(1, 4):
            #     percent_metrics += list(set(metrics_list[_i].float_fields) & set(summaries[_i].keys()))
            #
            for h in percent_metrics:
                cmb_res_dict[h] *= 100

            cmb_res_dict['IDSR'] = cmb_res_dict['IDSW'] / cmb_res_dict['CLR_Re']
            cmb_res_dict['FMR'] = cmb_res_dict['Frag'] / cmb_res_dict['CLR_Re']
            cmb_res_dict['FAR'] = cmb_res_dict['FP_per_frame']
            cmb_res_dict['total_num_frames'] = cmb_res_dict['CLR_Frames']

            cmb_res_dict_all[seq] = cmb_res_dict

        return output_res, output_msg, cmb_res_dict_all, summaries_dict
