<!-- MarkdownTOC -->

- [preproc](#preproc_)
    - [detrac       @ preproc](#detrac___prepro_c_)
    - [MOT2015       @ preproc](#mot2015___prepro_c_)
    - [MOT2017       @ preproc](#mot2017___prepro_c_)
- [IBT       @ detrac_0_1_5_0](#ibt___detrac_0_1_5_0_)
    - [lost       @ IBT](#lost___ib_t_)
        - [detrac_0_9_40_60       @ lost/IBT](#detrac_0_9_40_60___lost_ibt_)
            - [iter_0       @ detrac_0_9_40_60/lost/IBT](#iter_0___detrac_0_9_40_60_lost_ib_t_)
        - [detrac_0_9_100_0       @ lost/IBT](#detrac_0_9_100_0___lost_ibt_)
            - [test_30_49_100       @ detrac_0_9_100_0/lost/IBT](#test_30_49_100___detrac_0_9_100_0_lost_ib_t_)
                - [iter_0       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#iter_0___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
                - [templ_1       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#templ_1___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
                - [templ_2       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#templ_2___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
                - [templ_3       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#templ_3___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
                - [templ_4       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#templ_4___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
                - [templ_5       @ test_30_49_100/detrac_0_9_100_0/lost/IBT](#templ_5___test_30_49_100_detrac_0_9_100_0_lost_ibt_)
        - [detrac_0_19_100_0       @ lost/IBT](#detrac_0_19_100_0___lost_ibt_)
            - [test_30_49_100       @ detrac_0_19_100_0/lost/IBT](#test_30_49_100___detrac_0_19_100_0_lost_ibt_)
                - [iter_0       @ test_30_49_100/detrac_0_19_100_0/lost/IBT](#iter_0___test_30_49_100_detrac_0_19_100_0_lost_ib_t_)
            - [debug       @ detrac_0_19_100_0/lost/IBT](#debug___detrac_0_19_100_0_lost_ibt_)
        - [detrac_0_59_40_60       @ lost/IBT](#detrac_0_59_40_60___lost_ibt_)
            - [iter_0       @ detrac_0_59_40_60/lost/IBT](#iter_0___detrac_0_59_40_60_lost_ibt_)
        - [detrac:train_0_59_test_60_99       @ lost/IBT](#detrac_train_0_59_test_60_99___lost_ibt_)
                - [tmpls2       @ detrac:train_0_59_test_60_99/lost/IBT](#tmpls2___detrac_train_0_59_test_60_99_lost_ib_t_)
        - [mot15:train_0_10_test_11_21       @ lost/IBT](#mot15_train_0_10_test_11_21___lost_ibt_)
                - [tmpls2       @ mot15:train_0_10_test_11_21/lost/IBT](#tmpls2___mot15_train_0_10_test_11_21_lost_ibt_)
        - [mot17:train_0_6_test_7_13       @ lost/IBT](#mot17_train_0_6_test_7_13___lost_ibt_)
                - [tmpls2       @ mot17:train_0_6_test_7_13/lost/IBT](#tmpls2___mot17_train_0_6_test_7_13_lost_ibt_)
- [LK](#l_k_)
    - [detrac_0_to_9_40_60       @ LK](#detrac_0_to_9_40_60___lk_)
        - [test_1       @ detrac_0_to_9_40_60/LK](#test_1___detrac_0_to_9_40_60_lk_)
        - [5_5       @ detrac_0_to_9_40_60/LK](#5_5___detrac_0_to_9_40_60_lk_)
    - [detrac_0_59_40_60       @ LK](#detrac_0_59_40_60___lk_)
    - [detrac_0_59_100_0       @ LK](#detrac_0_59_100_0___lk_)
        - [lost       @ detrac_0_59_100_0/LK](#lost___detrac_0_59_100_0_lk_)
            - [continue       @ lost/detrac_0_59_100_0/LK](#continue___lost_detrac_0_59_100_0_l_k_)
        - [lost_1       @ detrac_0_59_100_0/LK](#lost_1___detrac_0_59_100_0_lk_)
            - [24_48_64_128_64_48_24       @ lost_1/detrac_0_59_100_0/LK](#24_48_64_128_64_48_24___lost_1_detrac_0_59_100_0_l_k_)
                - [continue       @ 24_48_64_128_64_48_24/lost_1/detrac_0_59_100_0/LK](#continue___24_48_64_128_64_48_24_lost_1_detrac_0_59_100_0_l_k_)
    - [detrac_0_59_40_0       @ LK](#detrac_0_59_40_0___lk_)
        - [lost       @ detrac_0_59_40_0/LK](#lost___detrac_0_59_40_0_l_k_)
            - [continue_n60       @ lost/detrac_0_59_40_0/LK](#continue_n60___lost_detrac_0_59_40_0_lk_)
    - [detrac_0_59_n60_0       @ LK](#detrac_0_59_n60_0___lk_)
        - [lost       @ detrac_0_59_n60_0/LK](#lost___detrac_0_59_n60_0_lk_)
            - [continue       @ lost/detrac_0_59_n60_0/LK](#continue___lost_detrac_0_59_n60_0_l_k_)
    - [detrac:train_0_4:test_5_9:yolo       @ LK](#detrac_train_0_4_test_5_9_yolo___lk_)
        - [tmpls2:min10       @ detrac:train_0_4:test_5_9:yolo/LK](#tmpls2_min10___detrac_train_0_4_test_5_9_yolo_l_k_)
        - [0       @ detrac:train_0_4:test_5_9:yolo/LK](#0___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 0/detrac:train_0_4:test_5_9:yolo/LK](#test_self___0_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/0/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_0_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 0/detrac:train_0_4:test_5_9:yolo/LK](#test_203___0_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [2       @ detrac:train_0_4:test_5_9:yolo/LK](#2___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 2/detrac:train_0_4:test_5_9:yolo/LK](#test_self___2_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/2/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_2_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 2/detrac:train_0_4:test_5_9:yolo/LK](#test_203___2_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [4       @ detrac:train_0_4:test_5_9:yolo/LK](#4___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 4/detrac:train_0_4:test_5_9:yolo/LK](#test_self___4_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/4/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_4_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 4/detrac:train_0_4:test_5_9:yolo/LK](#test_203___4_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [14       @ detrac:train_0_4:test_5_9:yolo/LK](#14___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 14/detrac:train_0_4:test_5_9:yolo/LK](#test_self___14_detrac_train_0_4_test_5_9_yolo_lk_)
                - [on_train       @ test:self/14/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_14_detrac_train_0_4_test_5_9_yolo_lk_)
            - [test:203       @ 14/detrac:train_0_4:test_5_9:yolo/LK](#test_203___14_detrac_train_0_4_test_5_9_yolo_lk_)
        - [203       @ detrac:train_0_4:test_5_9:yolo/LK](#203___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_self___203_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/203/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:0       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_0___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:2       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_2___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:4       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_4___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:14       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_14___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:475       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_475___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:656       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_656___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:777       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_777___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:863       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_863___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:930       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_930___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:963       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_963___203_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:983       @ 203/detrac:train_0_4:test_5_9:yolo/LK](#test_983___203_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [475       @ detrac:train_0_4:test_5_9:yolo/LK](#475___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 475/detrac:train_0_4:test_5_9:yolo/LK](#test_self___475_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/475/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_475_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 475/detrac:train_0_4:test_5_9:yolo/LK](#test_203___475_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [656       @ detrac:train_0_4:test_5_9:yolo/LK](#656___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 656/detrac:train_0_4:test_5_9:yolo/LK](#test_self___656_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/656/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_656_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 656/detrac:train_0_4:test_5_9:yolo/LK](#test_203___656_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [777       @ detrac:train_0_4:test_5_9:yolo/LK](#777___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 777/detrac:train_0_4:test_5_9:yolo/LK](#test_self___777_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/777/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_777_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 777/detrac:train_0_4:test_5_9:yolo/LK](#test_203___777_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [863       @ detrac:train_0_4:test_5_9:yolo/LK](#863___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 863/detrac:train_0_4:test_5_9:yolo/LK](#test_self___863_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/863/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_863_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 863/detrac:train_0_4:test_5_9:yolo/LK](#test_203___863_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [930       @ detrac:train_0_4:test_5_9:yolo/LK](#930___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 930/detrac:train_0_4:test_5_9:yolo/LK](#test_self___930_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/930/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_930_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 930/detrac:train_0_4:test_5_9:yolo/LK](#test_203___930_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [963       @ detrac:train_0_4:test_5_9:yolo/LK](#963___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 963/detrac:train_0_4:test_5_9:yolo/LK](#test_self___963_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/963/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_963_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 963/detrac:train_0_4:test_5_9:yolo/LK](#test_203___963_detrac_train_0_4_test_5_9_yolo_l_k_)
        - [983       @ detrac:train_0_4:test_5_9:yolo/LK](#983___detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:self       @ 983/detrac:train_0_4:test_5_9:yolo/LK](#test_self___983_detrac_train_0_4_test_5_9_yolo_l_k_)
                - [on_train       @ test:self/983/detrac:train_0_4:test_5_9:yolo/LK](#on_train___test_self_983_detrac_train_0_4_test_5_9_yolo_l_k_)
            - [test:203       @ 983/detrac:train_0_4:test_5_9:yolo/LK](#test_203___983_detrac_train_0_4_test_5_9_yolo_l_k_)
    - [detrac:train_0_9       @ LK](#detrac_train_0_9___lk_)
        - [test_30_49       @ detrac:train_0_9/LK](#test_30_49___detrac_train_0_9_l_k_)
            - [tmpls2       @ test_30_49/detrac:train_0_9/LK](#tmpls2___test_30_49_detrac_train_0_9_lk_)
                - [active_train       @ tmpls2/test_30_49/detrac:train_0_9/LK](#active_train___tmpls2_test_30_49_detrac_train_0_9_l_k_)
                    - [no_gh       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK](#no_gh___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_)
                    - [no_th       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK](#no_th___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_)
                    - [no_reconnect       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK](#no_reconnect___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_)
                    - [no_gh_no_reconnect       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK](#no_gh_no_reconnect___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_)
                    - [no_gh_no_th       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK](#no_gh_no_th___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_)
            - [siamfc       @ test_30_49/detrac:train_0_9/LK](#siamfc___test_30_49_detrac_train_0_9_lk_)
            - [darpn       @ test_30_49/detrac:train_0_9/LK](#darpn___test_30_49_detrac_train_0_9_lk_)
                - [max_lost50       @ darpn/test_30_49/detrac:train_0_9/LK](#max_lost50___darpn_test_30_49_detrac_train_0_9_lk_)
            - [siamx:fc       @ test_30_49/detrac:train_0_9/LK](#siamx_fc___test_30_49_detrac_train_0_9_lk_)
            - [siamx:rpn_nxt       @ test_30_49/detrac:train_0_9/LK](#siamx_rpn_nxt___test_30_49_detrac_train_0_9_lk_)
            - [siamx:rpnpp       @ test_30_49/detrac:train_0_9/LK](#siamx_rpnpp___test_30_49_detrac_train_0_9_lk_)
            - [pyt:atom       @ test_30_49/detrac:train_0_9/LK](#pyt_atom___test_30_49_detrac_train_0_9_lk_)
            - [pyt:dimp       @ test_30_49/detrac:train_0_9/LK](#pyt_dimp___test_30_49_detrac_train_0_9_lk_)
            - [pyt:prdimp       @ test_30_49/detrac:train_0_9/LK](#pyt_prdimp___test_30_49_detrac_train_0_9_lk_)
            - [tmpls5       @ test_30_49/detrac:train_0_9/LK](#tmpls5___test_30_49_detrac_train_0_9_lk_)
        - [test_10_59       @ detrac:train_0_9/LK](#test_10_59___detrac_train_0_9_l_k_)
            - [tmpls2       @ test_10_59/detrac:train_0_9/LK](#tmpls2___test_10_59_detrac_train_0_9_lk_)
                - [active_train       @ tmpls2/test_10_59/detrac:train_0_9/LK](#active_train___tmpls2_test_10_59_detrac_train_0_9_l_k_)
    - [detrac:0_39_30_70       @ LK](#detrac_0_39_30_70___lk_)
        - [tmpls2       @ detrac:0_39_30_70/LK](#tmpls2___detrac_0_39_30_70_lk_)
        - [tmpls5       @ detrac:0_39_30_70/LK](#tmpls5___detrac_0_39_30_70_lk_)
    - [detrac:train_0_59_test_60_99       @ LK](#detrac_train_0_59_test_60_99___lk_)
        - [tmpls2       @ detrac:train_0_59_test_60_99/LK](#tmpls2___detrac_train_0_59_test_60_99_l_k_)
            - [min10       @ tmpls2/detrac:train_0_59_test_60_99/LK](#min10___tmpls2_detrac_train_0_59_test_60_99_lk_)
        - [tmpls5       @ detrac:train_0_59_test_60_99/LK](#tmpls5___detrac_train_0_59_test_60_99_l_k_)
    - [mot15:train_0_10_test_11_21       @ LK](#mot15_train_0_10_test_11_21___lk_)
        - [min10       @ mot15:train_0_10_test_11_21/LK](#min10___mot15_train_0_10_test_11_21_lk_)
    - [mot17:train_0_6_test_7_13       @ LK](#mot17_train_0_6_test_7_13___lk_)
        - [min10       @ mot17:train_0_6_test_7_13/LK](#min10___mot17_train_0_6_test_7_13_lk_)
    - [detrac:0_59:60_99       @ LK](#detrac_0_59_60_99___lk_)
        - [tmpls2       @ detrac:0_59:60_99/LK](#tmpls2___detrac_0_59_60_99_lk_)
            - [active_mlp       @ tmpls2/detrac:0_59:60_99/LK](#active_mlp___tmpls2_detrac_0_59_60_99_l_k_)
            - [on_train       @ tmpls2/detrac:0_59:60_99/LK](#on_train___tmpls2_detrac_0_59_60_99_l_k_)
            - [ctm:darpn       @ tmpls2/detrac:0_59:60_99/LK](#ctm_darpn___tmpls2_detrac_0_59_60_99_l_k_)
                - [ign       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#ign___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                - [tracked_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#tracked_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                    - [ign       @ tracked_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#ign___tracked_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                - [tracked_lost_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#tracked_lost_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                - [all_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#all_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                    - [ign       @ all_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#ign___all_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
                    - [active_tracked_pos       @ all_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK](#active_tracked_pos___all_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_)
            - [ctm:pyt:prdimp:18       @ tmpls2/detrac:0_59:60_99/LK](#ctm_pyt_prdimp_18___tmpls2_detrac_0_59_60_99_l_k_)
            - [ctm:siamx:rpn_nxt       @ tmpls2/detrac:0_59:60_99/LK](#ctm_siamx_rpn_nxt___tmpls2_detrac_0_59_60_99_l_k_)
                - [tracked_pos       @ ctm:siamx:rpn_nxt/tmpls2/detrac:0_59:60_99/LK](#tracked_pos___ctm_siamx_rpn_nxt_tmpls2_detrac_0_59_60_99_l_k_)
                    - [active_tracked_pos       @ tracked_pos/ctm:siamx:rpn_nxt/tmpls2/detrac:0_59:60_99/LK](#active_tracked_pos___tracked_pos_ctm_siamx_rpn_nxt_tmpls2_detrac_0_59_60_99_l_k_)
        - [tmpls5       @ detrac:0_59:60_99/LK](#tmpls5___detrac_0_59_60_99_lk_)
    - [detrac:train:60_99:test:0_59       @ LK](#detrac_train_60_99_test_0_59___lk_)
        - [tmpls2       @ detrac:train:60_99:test:0_59/LK](#tmpls2___detrac_train_60_99_test_0_59_l_k_)
        - [on_train       @ detrac:train:60_99:test:0_59/LK](#on_train___detrac_train_60_99_test_0_59_l_k_)
    - [mot15:0_4:5_10       @ LK](#mot15_0_4_5_10___lk_)
        - [tmpls2       @ mot15:0_4:5_10/LK](#tmpls2___mot15_0_4_5_10_l_k_)
    - [mot15:0_10:11_21       @ LK](#mot15_0_10_11_21___lk_)
        - [tmpls2       @ mot15:0_10:11_21/LK](#tmpls2___mot15_0_10_11_21_l_k_)
            - [on_train       @ tmpls2/mot15:0_10:11_21/LK](#on_train___tmpls2_mot15_0_10_11_21_lk_)
        - [tmpls5       @ mot15:0_10:11_21/LK](#tmpls5___mot15_0_10_11_21_l_k_)
            - [on_train       @ tmpls5/mot15:0_10:11_21/LK](#on_train___tmpls5_mot15_0_10_11_21_lk_)
    - [mot17:train:0_6:test:7_13       @ LK](#mot17_train_0_6_test_7_13___lk__1)
        - [tmpls2       @ mot17:train:0_6:test:7_13/LK](#tmpls2___mot17_train_0_6_test_7_13_lk_)
            - [on_train       @ tmpls2/mot17:train:0_6:test:7_13/LK](#on_train___tmpls2_mot17_train_0_6_test_7_13_l_k_)
        - [tmpls5       @ mot17:train:0_6:test:7_13/LK](#tmpls5___mot17_train_0_6_test_7_13_lk_)
            - [on_train       @ tmpls5/mot17:train:0_6:test:7_13/LK](#on_train___tmpls5_mot17_train_0_6_test_7_13_l_k_)
    - [mot17:sdp:train:0_6:test:7_13       @ LK](#mot17_sdp_train_0_6_test_7_13___lk_)
        - [tmpls2       @ mot17:sdp:train:0_6:test:7_13/LK](#tmpls2___mot17_sdp_train_0_6_test_7_13_lk_)
            - [on_train       @ tmpls2/mot17:sdp:train:0_6:test:7_13/LK](#on_train___tmpls2_mot17_sdp_train_0_6_test_7_13_l_k_)
    - [mot17:dpm:train:0_6:test:7_13       @ LK](#mot17_dpm_train_0_6_test_7_13___lk_)
        - [tmpls2       @ mot17:dpm:train:0_6:test:7_13/LK](#tmpls2___mot17_dpm_train_0_6_test_7_13_lk_)
            - [on_train       @ tmpls2/mot17:dpm:train:0_6:test:7_13/LK](#on_train___tmpls2_mot17_dpm_train_0_6_test_7_13_l_k_)
    - [gram:0_2:0_2:40_n60       @ LK](#gram_0_2_0_2_40_n60___lk_)
        - [tmpls2       @ gram:0_2:0_2:40_n60/LK](#tmpls2___gram_0_2_0_2_40_n60_lk_)
            - [on_train       @ tmpls2/gram:0_2:0_2:40_n60/LK](#on_train___tmpls2_gram_0_2_0_2_40_n60_l_k_)
            - [on_all       @ tmpls2/gram:0_2:0_2:40_n60/LK](#on_all___tmpls2_gram_0_2_0_2_40_n60_l_k_)
    - [mot15:0_10:0_0       @ LK](#mot15_0_10_0_0___lk_)
    - [mot17:train:0_6:test:0_0:100_5       @ LK](#mot17_train_0_6_test_0_0_100_5___lk_)

<!-- /MarkdownTOC -->

<a id="preproc_"></a>
# preproc

<a id="detrac___prepro_c_"></a>
## detrac       @ preproc-->dmdp_svm

python3 xmlDETRACToMOT.py root_dir=G:/Datasets actor_id=6

<a id="mot2015___prepro_c_"></a>
## MOT2015       @ preproc-->dmdp_svm

python3 processMOT.py root_dir=/data/MOT2015 db_dir=2DMOT2015 db_type=train ignore_img=0

python3 processMOT.py root_dir=/data/MOT2015 db_dir=2DMOT2015 db_type=test ignore_img=0


<a id="mot2017___prepro_c_"></a>
## MOT2017       @ preproc-->dmdp_svm

python3 processMOT.py root_dir=/data/MOT2017 db_type=train ignore_img=0

python3 processMOT.py root_dir=/data/MOT2017 db_type=test ignore_img=0

<a id="ibt___detrac_0_1_5_0_"></a>
# IBT       @ detrac_0_1_5_0

<a id="lost___ib_t_"></a>
## lost       @ IBT-->dmdp_svm

<a id="detrac_0_9_40_60___lost_ibt_"></a>
### detrac_0_9_40_60       @ lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_9:d:40_60,svm:active,svm:lost,ibt:acc:lost:detrac_0_9_40_60:lk:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:d:n60_n60 start=12

<a id="iter_0___detrac_0_9_40_60_lost_ib_t_"></a>
#### iter_0       @ detrac_0_9_40_60/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_9:d:40_60,svm:active,svm:lost,ibt:acc:lost:detrac_0_9_40_60:lk:svm:n6:t2 @ibt test_iters=0 test_cfgs=detrac:d:n60_n60 phases=2,3

<a id="detrac_0_9_100_0___lost_ibt_"></a>
### detrac_0_9_100_0       @ lost/IBT-->dmdp_svm

<a id="test_30_49_100___detrac_0_9_100_0_lost_ib_t_"></a>
#### test_30_49_100       @ detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:1,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,ibt:acc:lost:detrac_0_9_100_0:lk:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=12

<a id="iter_0___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### iter_0       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm


python3 main.py cfg=gpu:1,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,ibt:acc:lost:detrac_0_9_100_0:lk:svm:n6:t2 @ibt test_iters=0 test_cfgs=detrac:s:30_49:d:100_100  phases=2,3

<a id="templ_1___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### templ_1       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,templ:1,ibt:acc:lost:detrac_0_9_100_0:lk:templ_1:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0

<a id="templ_2___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### templ_2       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,templ:2,ibt:acc:lost:detrac_0_9_100_0:lk:templ_2:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0

<a id="templ_3___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### templ_3       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,templ:3,ibt:acc:lost:detrac_0_9_100_0:lk:templ_3:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0

<a id="templ_4___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### templ_4       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:1,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,templ:4,ibt:acc:lost:detrac_0_9_100_0:lk:templ_4:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0

<a id="templ_5___test_30_49_100_detrac_0_9_100_0_lost_ibt_"></a>
##### templ_5       @ test_30_49_100/detrac_0_9_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:1,lk,detrac:s:0_9:d:100_0,svm:active,svm:lost,templ:5,ibt:acc:lost:detrac_0_9_100_0:lk:templ_5:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0


<a id="detrac_0_19_100_0___lost_ibt_"></a>
### detrac_0_19_100_0       @ lost/IBT-->dmdp_svm

<a id="test_30_49_100___detrac_0_19_100_0_lost_ibt_"></a>
#### test_30_49_100       @ detrac_0_19_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_19:d:100_0,svm:active,svm:lost,ibt:acc:lost:detrac_0_19_100_0:lk:svm:n6:t2 @ibt test_iters=0,1,3,5 start=0 test_cfgs=detrac:s:30_49:d:100_100

<a id="iter_0___test_30_49_100_detrac_0_19_100_0_lost_ib_t_"></a>
##### iter_0       @ test_30_49_100/detrac_0_19_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_19:d:100_0,svm:active,svm:lost,ibt:acc:lost:detrac_0_19_100_0:lk:svm:n6:t2 @ibt test_iters=0 start=0 test_cfgs=detrac:s:30_49:d:100_100 phases=2,3

<a id="debug___detrac_0_19_100_0_lost_ibt_"></a>
#### debug       @ detrac_0_19_100_0/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:2,lk,detrac:s:0_19:d:100_0,svm:active,svm:lost,ibt:acc:lost:detrac_0_19_100_0:lk:svm:n6:t2,d:lost:v2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:s:30_49:d:100_100 start=0

<a id="detrac_0_59_40_60___lost_ibt_"></a>
### detrac_0_59_40_60       @ lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:1,lk,detrac:s:0_59:d:40_60,svm:active,svm:lost,ibt:acc:lost:detrac_0_59_40_60:lk:svm:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs=detrac:d:n60_n60 start=0 

<a id="iter_0___detrac_0_59_40_60_lost_ibt_"></a>
#### iter_0       @ detrac_0_59_40_60/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:1,lk,detrac:s:0_59:d:40_60,svm:active,svm:lost,ibt:acc:lost:detrac_0_59_40_60:lk:svm:n6:t2 @ibt test_iters=0 test_cfgs=detrac:d:n60_n60 phases=2,3


<a id="detrac_train_0_59_test_60_99___lost_ibt_"></a>
### detrac:train_0_59_test_60_99       @ lost/IBT-->dmdp_svm

<a id="tmpls2___detrac_train_0_59_test_60_99_lost_ib_t_"></a>
##### tmpls2       @ detrac:train_0_59_test_60_99/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_detrac_:strain0_59:stest60_99:d-100_100,_svm_:active,_svm_:lost,_ibt_:acc:n2:t2:tmpls2:lost:detrac:s0_59:d-100_100:lk:svm:tmpls2 @ibt test_iters=1,2 test_cfgs+=0:_detrac_:s:60_99:d:100_100,_test_:max_lost0:active_train start=00


<a id="mot15_train_0_10_test_11_21___lost_ibt_"></a>
### mot15:train_0_10_test_11_21       @ lost/IBT-->dmdp_svm

<a id="tmpls2___mot15_train_0_10_test_11_21_lost_ibt_"></a>
##### tmpls2       @ mot15:train_0_10_test_11_21/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2,_mot15_:strain0_10:stest11_21:d-100_100,_svm_:active,_svm_:lost,_ibt_:acc:n2:t2:tmpls2:lost:mot15:s0_10:d-100_100:lk:svm:tmpls2 @ibt test_iters=1,2 test_cfgs+=0:_mot15_:s:11_21:d:100_100,_test_:max_lost0 start=00

<a id="mot17_train_0_6_test_7_13___lost_ibt_"></a>
### mot17:train_0_6_test_7_13       @ lost/IBT-->dmdp_svm

<a id="tmpls2___mot17_train_0_6_test_7_13_lost_ibt_"></a>
##### tmpls2       @ mot17:train_0_6_test_7_13/lost/IBT-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_mot17_:strain0_6:stest7_13:d-100_100,_svm_:active,_svm_:lost,_ibt_:acc:n2:t2:tmpls2:lost:mot17:s0_6:d-100_100:lk:svm:tmpls2 @ibt test_iters=1,2 test_cfgs+=0:_mot17_:s:7_13:d:100_100,_test_:max_lost0 start=00


<a id="l_k_"></a>
# LK

<a id="detrac_0_to_9_40_60___lk_"></a>
## detrac_0_to_9_40_60       @ LK-->dmdp_svm

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(10)" results_dir=log/detrac_0to9_40_60_lk @test seq_ids="range(10)" @data ratios.detrac="(0.4,0)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10 @train load=0

<a id="test_1___detrac_0_to_9_40_60_lk_"></a>
### test_1       @ detrac_0_to_9_40_60/LK-->dmdp_svm

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(10)" results_dir=log/detrac_0to9_40_60_lk @test seq_ids="range(10)" @data ratios.detrac="(0.4,-0.01)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10 @train load=1

<a id="5_5___detrac_0_to_9_40_60_lk_"></a>
### 5_5       @ detrac_0_to_9_40_60/LK-->dmdp_svm

CUDA_VISIBLE_DEVICES=1 python3 main.py @train seq_ids="range(10)" @test seq_ids="range(10)" @data ratios.detrac="(0.05,0.05)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=2

<a id="detrac_0_59_40_60___lk_"></a>
## detrac_0_59_40_60       @ LK-->dmdp_svm

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(60)" results_dir=log/detrac_0to59_40_60_lk @test seq_ids="range(60)" @data ratios.detrac="(0.4,0)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10


<a id="detrac_0_59_100_0___lk_"></a>
## detrac_0_59_100_0       @ LK-->dmdp_svm

<a id="lost___detrac_0_59_100_0_lk_"></a>
### lost       @ detrac_0_59_100_0/LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,svm:lost,svm:active,detrac:s:0_59::d:100_0,svm_batch_train:lost

<a id="continue___lost_detrac_0_59_100_0_l_k_"></a>
#### continue       @ lost/detrac_0_59_100_0/LK-->dmdp_svm

python3 main.py cfg=gpu:2,lk,svm:lost,svm:active,detrac:s:0_59::d:100_0,svm_continue:async:detrac_0_59_100_0

<a id="lost_1___detrac_0_59_100_0_lk_"></a>
### lost_1       @ detrac_0_59_100_0/LK-->dmdp_svm

<a id="24_48_64_128_64_48_24___lost_1_detrac_0_59_100_0_l_k_"></a>
#### 24_48_64_128_64_48_24       @ lost_1/detrac_0_59_100_0/LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,detrac:s:0_59::d:100_0,batch_train:lost_1:24_48_64_128_64_48_24_bn,mlp:lost:24_48_64_128_64_48_24:bn

<a id="continue___24_48_64_128_64_48_24_lost_1_detrac_0_59_100_0_l_k_"></a>
##### continue       @ 24_48_64_128_64_48_24/lost_1/detrac_0_59_100_0/LK-->dmdp_svm

python3 main.py cfg=gpu:2,detrac:s:0_59::d:100_0,mlp:lost:24_48_64_128_64_48_24:bn,mlp:active:24_48_64_128_64_48_24:bn,continue:batch_1

<a id="detrac_0_59_40_0___lk_"></a>
## detrac_0_59_40_0       @ LK-->dmdp_svm

<a id="lost___detrac_0_59_40_0_l_k_"></a>
### lost       @ detrac_0_59_40_0/LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,svm:lost,svm:active,detrac:s:0_59::d:40_0,svm_batch_train:lost:detrac_0_59_40_0

<a id="continue_n60___lost_detrac_0_59_40_0_lk_"></a>
#### continue_n60       @ lost/detrac_0_59_40_0/LK-->dmdp_svm

python3 main.py cfg=gpu:2,lk,svm:lost,svm:active,detrac:s:0_59::d:n60_0,svm_continue:async:detrac_0_59_100_0


<a id="detrac_0_59_n60_0___lk_"></a>
## detrac_0_59_n60_0       @ LK-->dmdp_svm

<a id="lost___detrac_0_59_n60_0_lk_"></a>
### lost       @ detrac_0_59_n60_0/LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,svm:lost,svm:active,detrac:s:0_59::d:n60_0,svm_batch_train:lost

<a id="continue___lost_detrac_0_59_n60_0_l_k_"></a>
#### continue       @ lost/detrac_0_59_n60_0/LK-->dmdp_svm

python3 main.py cfg=gpu:2,lk,svm:lost,svm:active,detrac:s:0_59::d:n60_0,svm_continue:async:detrac_0_59_100_0

<a id="detrac_train_0_4_test_5_9_yolo___lk_"></a>
## detrac:train_0_4:test_5_9:yolo       @ LK-->dmdp_svm

<a id="tmpls2_min10___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### tmpls2:min10       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="0___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 0       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___0_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 0/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-0 @test load=0 evaluate=1 @train load=1

__dbg__

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_6:d-100_3,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-0 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_0_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/0/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:1,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-0 @test load=0 evaluate=1 @train load=1


<a id="test_203___0_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 0/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-0:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1


<a id="2___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 2       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___2_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 2/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-2 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_2_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/2/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:1,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-2 @test load=0 evaluate=1 @train load=1

<a id="test_203___2_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 2/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-2:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1


<a id="4___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 4       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___4_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 4/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-4 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_4_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/4/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-4 @test load=0 evaluate=1 @train load=1

<a id="test_203___4_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 4/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-4:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1


<a id="14___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 14       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___14_detrac_train_0_4_test_5_9_yolo_lk_"></a>
#### test:self       @ 14/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-14 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_14_detrac_train_0_4_test_5_9_yolo_lk_"></a>
##### on_train       @ test:self/14/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-14 @test load=0 evaluate=1 @train load=1

<a id="test_203___14_detrac_train_0_4_test_5_9_yolo_lk_"></a>
#### test:203       @ 14/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-14:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1


<a id="203___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 203       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="test_0___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:0       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-0 @test load=0 evaluate=1 @train load=1

<a id="test_2___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:2       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-2 @test load=0 evaluate=1 @train load=1

<a id="test_4___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:4       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-4 @test load=0 evaluate=1 @train load=1

<a id="test_14___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:14       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-14 @test load=0 evaluate=1 @train load=1

<a id="test_475___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:475       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-475 @test load=0 evaluate=1 @train load=1


<a id="test_656___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:656       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-656 @test load=0 evaluate=1 @train load=1

<a id="test_777___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:777       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-777 @test load=0 evaluate=1 @train load=1

<a id="test_863___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:863       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-863 @test load=0 evaluate=1 @train load=1

<a id="test_930___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:930       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-930 @test load=0 evaluate=1 @train load=1

<a id="test_963___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:963       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-963 @test load=0 evaluate=1 @train load=1

<a id="test_983___203_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:983       @ 203/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-203:++det_test:yolo-983 @test load=0 evaluate=1 @train load=1

<a id="475___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 475       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm

<a id="test_self___475_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 475/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-475 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_475_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/475/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-475 @test load=0 evaluate=1 @train load=1

<a id="test_203___475_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 475/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_4,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-475:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

__dbg__

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_6:d-100_2,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-475:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1


<a id="656___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 656       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___656_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 656/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-656 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_656_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/656/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-656 @test load=0 evaluate=1 @train load=1

<a id="test_203___656_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 656/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-656:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="777___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 777       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___777_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 777/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-777 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_777_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/777/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-777 @test load=0 evaluate=1 @train load=1

<a id="test_203___777_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 777/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-777:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="863___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 863       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___863_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 863/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-863 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_863_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/863/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-863 @test load=0 evaluate=1 @train load=1

<a id="test_203___863_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 863/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-863:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="930___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 930       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___930_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 930/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-930 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_930_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/930/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-930 @test load=0 evaluate=1 @train load=1

<a id="test_203___930_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 930/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-930:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="963___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 963       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___963_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 963/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-963 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_963_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/963/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-963 @test load=0 evaluate=1 @train load=1

<a id="test_203___963_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 963/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-963:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="983___detrac_train_0_4_test_5_9_yolo_l_k_"></a>
### 983       @ detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
<a id="test_self___983_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:self       @ 983/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-983 @test load=0 evaluate=1 @train load=1

<a id="on_train___test_self_983_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
##### on_train       @ test:self/983/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train+det_test:yolo-983 @test load=0 evaluate=1 @train load=1

<a id="test_203___983_detrac_train_0_4_test_5_9_yolo_l_k_"></a>
#### test:203       @ 983/detrac:train_0_4:test_5_9:yolo/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:det_train:yolo-983:++det_test:yolo-203 @test load=0 evaluate=1 @train load=1

<a id="detrac_train_0_9___lk_"></a>
## detrac:train_0_9       @ LK-->dmdp_svm

<a id="test_30_49___detrac_train_0_9_l_k_"></a>
### test_30_49       @ detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm train.load=1

__debug__       @ test_30_49/detrac:train_0_9/LK

python3 main.py cfg=gpu:0,lk,_detrac_:strain0_0:stest30_30:d:10_10,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_0:d10_10:lk:svm

<a id="tmpls2___test_30_49_detrac_train_0_9_lk_"></a>
#### tmpls2       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 @test load=1 evaluate=1

<a id="active_train___tmpls2_test_30_49_detrac_train_0_9_l_k_"></a>
##### active_train       @ tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1

<a id="no_gh___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_"></a>
###### no_gh       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train:no_gh:sort:conflict:filter,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1

<a id="no_th___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_"></a>
###### no_th       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train:no_th:lost:tracked:reconnect,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1

<a id="no_reconnect___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_"></a>
###### no_reconnect       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train:no_th:reconnect,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1

<a id="no_gh_no_reconnect___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_"></a>
###### no_gh_no_reconnect       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train:no_gh:sort:conflict:filter:no_th:reconnect,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1

<a id="no_gh_no_th___active_train_tmpls2_test_30_49_detrac_train_0_9_lk_"></a>
###### no_gh_no_th       @ active_train/tmpls2/test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train:no_gh:sort:conflict:filter:no_th:lost:tracked:reconnect,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 train.load=1


<a id="siamfc___test_30_49_detrac_train_0_9_lk_"></a>
#### siamfc       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_siamfc_:tracked:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:siamfc

<a id="darpn___test_30_49_detrac_train_0_9_lk_"></a>
#### darpn       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:darpn train.load=1 test.start=13

<a id="max_lost50___darpn_test_30_49_detrac_train_0_9_lk_"></a>
##### max_lost50       @ darpn/test_30_49/detrac:train_0_9/LK-->dmdp_svm
python3 main.py cfg=gpu:1,_lk_:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost50,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:darpn train.load=1 test.start=0

<a id="siamx_fc___test_30_49_detrac_train_0_9_lk_"></a>
#### siamx:fc       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_siamx_:tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:siamx:fc 

<a id="siamx_rpn_nxt___test_30_49_detrac_train_0_9_lk_"></a>
#### siamx:rpn_nxt       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:siamx:rpn_nxt train.load=1 test.start=11

<a id="siamx_rpnpp___test_30_49_detrac_train_0_9_lk_"></a>
#### siamx:rpnpp       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2,_siamx_:tracked:rpnpp:nms0:f0,_detrac_:strain0_9:stest30_49:d:10_10,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:siamx:rpnpp 

<a id="pyt_atom___test_30_49_detrac_train_0_9_lk_"></a>
#### pyt:atom       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2,_pyt_:tracked:atom:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:pyt:atom train.load=1

<a id="pyt_dimp___test_30_49_detrac_train_0_9_lk_"></a>
#### pyt:dimp       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_pyt_:tracked:dimp:18:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:pyt:dimp:18 

<a id="pyt_prdimp___test_30_49_detrac_train_0_9_lk_"></a>
#### pyt:prdimp       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_pyt_:tracked:prdimp:18:nms0:f0,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2:tracked:pyt:prdimp:18 

<a id="tmpls5___test_30_49_detrac_train_0_9_lk_"></a>
#### tmpls5       @ test_30_49/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls5,_detrac_:strain0_9:stest30_49:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls5 @test load=1 evaluate=1


<a id="test_10_59___detrac_train_0_9_l_k_"></a>
### test_10_59       @ detrac:train_0_9/LK-->dmdp_svm

<a id="tmpls2___test_10_59_detrac_train_0_9_lk_"></a>
#### tmpls2       @ test_10_59/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2,_detrac_:strain0_9:stest10_59:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 @train load=1

<a id="active_train___tmpls2_test_10_59_detrac_train_0_9_l_k_"></a>
##### active_train       @ tmpls2/test_10_59/detrac:train_0_9/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:strain0_9:stest10_59:d:100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train,_train_:detrac:s0_9:d-100_100:lk:svm:tmpls2 @train load=1

<a id="detrac_0_39_30_70___lk_"></a>
## detrac:0_39_30_70       @ LK-->dmdp_svm

python3 main.py cfg=gpu:0,lk,_detrac_:s0_39:d30_70,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_39:d30_70:lk:svm

<a id="tmpls2___detrac_0_39_30_70_lk_"></a>
### tmpls2       @ detrac:0_39_30_70/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2,_detrac_:s:0_39:d:30_70,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_39:d30_70:lk:svm:tmpls2

<a id="tmpls5___detrac_0_39_30_70_lk_"></a>
### tmpls5       @ detrac:0_39_30_70/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls5,_detrac_:s:0_39:d:30_70,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_39:d30_70:lk:svm:tmpls5

<a id="detrac_train_0_59_test_60_99___lk_"></a>
## detrac:train_0_59_test_60_99       @ LK-->dmdp_svm

<a id="tmpls2___detrac_train_0_59_test_60_99_l_k_"></a>
### tmpls2       @ detrac:train_0_59_test_60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain0_59:stest60_99:d-100_100,_svm_:active,_svm_:lost,_test_:max_lost0:active_train,_train_:detrac:s0_59:d-100_100:lk:wrapper:svm:tmpls2:++active_pt:svm train.load=0

<a id="min10___tmpls2_detrac_train_0_59_test_60_99_lk_"></a>
#### min10       @ tmpls2/detrac:train_0_59_test_60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:wrapper:tmpls2,_detrac_:strain0_59:stest60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:active_train,_train_:detrac:s0_59:d-100_100:lk:wrapper:svm:min10:tmpls2:++active_pt:svm train.load=0

<a id="tmpls5___detrac_train_0_59_test_60_99_l_k_"></a>
### tmpls5       @ detrac:train_0_59_test_60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls5,_detrac_:strain0_59:stest60_99:d-100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:detrac:s0_59:d-100_100:lk:svm:tmpls5 train.load=0


<a id="mot15_train_0_10_test_11_21___lk_"></a>
## mot15:train_0_10_test_11_21       @ LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2:wrapper,_mot15_:strain0_10:stest11_21:d-100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:mot15:s0_10:d-100_100:lk:wrapper:svm:tmpls2:++active_pt:svm train.load=0

<a id="min10___mot15_train_0_10_test_11_21_lk_"></a>
### min10       @ mot15:train_0_10_test_11_21/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2:wrapper,_mot15_:strain0_10:stest11_21:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s0_10:d-100_100:lk:wrapper:svm:min10:tmpls2:++active_pt:svm train.load=0

<a id="mot17_train_0_6_test_7_13___lk_"></a>
## mot17:train_0_6_test_7_13       @ LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_mot17_:strain0_6:stest7_13:d-100_100,_svm_:active,_svm_:lost,_test_:max_lost0:vis,_train_:mot17:s0_6:d-100_100:lk:wrapper:svm:tmpls2:++active_pt:svm train.load=0

<a id="min10___mot17_train_0_6_test_7_13_lk_"></a>
### min10       @ mot17:train_0_6_test_7_13/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_mot17_:strain0_6:stest7_13:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s0_6:d-100_100:lk:wrapper:svm:min10:tmpls2:++active_pt:svm train.load=0

<a id="detrac_0_59_60_99___lk_"></a>
## detrac:0_59:60_99       @ LK-->dmdp_svm

<a id="tmpls2___detrac_0_59_60_99_lk_"></a>
### tmpls2       @ detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1

__dbg__
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_59:stest-76_76:d-100_1,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1

<a id="active_mlp___tmpls2_detrac_0_59_60_99_l_k_"></a>
#### active_mlp       @ tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_detrac_:strain-0_59:stest-60_99:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm,_active_:detrac:s-0_59:d-100_100:24_48_64_48_24_bn_ohem2 @test load=0 evaluate=1 @train load=1 @test start=16


<a id="on_train___tmpls2_detrac_0_59_60_99_l_k_"></a>
#### on_train       @ tmpls2/detrac:0_59:60_99/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-0_59:stest-0_59:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1

<a id="ctm_darpn___tmpls2_detrac_0_59_60_99_l_k_"></a>
#### ctm:darpn       @ tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0,_tracked_:none:thresh-10 @test load=0 evaluate=1 @train load=0

<a id="ign___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
##### ign       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis:tracked:ign,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0,_tracked_:none:ign @test load=0 evaluate=1 @train load=1


<a id="tracked_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
##### tracked_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked,_test_:max_lost0:vis:++tracked:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="ign___tracked_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
###### ign       @ tracked_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked:ign,_test_:max_lost0:vis:++tracked:pos:ign,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="tracked_lost_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
##### tracked_lost_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_pos_:lost,_pos_:tracked,_test_:max_lost0:vis:++lost:pos:++tracked:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="all_pos___ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
##### all_pos       @ ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_pos_:active,_pos_:lost,_pos_:tracked,_test_:max_lost0:vis:++active:pos:++lost:pos:++tracked:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="ign___all_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
###### ign       @ all_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_pos_:active,_pos_:lost,_pos_:tracked:ign,_test_:max_lost0:vis:++active:pos:++lost:pos:++tracked:pos:ign,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="active_tracked_pos___all_pos_ctm_darpn_tmpls2_detrac_0_59_60_99_l_k_"></a>
###### active_tracked_pos       @ all_pos/ctm:darpn/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_darpn_:tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked:++active,_test_:max_lost0:vis:++tracked:pos:++active:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:darpn:m0 @test load=0 evaluate=1 @train load=1

<a id="ctm_pyt_prdimp_18___tmpls2_detrac_0_59_60_99_l_k_"></a>
#### ctm:pyt:prdimp:18       @ tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:1,_lk_:tmpls2:wrapper,_pyt_:tracked:dimp-18:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:pyt:prdimp-18,_tracked_:none:thresh-10 @test load=0 evaluate=1 @train load=0

<a id="ctm_siamx_rpn_nxt___tmpls2_detrac_0_59_60_99_l_k_"></a>
#### ctm:siamx:rpn_nxt       @ tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:siamx:rpn_nxt,_tracked_:none:thresh-10 @test load=0 evaluate=1 @train load=0

<a id="tracked_pos___ctm_siamx_rpn_nxt_tmpls2_detrac_0_59_60_99_l_k_"></a>
##### tracked_pos       @ ctm:siamx:rpn_nxt/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked,_test_:max_lost0:vis:++tracked:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:siamx:rpn_nxt @test load=0 evaluate=1 @train load=1
__dbg__
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked,_test_:max_lost0:vis:++tracked:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:siamx:rpn_nxt @test load=0 evaluate=1 @train load=1

<a id="active_tracked_pos___tracked_pos_ctm_siamx_rpn_nxt_tmpls2_detrac_0_59_60_99_l_k_"></a>
###### active_tracked_pos       @ tracked_pos/ctm:siamx:rpn_nxt/tmpls2/detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:2,_lk_:tmpls2:wrapper,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_pos_:tracked:++active,_test_:max_lost0:vis:++tracked:pos:++active:pos,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm:++ctm:siamx:rpn_nxt @test load=0 evaluate=1 @train load=1


<a id="tmpls5___detrac_0_59_60_99_lk_"></a>
### tmpls5       @ detrac:0_59:60_99/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls5:wrapper,_detrac_:strain-0_59:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-0_59:d-100_100:lk:svm:wrapper:tmpls5:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1

<a id="detrac_train_60_99_test_0_59___lk_"></a>
## detrac:train:60_99:test:0_59       @ LK-->dmdp_svm

<a id="tmpls2___detrac_train_60_99_test_0_59_l_k_"></a>
### tmpls2       @ detrac:train:60_99:test:0_59/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-60_99:stest-0_59:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-60_99:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=0

<a id="on_train___detrac_train_60_99_test_0_59_l_k_"></a>
### on_train       @ detrac:train:60_99:test:0_59/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_detrac_:strain-60_99:stest-60_99:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:detrac:s-60_99:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1


<a id="mot15_0_4_5_10___lk_"></a>
## mot15:0_4:5_10       @ LK-->dmdp_svm

<a id="tmpls2___mot15_0_4_5_10_l_k_"></a>
### tmpls2       @ mot15:0_4:5_10/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_4:stest-5_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_4:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=0 

<a id="mot15_0_10_11_21___lk_"></a>
## mot15:0_10:11_21       @ LK-->dmdp_svm

<a id="tmpls2___mot15_0_10_11_21_l_k_"></a>
### tmpls2       @ mot15:0_10:11_21/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-11_21:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1 @tester save_debug_info=0 


<a id="on_train___tmpls2_mot15_0_10_11_21_lk_"></a>
#### on_train       @ tmpls2/mot15:0_10:11_21/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester devkit=1

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester devkit=0

<a id="tmpls5___mot15_0_10_11_21_l_k_"></a>
### tmpls5       @ mot15:0_10:11_21/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls5:wrapper,_mot15_:strain-0_10:stest-11_21:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls5:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1

<a id="on_train___tmpls5_mot15_0_10_11_21_lk_"></a>
#### on_train       @ tmpls5/mot15:0_10:11_21/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls5:wrapper,_mot15_:strain-0_10:stest-0_10:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls5:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1


<a id="mot17_train_0_6_test_7_13___lk__1"></a>
## mot17:train:0_6:test:7_13       @ LK-->dmdp_svm

<a id="tmpls2___mot17_train_0_6_test_7_13_lk_"></a>
### tmpls2       @ mot17:train:0_6:test:7_13/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-7_13:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0

<a id="on_train___tmpls2_mot17_train_0_6_test_7_13_l_k_"></a>
#### on_train       @ tmpls2/mot17:train:0_6:test:7_13/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester devkit=0

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester devkit=1

<a id="tmpls5___mot17_train_0_6_test_7_13_lk_"></a>
### tmpls5       @ mot17:train:0_6:test:7_13/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls5:wrapper,_mot17_:strain-0_6:stest-7_13:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls5:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0

<a id="on_train___tmpls5_mot17_train_0_6_test_7_13_l_k_"></a>
#### on_train       @ tmpls5/mot17:train:0_6:test:7_13/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls5:wrapper,_mot17_:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls5:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0


<a id="mot17_sdp_train_0_6_test_7_13___lk_"></a>
## mot17:sdp:train:0_6:test:7_13       @ LK-->dmdp_svm

<a id="tmpls2___mot17_sdp_train_0_6_test_7_13_lk_"></a>
### tmpls2       @ mot17:sdp:train:0_6:test:7_13/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:sdp:strain-0_6:stest-7_13:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:sdp:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0

<a id="on_train___tmpls2_mot17_sdp_train_0_6_test_7_13_l_k_"></a>
#### on_train       @ tmpls2/mot17:sdp:train:0_6:test:7_13/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:sdp:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:sdp:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester  save_debug_info=0

<a id="mot17_dpm_train_0_6_test_7_13___lk_"></a>
## mot17:dpm:train:0_6:test:7_13       @ LK-->dmdp_svm

<a id="tmpls2___mot17_dpm_train_0_6_test_7_13_lk_"></a>
### tmpls2       @ mot17:dpm:train:0_6:test:7_13/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:dpm:strain-0_6:stest-7_13:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:dpm:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0

<a id="on_train___tmpls2_mot17_dpm_train_0_6_test_7_13_l_k_"></a>
#### on_train       @ tmpls2/mot17:dpm:train:0_6:test:7_13/LK-->dmdp_svm
python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:dpm:strain-0_6:stest-0_6:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:mot17:dpm:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0

<a id="gram_0_2_0_2_40_n60___lk_"></a>
## gram:0_2:0_2:40_n60       @ LK-->dmdp_svm

<a id="tmpls2___gram_0_2_0_2_40_n60_lk_"></a>
### tmpls2       @ gram:0_2:0_2:40_n60/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_gram_:strain-0_2:stest-0_2:d-40_n60,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:gram:s-0_2:d-40_n60:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1 

__dbg__

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_gram_:strain-0_2:stest-0_2:d-100_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:gram:s-0_2:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 start=6 evaluate=1 @train load=0 @tester visualizer.mode=0,1,1

<a id="on_train___tmpls2_gram_0_2_0_2_40_n60_l_k_"></a>
#### on_train       @ tmpls2/gram:0_2:0_2:40_n60/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_gram_:strain-0_2:stest-0_2:d-40_40,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:gram:s-0_2:d-40_n60:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1 

<a id="on_all___tmpls2_gram_0_2_0_2_40_n60_l_k_"></a>
#### on_all       @ tmpls2/gram:0_2:0_2:40_n60/LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_gram_:strain-0_2:stest-0_2:d-40_100,_svm_:active,_svm_:lost:minr10,_test_:max_lost0:vis,_train_:gram:s-0_2:d-40_n60:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 @train load=1 

<a id="mot15_0_10_0_0___lk_"></a>
## mot15:0_10:0_0       @ LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_0:d-100_5,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester save_debug_info=0 

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot15_:strain-0_10:stest-0_0:d-100_5,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:mot15:s-0_10:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 @train load=1 @tester save_debug_info=0 devkit=0

<a id="mot17_train_0_6_test_0_0_100_5___lk_"></a>
## mot17:train:0_6:test:0_0:100_5       @ LK-->dmdp_svm

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_0:d-100_5,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=0 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0 

python3 main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_mot17_:strain-0_6:stest-0_0:d-100_5,_svm_:active,_svm_:lost:minr10,_test_:max_lost0,_train_:mot17:s-0_6:d-100_100:lk:svm:wrapper:tmpls2:min10:++active_pt:svm @test load=1 evaluate=1 subseq_postfix=0 @train load=1 @tester save_debug_info=0 devkit=0
