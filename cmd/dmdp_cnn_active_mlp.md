<!-- MarkdownTOC -->

- [detrac_0_0_1_0       @ detrac](#detrac_0_0_1_0___detrac_)
- [detrac_0_0_5_0       @ detrac](#detrac_0_0_5_0___detrac_)
    - [yolo3d       @ detrac_0_0_5_0](#yolo3d___detrac_0_0_5_0_)
- [detrac:0_0_100_0       @ detrac](#detrac_0_0_100_0___detrac_)
    - [test_1_1_100       @ detrac:0_0_100_0](#test_1_1_100___detrac_0_0_100_0_)
        - [alx       @ test_1_1_100/detrac:0_0_100_0](#alx___test_1_1_100_detrac_0_0_100_0_)
            - [tracked_shr1       @ alx/test_1_1_100/detrac:0_0_100_0](#tracked_shr1___alx_test_1_1_100_detrac_0_0_100_0_)
        - [cnv0       @ test_1_1_100/detrac:0_0_100_0](#cnv0___test_1_1_100_detrac_0_0_100_0_)
        - [cnv1       @ test_1_1_100/detrac:0_0_100_0](#cnv1___test_1_1_100_detrac_0_0_100_0_)
        - [cnv2       @ test_1_1_100/detrac:0_0_100_0](#cnv2___test_1_1_100_detrac_0_0_100_0_)
            - [darpn       @ cnv2/test_1_1_100/detrac:0_0_100_0](#darpn___cnv2_test_1_1_100_detrac_0_0_100_0_)
    - [debug       @ detrac:0_0_100_0](#debug___detrac_0_0_100_0_)
    - [no_ohem       @ detrac:0_0_100_0](#no_ohem___detrac_0_0_100_0_)
- [detrac:0_9_100_0       @ detrac](#detrac_0_9_100_0___detrac_)
    - [test_30_49_100       @ detrac:0_9_100_0](#test_30_49_100___detrac_0_9_100_0_)
        - [darpn       @ test_30_49_100/detrac:0_9_100_0](#darpn___test_30_49_100_detrac_0_9_100_0_)
            - [fc3_19       @ darpn/test_30_49_100/detrac:0_9_100_0](#fc3_19___darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0](#lost___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tracked       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0](#tracked___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0](#lost_tracked___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls5       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls5___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_)
            - [c2f_19       @ darpn/test_30_49_100/detrac:0_9_100_0](#c2f_19___darpn_test_30_49_100_detrac_0_9_100_0_)
            - [incp3       @ darpn/test_30_49_100/detrac:0_9_100_0](#incp3___darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost_ctm       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#lost_ctm___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls2       @ lost_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2___lost_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls5       @ lost_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls5___lost_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#lost_tracked___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls3:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls3_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls4:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls4_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2:kf10:no_smr:yolo       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2_kf10_no_smr_yolo___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [0       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#0___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [2       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#2___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [4       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#4___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [14       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#14___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [203       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#203___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [475       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#475___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [656       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#656___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [777       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#777___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [863       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#863___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [930       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#930___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [963       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#963___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [983       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#983___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked_ctm       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0](#lost_tracked_ctm___incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls2       @ lost_tracked_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2___lost_tracked_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_)
            - [mbn2       @ darpn/test_30_49_100/detrac:0_9_100_0](#mbn2___darpn_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked       @ mbn2/darpn/test_30_49_100/detrac:0_9_100_0](#lost_tracked___mbn2_darpn_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2       @ mbn2/darpn/test_30_49_100/detrac:0_9_100_0](#tmpls2___mbn2_darpn_test_30_49_100_detrac_0_9_100_0_)
        - [siamx:fc       @ test_30_49_100/detrac:0_9_100_0](#siamx_fc___test_30_49_100_detrac_0_9_100_0_)
            - [mbn2       @ siamx:fc/test_30_49_100/detrac:0_9_100_0](#mbn2___siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked       @ mbn2/siamx:fc/test_30_49_100/detrac:0_9_100_0](#lost_tracked___mbn2_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2       @ mbn2/siamx:fc/test_30_49_100/detrac:0_9_100_0](#tmpls2___mbn2_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
            - [incp3       @ siamx:fc/test_30_49_100/detrac:0_9_100_0](#incp3___siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                - [lost_ctm       @ incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0](#lost_ctm___incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls2       @ lost_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0](#tmpls2___lost_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls5       @ lost_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0](#tmpls5___lost_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked_ctm       @ incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0](#lost_tracked_ctm___incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
                    - [tmpls2       @ lost_tracked_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0](#tmpls2___lost_tracked_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_)
        - [siamx:rpn_nxt       @ test_30_49_100/detrac:0_9_100_0](#siamx_rpn_nxt___test_30_49_100_detrac_0_9_100_0_)
            - [mbn2       @ siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#mbn2___siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                - [lost_tracked       @ mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#lost_tracked___mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                - [tmpls2       @ mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#tmpls2___mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                    - [continuous       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#continuous___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                    - [pyt:atom       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#pyt_atom___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                    - [pyt:eco       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#pyt_eco___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                    - [pyt:dimp18       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#pyt_dimp18___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
                    - [pyt:prdimp18       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0](#pyt_prdimp18___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_)
- [detrac:train:60_99:test:0_59:darpn:incp3](#detrac_train_60_99_test_0_59_darpn_incp_3_)
    - [lost_ctm       @ detrac:train:60_99:test:0_59:darpn:incp3](#lost_ctm___detrac_train_60_99_test_0_59_darpn_incp3_)
        - [tmpls2       @ lost_ctm/detrac:train:60_99:test:0_59:darpn:incp3](#tmpls2___lost_ctm_detrac_train_60_99_test_0_59_darpn_incp_3_)
- [detrac:train:0_9:test:10_19:darpn:incp3:lost_ctm:tmpls2](#detrac_train_0_9_test_10_19_darpn_incp3_lost_ctm_tmpls2_)
- [detrac:train_0_59_test_60_99:darpn:incp3](#detrac_train_0_59_test_60_99_darpn_incp_3_)
    - [lost_ctm       @ detrac:train_0_59_test_60_99:darpn:incp3](#lost_ctm___detrac_train_0_59_test_60_99_darpn_incp3_)
        - [tmpls2       @ lost_ctm/detrac:train_0_59_test_60_99:darpn:incp3](#tmpls2___lost_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_)
        - [tmpls5       @ lost_ctm/detrac:train_0_59_test_60_99:darpn:incp3](#tmpls5___lost_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_)
    - [lost_tracked       @ detrac:train_0_59_test_60_99:darpn:incp3](#lost_tracked___detrac_train_0_59_test_60_99_darpn_incp3_)
        - [tmpls2       @ lost_tracked/detrac:train_0_59_test_60_99:darpn:incp3](#tmpls2___lost_tracked_detrac_train_0_59_test_60_99_darpn_incp_3_)
    - [lost_tracked_ctm       @ detrac:train_0_59_test_60_99:darpn:incp3](#lost_tracked_ctm___detrac_train_0_59_test_60_99_darpn_incp3_)
        - [tmpls2       @ lost_tracked_ctm/detrac:train_0_59_test_60_99:darpn:incp3](#tmpls2___lost_tracked_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_)
- [mot15:train_0_10_test_11_21:darpn](#mot15_train_0_10_test_11_21_darpn_)
    - [incp3       @ mot15:train_0_10_test_11_21:darpn](#incp3___mot15_train_0_10_test_11_21_darp_n_)
        - [lost_ctm       @ incp3/mot15:train_0_10_test_11_21:darpn](#lost_ctm___incp3_mot15_train_0_10_test_11_21_darp_n_)
            - [tmpls2       @ lost_ctm/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls2___lost_ctm_incp3_mot15_train_0_10_test_11_21_darpn_)
            - [tmpls5       @ lost_ctm/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls5___lost_ctm_incp3_mot15_train_0_10_test_11_21_darpn_)
        - [lost_tracked       @ incp3/mot15:train_0_10_test_11_21:darpn](#lost_tracked___incp3_mot15_train_0_10_test_11_21_darp_n_)
            - [tmpls2:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls2_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_)
            - [tmpls3:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls3_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_)
            - [tmpls4:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls4_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_)
        - [lost_tracked_ctm       @ incp3/mot15:train_0_10_test_11_21:darpn](#lost_tracked_ctm___incp3_mot15_train_0_10_test_11_21_darp_n_)
            - [tmpls2       @ lost_tracked_ctm/incp3/mot15:train_0_10_test_11_21:darpn](#tmpls2___lost_tracked_ctm_incp3_mot15_train_0_10_test_11_21_darpn_)
- [mot17:train_0_6_test_7_13:darpn](#mot17_train_0_6_test_7_13_darpn_)
    - [incp3       @ mot17:train_0_6_test_7_13:darpn](#incp3___mot17_train_0_6_test_7_13_darp_n_)
        - [lost_ctm       @ incp3/mot17:train_0_6_test_7_13:darpn](#lost_ctm___incp3_mot17_train_0_6_test_7_13_darp_n_)
            - [tmpls2       @ lost_ctm/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls2___lost_ctm_incp3_mot17_train_0_6_test_7_13_darpn_)
            - [tmpls5       @ lost_ctm/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls5___lost_ctm_incp3_mot17_train_0_6_test_7_13_darpn_)
        - [lost_tracked       @ incp3/mot17:train_0_6_test_7_13:darpn](#lost_tracked___incp3_mot17_train_0_6_test_7_13_darp_n_)
            - [tmpls2:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls2_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_)
            - [tmpls3:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls3_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_)
            - [tmpls4:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls4_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_)
        - [lost_tracked_ctm       @ incp3/mot17:train_0_6_test_7_13:darpn](#lost_tracked_ctm___incp3_mot17_train_0_6_test_7_13_darp_n_)
            - [tmpls2       @ lost_tracked_ctm/incp3/mot17:train_0_6_test_7_13:darpn](#tmpls2___lost_tracked_ctm_incp3_mot17_train_0_6_test_7_13_darpn_)
- [detrac:60_99:test:0_59:24_48_64_48_24:darpn:tracked:none](#detrac_60_99_test_0_59_24_48_64_48_24_darpn_tracked_non_e_)
    - [ctm:darpn       @ detrac:60_99:test:0_59:24_48_64_48_24:darpn:tracked:none](#ctm_darpn___detrac_60_99_test_0_59_24_48_64_48_24_darpn_tracked_none_)
- [detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_non_e_)
    - [ctm:darpn       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#ctm_darpn___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_)
        - [on_train       @ ctm:darpn/detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#on_train___ctm_darpn_detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_)
    - [ctm:pyt:dimp:18       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#ctm_pyt_dimp_18___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_)
    - [ctm:pyt:prdimp:18       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#ctm_pyt_prdimp_18___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_)
    - [ctm:siamx:rpn_nxt       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none](#ctm_siamx_rpn_nxt___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_)
- [mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none](#mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_non_e_)
    - [ctm:darpn       @ mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none](#ctm_darpn___mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_none_)
        - [on_train       @ ctm:darpn/mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none](#on_train___ctm_darpn_mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_none_)
- [mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none](#mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_non_e_)
    - [ctm:darpn       @ mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none](#ctm_darpn___mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_none_)
        - [on_train       @ ctm:darpn/mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none](#on_train___ctm_darpn_mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_none_)
- [dtest:no_ohem](#dtest_no_ohem_)
    - [gram:0_2:40_n60:24_48_64_48_24:darpn:tracked:none       @ dtest:no_ohem](#gram_0_2_40_n60_24_48_64_48_24_darpn_tracked_none___dtest_no_ohe_m_)
    - [detrac:0_9:10_19:24_48_64_48_24:darpn:tracked:none       @ dtest:no_ohem](#detrac_0_9_10_19_24_48_64_48_24_darpn_tracked_none___dtest_no_ohe_m_)

<!-- /MarkdownTOC -->



<a id="detrac_0_0_1_0___detrac_"></a>
# detrac_0_0_1_0       @ detrac

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:s:0_0:d:1_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:e1k:b1,_ibt_:acc:lost:detrac:0_9_100_0:siamfc:cnn:n6:t2 @ibt test_iters=0,1,3,5 start=0 test_cfgs+=detrac:s:30_49:d:100_100

<a id="detrac_0_0_5_0___detrac_"></a>
# detrac_0_0_5_0       @ detrac

<a id="yolo3d___detrac_0_0_5_0_"></a>
## yolo3d       @ detrac_0_0_5_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc1,_detrac_:s:0_0:d:5_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:e1k:b1,_ibt_:acc:lost:detrac:0_9_100_0:siamfc:cnn:n6:t2 @ibt test_iters=0,1,3,5 start=0 test_cfgs+=detrac:s:30_49:d:100_100 @trainer yolo3d=1 

<a id="detrac_0_0_100_0___detrac_"></a>
# detrac:0_0_100_0       @ detrac

<a id="test_1_1_100___detrac_0_0_100_0_"></a>
## test_1_1_100       @ detrac:0_0_100_0

<a id="alx___test_1_1_100_detrac_0_0_100_0_"></a>
### alx       @ test_1_1_100/detrac:0_0_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:alx:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:sc2:cnn:alx:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:r1_1:d:100_100 start=03

<a id="tracked_shr1___alx_test_1_1_100_detrac_0_0_100_0_"></a>
#### tracked_shr1       @ alx/test_1_1_100/detrac:0_0_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:shr1:lost:alx:e1k:ohem2:lr1_4:b10:s:70_30,_cnn_:shr1:tracked:alx:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:sc2:cnn:alx:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s2_2:d:100_100 start=03

<a id="cnv0___test_1_1_100_detrac_0_0_100_0_"></a>
### cnv0       @ test_1_1_100/detrac:0_0_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc1,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:cnv:0:lost:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:sc1:cnn:cnv:0:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s1_1:d:100_100 start=00

<a id="cnv1___test_1_1_100_detrac_0_0_100_0_"></a>
### cnv1       @ test_1_1_100/detrac:0_0_100_0

python3 main.py cfg=gpu:1,_siamfc_:nms1:f0:sc1,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:cnv:1:lost:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:sc1:cnn:cnv:1:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:r1_1:d:100_100 start=01

<a id="cnv2___test_1_1_100_detrac_0_0_100_0_"></a>
### cnv2       @ test_1_1_100/detrac:0_0_100_0

<a id="darpn___cnv2_test_1_1_100_detrac_0_0_100_0_"></a>
#### darpn       @ cnv2/test_1_1_100/detrac:0_0_100_0

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv2:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:darpn:m0:cnn:cnv2:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:r1_1:d:100_100 start=01

<a id="debug___detrac_0_0_100_0_"></a>
## debug       @ detrac:0_0_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:cnn:n6:t2,_d_ @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:r1_1:d:100_100 start=03 



<a id="no_ohem___detrac_0_0_100_0_"></a>
## no_ohem       @ detrac:0_0_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:s:0_0:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:e1k:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_0_100_0:siamfc:cnn:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:r1_1:d:100_100 start=01


<a id="detrac_0_9_100_0___detrac_"></a>
# detrac:0_9_100_0       @ detrac

<a id="test_30_49_100___detrac_0_9_100_0_"></a>
## test_30_49_100       @ detrac:0_9_100_0

python3 main.py cfg=gpu:0,_siamfc_:nms1:f0:sc2,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:e1k:b1,_ibt_:acc:lost:detrac:0_9_100_0:siamfc:cnn:n6:t2 @ibt test_iters=0,1,3,5 start=0 test_cfgs+=detrac:s:30_49:d:100_100

<a id="darpn___test_30_49_100_detrac_0_9_100_0_"></a>
### darpn       @ test_30_49_100/detrac:0_9_100_0

<a id="fc3_19___darpn_test_30_49_100_detrac_0_9_100_0_"></a>
#### fc3_19       @ darpn/test_30_49_100/detrac:0_9_100_0
        ## mlp
<a id="lost___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:fc3_19:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_9_100_0:darpn:m0:cnn:fc3_19:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:30_49:d:100_100 start=00

<a id="tracked___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tracked       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:2,_darpn_:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:tracked:cnv:fc3_19:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tracked:detrac:0_9_100_0:darpn:m0:cnn:fc3_19:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:30_49:d:100_100 start=00


<a id="lost_tracked___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:2,_darpn_:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:fc3_19:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:cnv:c2f_19:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:lost:detrac:0_9_100_0:darpn:m0:cnn:fc3_19,_ibt_:tracked:cnn:c2f_19 @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=10

__debug_0_0_1_0_e2__       

python3 main.py cfg=gpu:2,_darpn_:nms0:f0:m0,_detrac_:s:0_0:d:1_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:fc3_19:e2:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:cnv:c2f_19:e2:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:lost:detrac:0_9_100_0:darpn:m0:cnn:fc3_19,_ibt_:tracked:cnn:c2f_19 @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:1_1:d:1_1 start=10

<a id="tmpls2___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:tmpls2:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:fc3_19:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:cnv:c2f_19:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:darpn:m0:cnn:fc3_19,_ibt_:tracked:cnn:c2f_19 @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100:start:test:4 start=13


<a id="tmpls5___fc3_19_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls5       @ fc3_19/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_darpn_:tmpls5:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:fc3_19:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:cnv:c2f_19:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls5:lost:detrac:0_9_100_0:darpn:m0:cnn:fc3_19,_ibt_:tracked:cnn:c2f_19 @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=10


<a id="c2f_19___darpn_test_30_49_100_detrac_0_9_100_0_"></a>
#### c2f_19       @ darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_darpn_:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:cnv:c2f_19:e1k:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:lost:detrac:0_9_100_0:darpn:m0:cnn:cnv:c2f_19:n6:t2 @ibt test_iters=0,1,3,5 test_cfgs+=detrac:s:30_49:d:100_100 start=01


<a id="incp3___darpn_test_30_49_100_detrac_0_9_100_0_"></a>
#### incp3       @ darpn/test_30_49_100/detrac:0_9_100_0

<a id="lost_ctm___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_ctm       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

<a id="tmpls2___lost_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls2       @ lost_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:darpn:m0 @ibt test_iters=3,5 test_cfgs+=0:detrac:stest30_49:d-100_100,_test_:load start=53

<a id="tmpls5___lost_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls5       @ lost_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls5,_darpn_:tracked:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:darpn:m0 @ibt test_iters=3,5 test_cfgs+=0:_detrac_:stest30_49:d-100_100,_test_:load start=53

<a id="lost_tracked___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

<a id="tmpls2_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e40:ohem2:lr3:schd:step1:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:kf10:n3:t2:detrac:s0_9:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=000

<a id="tmpls3_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls3:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_darpn_:tmpls3:kf10,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e40:ohem2:lr3:schd:step1:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls3:kf10:n3:t2:detrac:s0_9:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=000

<a id="tmpls4_kf10_no_smr___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls4:kf10:no_smr       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:2,_darpn_:tmpls4:kf10,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e40:ohem2:lr3:schd:step1:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls4:kf10:n3:t2:detrac:s0_9:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=000

<a id="tmpls2_kf10_no_smr_yolo___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2:kf10:no_smr:yolo       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

<a id="0___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 0       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="2___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 2       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:1,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-2:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="4___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 4       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:2,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-4:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="14___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 14       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-14:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="203___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 203       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:1,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-203:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="475___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 475       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-475:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="656___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 656       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-656:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="777___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 777       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-777:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="863___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 863       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-863:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="930___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 930       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-930:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="963___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 963       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-963:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000

<a id="983___tmpls2_kf10_no_smr_yolo_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### 983       @ tmpls2:kf10:no_smr:yolo/incp3/darpn/test_30_49_100/detrac:0_9_100_0
python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain0_4:stest5_9:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-983:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,test:max_lost-100 start=000



<a id="lost_tracked_ctm___incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked_ctm       @ incp3/darpn/test_30_49_100/detrac:0_9_100_0

<a id="tmpls2___lost_tracked_ctm_incp3_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls2       @ lost_tracked_ctm/incp3/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:2,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:n4:t2:detrac:s0_9:d-100_100:+ctm:darpn:m0:lost+tracked:cnn:incp3:pt @ibt test_iters=2,3,4 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_detrac_:stest30_49:d-100_100 start=00

<a id="mbn2___darpn_test_30_49_100_detrac_0_9_100_0_"></a>
#### mbn2       @ darpn/test_30_49_100/detrac:0_9_100_0

<a id="lost_tracked___mbn2_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked       @ mbn2/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:lost:detrac:0_9_100_0:darpn:m0:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=01

<a id="tmpls2___mbn2_darpn_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2       @ mbn2/darpn/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_darpn_:tmpls2:nms0:f0:m0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:darpn:m0:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=01

<a id="siamx_fc___test_30_49_100_detrac_0_9_100_0_"></a>
### siamx:fc       @ test_30_49_100/detrac:0_9_100_0

<a id="mbn2___siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
#### mbn2       @ siamx:fc/test_30_49_100/detrac:0_9_100_0

<a id="lost_tracked___mbn2_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked       @ mbn2/siamx:fc/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_siamx_:fc:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:detrac:0_9_100_0:siamx:fc:lost+tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00

<a id="tmpls2___mbn2_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2       @ mbn2/siamx:fc/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:fc:tmpls2:nms0:f0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:fc:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=01

<a id="incp3___siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
#### incp3       @ siamx:fc/test_30_49_100/detrac:0_9_100_0

<a id="lost_ctm___incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_ctm       @ incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0

<a id="tmpls2___lost_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls2       @ lost_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:2,_siamx_:tmpls2:+tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:siamx:fc @ibt test_iters=3,5 test_cfgs+=0:_detrac_:stest30_49:d-100_100 start=00

__ctm_rpn_nxt__

python3 main.py cfg=gpu:2,_siamx_:tmpls2:fc:nms0:f0:++tracked:rpn_nxt:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:siamx:fc:++ctm:siamx:rpn_nxt @ibt test_iters=3,5 test_cfgs+=0:_detrac_:stest30_49:d-100_100 start=00


<a id="tmpls5___lost_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls5       @ lost_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:tmpls5:+tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:siamx:fc @ibt test_iters=3,5 test_cfgs+=0:_detrac_:stest30_49:d-100_100 start=00

__test_active_train__

python3 main.py cfg=gpu:1,_siamx_:tmpls5:+tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:siamx:fc @ibt test_iters=3,5 test_cfgs+=_detrac_:stest30_49:d-100_100,test:active_train start=33

__test_active_train_max_lost_50__

python3 main.py cfg=gpu:1,_siamx_:tmpls5:+tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:detrac:s0_9:d-100_100:cnn:incp3:n6:t2:+ctm:siamx:fc @ibt test_iters=3,5 test_cfgs+=_detrac_:stest30_49:d-100_100,test:active_train:max_lost50:load start=53

<a id="lost_tracked_ctm___incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked_ctm       @ incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0

<a id="tmpls2___lost_tracked_ctm_incp3_siamx_fc_test_30_49_100_detrac_0_9_100_0_"></a>
###### tmpls2       @ lost_tracked_ctm/incp3/siamx:fc/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:tmpls2:+tracked:fc:nms0:f0,_detrac_:strain0_9:stest30_49:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:n4:t2:detrac:s0_9:d-100_100:+ctm:siamx:fc:lost+tracked:cnn:incp3:pt @ibt test_iters=2,3,4 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_detrac_:stest30_49:d-100_100 start=00

<a id="siamx_rpn_nxt___test_30_49_100_detrac_0_9_100_0_"></a>
### siamx:rpn_nxt       @ test_30_49_100/detrac:0_9_100_0

<a id="mbn2___siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
#### mbn2       @ siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

<a id="lost_tracked___mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
##### lost_tracked       @ mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:0,_siamx_:rpn_nxt:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00

<a id="tmpls2___mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
##### tmpls2       @ mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:lost+tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=01


<a id="continuous___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
###### continuous       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:s:0_9:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00

<a id="pyt_atom___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
###### pyt:atom       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_pyt_:tracked:atom:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00

<a id="pyt_eco___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
###### pyt:eco       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_pyt_:tracked:eco:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00


<a id="pyt_dimp18___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
###### pyt:dimp18       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_pyt_:tracked:dimp:18:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00


<a id="pyt_prdimp18___tmpls2_mbn2_siamx_rpn_nxt_test_30_49_100_detrac_0_9_100_0_"></a>
###### pyt:prdimp18       @ tmpls2/mbn2/siamx:rpn_nxt/test_30_49_100/detrac:0_9_100_0

python3 main.py cfg=gpu:1,_siamx_:rpn_nxt:tmpls2:nms0:f0,_pyt_:tracked:prdimp:18:nms0:f0,_detrac_:s:0_9:d:10_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:mbn2:pt:e10:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_9_100_0:siamx:rpn_nxt:cnn:mbn2:pt,_ibt_:tracked:cnn:mbn2:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:30_49:d:100_100 start=00

<a id="detrac_train_60_99_test_0_59_darpn_incp_3_"></a>
# detrac:train:60_99:test:0_59:darpn:incp3   

<a id="lost_ctm___detrac_train_60_99_test_0_59_darpn_incp3_"></a>
## lost_ctm       @ detrac:train:60_99:test:0_59:darpn:incp3

<a id="tmpls2___lost_ctm_detrac_train_60_99_test_0_59_darpn_incp_3_"></a>
### tmpls2       @ lost_ctm/detrac:train:60_99:test:0_59:darpn:incp3

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain-60_99:stest-0_59:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s-60_99:d-100_100:cnn:incp3:n2:t2:+ctm:darpn:m0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=1 test_cfgs+=0:detrac:stest-0_59:d-100_100,_test_:load start=00


<a id="detrac_train_0_9_test_10_19_darpn_incp3_lost_ctm_tmpls2_"></a>
# detrac:train:0_9:test:10_19:darpn:incp3:lost_ctm:tmpls2

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain-0_9:stest-10_19:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s-0_9:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 test_cfgs+=0:detrac:stest-10_19:d-100_100,_test_:load start=12

<a id="detrac_train_0_59_test_60_99_darpn_incp_3_"></a>
# detrac:train_0_59_test_60_99:darpn:incp3   

<a id="lost_ctm___detrac_train_0_59_test_60_99_darpn_incp3_"></a>
## lost_ctm       @ detrac:train_0_59_test_60_99:darpn:incp3

<a id="tmpls2___lost_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_"></a>
### tmpls2       @ lost_ctm/detrac:train_0_59_test_60_99:darpn:incp3

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_detrac_:strain0_59:stest60_99:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:detrac:s-0_59:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 test_cfgs+=0:detrac:stest60_99:d-100_100,_test_:load start=00

<a id="tmpls5___lost_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_"></a>
### tmpls5       @ lost_ctm/detrac:train_0_59_test_60_99:darpn:incp3

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls5,_darpn_:tracked:nms0:f0,_detrac_:strain0_59:stest60_99:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:detrac:s-0_59:d-100_100:cnn:incp3:n6:t2:+ctm:darpn:m0 @ibt test_iters=3,5 test_cfgs+=0:_detrac_:stest60_99:d-100_100,_test_:load start=53

<a id="lost_tracked___detrac_train_0_59_test_60_99_darpn_incp3_"></a>
## lost_tracked       @ detrac:train_0_59_test_60_99:darpn:incp3

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0,_detrac_:s:0_59:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:incp3:pt:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:lost:detrac:0_59_100_0:darpn:m0:cnn:incp3:pt,_ibt_:tracked:cnn:incp3:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:60_99:d:100_100 start=01

<a id="tmpls2___lost_tracked_detrac_train_0_59_test_60_99_darpn_incp_3_"></a>
### tmpls2       @ lost_tracked/detrac:train_0_59_test_60_99:darpn:incp3

python3 main.py cfg=gpu:0,_darpn_:tmpls2:nms0:f0:m0,_detrac_:s:0_59:d:100_0,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e100:ohem2:lr1_4:b10:s:70_30,_cnn_:tracked:incp3:pt:e100:ohem2:lr1_4:b10:s:70_30,_ibt_:lost_tracked:acc:n6:t2:tmpls2:lost:detrac:0_59_100_0:darpn:m0:cnn:incp3:pt,_ibt_:tracked:cnn:incp3:pt @ibt test_iters=1,3,5 test_cfgs+=0:_detrac_:s:60_99:d:100_100 start=01

<a id="lost_tracked_ctm___detrac_train_0_59_test_60_99_darpn_incp3_"></a>
## lost_tracked_ctm       @ detrac:train_0_59_test_60_99:darpn:incp3

<a id="tmpls2___lost_tracked_ctm_detrac_train_0_59_test_60_99_darpn_incp_3_"></a>
### tmpls2       @ lost_tracked_ctm/detrac:train_0_59_test_60_99:darpn:incp3

python3 main.py cfg=gpu:2,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain0_59:stest60_99:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:n4:t2:detrac:s-0_59:d-100_100:+ctm:darpn:m0:lost+tracked:cnn:incp3:pt @ibt test_iters=2,3,4 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_detrac_:stest60_99:d-100_100 start=00

<a id="mot15_train_0_10_test_11_21_darpn_"></a>
# mot15:train_0_10_test_11_21:darpn     

<a id="incp3___mot15_train_0_10_test_11_21_darp_n_"></a>
## incp3       @ mot15:train_0_10_test_11_21:darpn

<a id="lost_ctm___incp3_mot15_train_0_10_test_11_21_darp_n_"></a>
### lost_ctm       @ incp3/mot15:train_0_10_test_11_21:darpn

<a id="tmpls2___lost_ctm_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls2       @ lost_ctm/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:1,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:mot15:s0_10:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0 @ibt test_iters=2,3 test_cfgs+=0:mot15:stest11_21:d-100_100 start=00

<a id="tmpls5___lost_ctm_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls5       @ lost_ctm/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls5,_darpn_:tracked:nms0:f0,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:mot15:s0_10:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0 @ibt test_iters=2,3 test_cfgs+=0:_mot15_:stest11_21:d-100_100,_test_:load start=00

<a id="lost_tracked___incp3_mot15_train_0_10_test_11_21_darp_n_"></a>
### lost_tracked       @ incp3/mot15:train_0_10_test_11_21:darpn

<a id="tmpls2_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls2:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e30:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:kf10:n3:t2:mot15:s0_10:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_mot15_:stest11_21:d-100_100 start=010

<a id="tmpls3_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls3:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:1,_darpn_:tmpls3:kf10,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e30:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls3:kf10:n3:t2:mot15:s0_10:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_mot15_:stest11_21:d-100_100 start=010

<a id="tmpls4_kf10_no_smr___lost_tracked_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls4:kf10:no_smr       @ lost_tracked/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:2,_darpn_:tmpls4:kf10,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e30:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls4:kf10:n3:t2:mot15:s0_10:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e20 test_cfgs+=0:_mot15_:stest11_21:d-100_100 start=010

<a id="lost_tracked_ctm___incp3_mot15_train_0_10_test_11_21_darp_n_"></a>
### lost_tracked_ctm       @ incp3/mot15:train_0_10_test_11_21:darpn

<a id="tmpls2___lost_tracked_ctm_incp3_mot15_train_0_10_test_11_21_darpn_"></a>
#### tmpls2       @ lost_tracked_ctm/incp3/mot15:train_0_10_test_11_21:darpn

python3 main.py cfg=gpu:1,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot15_:strain0_10:stest11_21:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:n3:t2:mot15:s0_10:d-100_100:+ctm:darpn:m0:lost+tracked:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_mot15_:stest11_21:d-100_100 start=101

<a id="mot17_train_0_6_test_7_13_darpn_"></a>
# mot17:train_0_6_test_7_13:darpn  

<a id="incp3___mot17_train_0_6_test_7_13_darp_n_"></a>
## incp3       @ mot17:train_0_6_test_7_13:darpn

<a id="lost_ctm___incp3_mot17_train_0_6_test_7_13_darp_n_"></a>
### lost_ctm       @ incp3/mot17:train_0_6_test_7_13:darpn

<a id="tmpls2___lost_ctm_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls2       @ lost_ctm/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls2,_darpn_:tracked:nms0:f0,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls2:lost:mot17:s0_6:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 test_cfgs+=0:mot17:stest7_13:d-100_100 start=00

<a id="tmpls5___lost_ctm_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls5       @ lost_ctm/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:0,_darpn_:nms0:f0:m0:tmpls5,_darpn_:tracked:nms0:f0,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:e15:ohem2:lr1_4:b10:s:70_30,_ibt_:acc:tmpls5:lost:mot17:s0_6:d-100_100:cnn:incp3:n3:t2:+ctm:darpn:m0:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 test_cfgs+=0:_mot17_:stest7_13:d-100_100,_test_:load start=00


<a id="lost_tracked___incp3_mot17_train_0_6_test_7_13_darp_n_"></a>
### lost_tracked       @ incp3/mot17:train_0_6_test_7_13:darpn

<a id="tmpls2_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls2:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:kf10:n3:t2:mot17:s0_6:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_mot17_:stest7_13:d-100_100 start=000

<a id="tmpls3_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls3:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:1,_darpn_:tmpls3:kf10,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls3:kf10:n3:t2:mot17:s0_6:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_mot17_:stest7_13:d-100_100 start=000

<a id="tmpls4_kf10_no_smr___lost_tracked_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls4:kf10:no_smr       @ lost_tracked/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:2,_darpn_:tmpls4:kf10,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:no_smr:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls4:kf10:n3:t2:mot17:s0_6:d-100_100:lost+tracked:no_smr:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_mot17_:stest7_13:d-100_100 start=000

<a id="lost_tracked_ctm___incp3_mot17_train_0_6_test_7_13_darp_n_"></a>
### lost_tracked_ctm       @ incp3/mot17:train_0_6_test_7_13:darpn

<a id="tmpls2___lost_tracked_ctm_incp3_mot17_train_0_6_test_7_13_darpn_"></a>
#### tmpls2       @ lost_tracked_ctm/incp3/mot17:train_0_6_test_7_13:darpn

python3 main.py cfg=gpu:2,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot17_:strain0_6:stest7_13:d-100_100,_mlp_:active:24_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:lost_tracked:acc:tmpls2:n3:t2:mot17:s0_6:d-100_100:+ctm:darpn:m0:lost+tracked:cnn:incp3:pt:++active_pt-24_48_24_bn_ohem2 @ibt test_iters=2,3 cfgs+=10:_cnn_:lost+tracked:e5 test_cfgs+=0:_mot17_:stest7_13:d-100_100 start=00


<a id="detrac_60_99_test_0_59_24_48_64_48_24_darpn_tracked_non_e_"></a>
# detrac:60_99:test:0_59:24_48_64_48_24:darpn:tracked:none

<a id="ctm_darpn___detrac_60_99_test_0_59_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:darpn       @ detrac:60_99:test:0_59:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain-60_99:stest-0_59:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-60_99:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-0_59:d-100_100 start=00

<a id="detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_non_e_"></a>
# detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

<a id="ctm_darpn___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:darpn       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-0_59:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-60_99:d-100_100 start=12 @test load=1

<a id="on_train___ctm_darpn_detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_"></a>
### on_train       @ ctm:darpn/detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain-0_59:stest-0_59:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-0_59:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-0_59:d-100_100 start=12

<a id="pyt_dimp18___darpn_detrac_0_59_test_60_99_24_48_64_48_24_"></a>

<a id="ctm_pyt_dimp_18___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:pyt:dimp:18       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:1,_darpn_:tmpls2:m0:nms0:f0,_pyt_:tracked:dimp-18:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-0_59:d-100_100:darpn:m0:++ctm:pyt:dimp-18:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-60_99:d-100_100 start=02

<a id="ctm_pyt_prdimp_18___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:pyt:prdimp:18       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:1,_darpn_:tmpls2:m0:nms0:f0,_pyt_:tracked:prdimp-18:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-0_59:d-100_100:darpn:m0:++ctm:pyt:prdimp-18:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-60_99:d-100_100 start=02

<a id="siamx_rpn_nxt___darpn_detrac_0_59_test_60_99_24_48_64_48_24_"></a>

<a id="ctm_siamx_rpn_nxt___detrac_0_59_test_60_99_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:siamx:rpn_nxt       @ detrac:0_59:test:60_99:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:2,_darpn_:tmpls2:m0:nms0:f0,_siamx_:tracked:rpn_nxt:nms0:f0,_detrac_:strain-0_59:stest-60_99:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:detrac:s-0_59:d-100_100:darpn:m0:++ctm:siamx:rpn_nxt:++lost:cnn:incp3:pt:++tracked:none:thresh-10:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_detrac_:stest-60_99:d-100_100 start=02


<a id="mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_non_e_"></a>
# mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none

<a id="ctm_darpn___mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:darpn       @ mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot15_:strain-0_10:stest-11_21:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:mot15:s-0_10:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_mot15_:stest-11_21:d-100_100 start=00

<a id="on_train___ctm_darpn_mot15_0_10_11_21_24_48_64_48_24_darpn_tracked_none_"></a>
### on_train       @ ctm:darpn/mot15:0_10:11_21:24_48_64_48_24:darpn:tracked:none
python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot15_:strain-0_10:stest-11_21:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:mot15:s-0_10:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_mot15_:stest-0_10:d-100_100 start=12 @test load=0 @tester max_lost_ratio=1

<a id="mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_non_e_"></a>
# mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none

<a id="ctm_darpn___mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_none_"></a>
## ctm:darpn       @ mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot17_:strain-0_6:stest-7_13:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:mot17:s-0_6:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_mot17_:stest-7_13:d-100_100 start=00

<a id="on_train___ctm_darpn_mot17_0_6_7_13_24_48_64_48_24_darpn_tracked_none_"></a>
### on_train       @ ctm:darpn/mot17:0_6:7_13:24_48_64_48_24:darpn:tracked:none
python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_mot17_:strain-0_6:stest-7_13:d-100_100,_mlp_:active:24_48_64_48_24:bn:ohem2:e1k:s:70_30:acc-994,_cnn_:lost:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:acc:tmpls2:n2:t2:mot17:s-0_6:d-100_100:+ctm:darpn:m0:++lost:cnn:incp3:pt:++active_pt-24_48_64_48_24_bn_ohem2,_tracked_:none:thresh-10 @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 test_cfgs+=0:_mot17_:stest-0_6:d-100_100 start=12 @test load=0 @tester max_lost_ratio=1

<a id="dtest_no_ohem_"></a>
# dtest:no_ohem

<a id="gram_0_2_40_n60_24_48_64_48_24_darpn_tracked_none___dtest_no_ohe_m_"></a>
## gram:0_2:40_n60:24_48_64_48_24:darpn:tracked:none       @ dtest:no_ohem

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_gram_:strain-0_2:stest-0_2:d-40_n60,_mlp_:active:24_48_64_48_24:bn:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:lr1_4:b10:s:70_30:acc994,_ibt_:dtest:tmpls2:n2:t2:gram:s-0_2:d-40_n60:+ctm:darpn:m0:++lost+tracked:cnn:incp3:pt:++active:mlp:24_48_64_48_24:bn,_active_:no_pt @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 start=012

__dbg__

python3 main.py cfg=gpu:0,_darpn_:tmpls2:+tracked:m0:nms0:f0,_gram_:strain-0_1:stest-0_1:d-10_10,_mlp_:active:24_48_64_48_24:bn:ohem2:e10:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:ohem2:lr1_4:b10:s:70_30:acc994,_ibt_:dtest:tmpls2:n2:t2:gram:s-0_2:d-40_n60:+ctm:darpn:m0:++lost+tracked:cnn:incp3:pt:++active:mlp:24_48_64_48_24:bn:ohem2,_active_:no_pt @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5

<a id="detrac_0_9_10_19_24_48_64_48_24_darpn_tracked_none___dtest_no_ohe_m_"></a>
## detrac:0_9:10_19:24_48_64_48_24:darpn:tracked:none       @ dtest:no_ohem

python3 main.py cfg=gpu:1,_darpn_:tmpls2:+tracked:m0:nms0:f0,_detrac_:strain-0_9:stest-10_19:d-100_100,_mlp_:active:24_48_64_48_24:bn:e1k:s:70_30:acc-994,_cnn_:lost+tracked:incp3:pt:e15:lr1_4:b10:s:70_30:acc994,_ibt_:dtest:tmpls2:n2:t2:detrac:s-0_9:d-100_100:+ctm:darpn:m0:++lost+tracked:cnn:incp3:pt:++active:mlp:24_48_64_48_24:bn,_active_:no_pt @ibt test_iters=1 cfgs+=10:_cnn_:lost:e5 start=01
