python3 main.py cfg=gpu:0,_darpn_:tmpls2:kf10,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-863:++active_pt-svm @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,_test_:max_lost-100 start=130

python3 main.py cfg=gpu:2,_darpn_:tmpls2:kf10,_detrac_:strain-0_4:stest-0_4:d-100_100,_svm_:active,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-863:++active_pt-svm @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:0_4:d:100_100,_test_:max_lost-100 start=130

python3 main.py cfg=gpu:1,_darpn_:tmpls2:kf10,_detrac_:strain-0_4:stest-5_9:d-100_100,_svm_:active,_cnn_:lost+tracked:no_smr:incp3:pt:e20:ohem2:lr3:schd:step1:b10:s:70_30:acc990,_ibt_:lost_tracked:acc:tmpls2:kf10:n2:t2:detrac:s0_4:d-100_100:lost+tracked:no_smr:cnn:incp3:pt::det_train+det_test:yolo-863:++active_pt-svm @ibt test_iters=1 cfgs+=10:_cnn_:lost+tracked:e10 test_cfgs+=0:_detrac_:s:5_9:d:100_100,_darpn_:tracked:nms0:f0:conf-10,_tracked_:none:thresh-10,_test_:max_lost-0:ctm:darpn:tracked:none:thresh-10 start=130
