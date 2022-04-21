# even_odd
python36 main.py cfg=gpu:0,_gtt_:exit0:tmpls1,_ctmc_:strain-0_46:even:+++stest-0_46:odd:+++d-100_100,_rand_:lost+active+tracked,_test_:++active:rand:++lost:rand:++tracked:rand:++glob,_train_:ctmc:s-0_46:even:d-100_100:gtt:exit0:tmpls1:det_train+det_test:rec-80:prec-80 @test load=1 evaluate=1 subseq_postfix=0 @trainer vis=0 @train load=-1 @tester input.tracking_res.clamp_scores=2 vis=0

# all
python36 main.py cfg=gpu:0,_gtt_:exit0:tmpls1,_ctmc_:strain-0_46:stest-0_46:d-100_100,_rand_:lost+active+tracked,_test_:++active:rand:++lost:rand:++tracked:rand:++glob,_train_:ctmc:s-0_46:even:d-100_100:gtt:exit0:tmpls1:det_train+det_test:rec-80:prec-80 @test load=1 evaluate=1 subseq_postfix=0 @trainer vis=0 @train load=-1 @tester input.tracking_res.clamp_scores=2 vis=0
