# ibt

## grs       @ ibt-->rep

### oracle       @ grs/ibt-->rep

#### rec-40:prec-40       @ oracle/grs/ibt-->rep

python36 main.py cfg=gpu:1,_lk_:tmpls1:nofeat2-64,_ctmc_:strain-0_46:stest-0_46:d-50_n50,_cnn_:lost+active:r18:pt:e50:s:70_30:acc994:r0:b20:lr-5_4:schd:step-20_2,_tracked_:none:thresh-10,_ibt_:dtest:tmpls1:n1:t2:ctmc:s-0_46:d-50_n50:lk:nofeat2-64:++active+lost:cnn:r18:pt:det_train+det_test:rec-40:prec-40,_active_:no_pt @ibt test_iters=0 min_samples=1 ips=030 @test load=0 @@replace modules=active,lost,tracked token=grs:13.1:210424_055144_836621 @tester vis=0

#### rec-100:prec-100       @ oracle/grs/ibt-->rep

python36 main.py cfg=gpu:0,_lk_:tmpls1:nofeat2-64,_ctmc_:strain-0_46:even:++stest-0_46:odd:++d-100_100,_cnn_:lost+active:r18:pt:e50:s:70_30:acc994:r0:b20:lr-5_4:schd:step-20_2,_tracked_:none:thresh-10,_ibt_:dtest:tmpls1:n1:t2:ctmc:s-0_46:even:d-100_100:lk:nofeat2-64:++active+lost:cnn:r18:pt:det_train+det_test:rec-100:prec-100,_active_:no_pt @ibt test_iters=0 min_samples=1 ips=030 @test load=0 @@replace modules=tracked token=grs:13.1:210424_055144_836621 @tester vis=2 verbose=2

### pos       @ grs/ibt-->rep

python36 main.py cfg=gpu:1,_lk_:tmpls1:nofeat2-64,_ctmc_:strain-0_46:stest-0_46:d-50_n50,_cnn_:lost+active:r18:pt:e50:s:70_30:acc994:r0:b20:lr-5_4:schd:step-20_2,_tracked_:none:thresh-10,_ibt_:dtest:tmpls1:n1:t2:ctmc:s-0_46:d-50_n50:lk:nofeat2-64:++active+lost:cnn:r18:pt:det_train+det_test:rec-40:prec-40,_active_:no_pt @ibt test_iters=0 min_samples=1 ips=030 @test load=0 @@replace modules=active,lost,tracked token=grs:14.1:210424_055144_836649 @tester vis=0

# lk
main.py cfg=gpu:0,_lk_:tmpls2:wrapper,_ctmc_:'strain-0_46:stest-0_46:d-50_n50,_svm_:lost+active:min10:max50,_train_:ctmc:s-0_46:d-50_n50:lk:svm:wrapper:tmpls2:min10:det_train+det_test:rec-40:prec-40,_active_:no_pt @train load=1 @test load=0 evaluate=1 @@replace modules=active token=grs:13.1:210424_055144_836621

