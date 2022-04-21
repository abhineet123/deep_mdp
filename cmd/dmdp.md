<!-- MarkdownTOC -->

- [build](#build_)
- [getBoxStatistics](#getboxstatistic_s_)
- [visualize](#visualize_)
    - [DETRAC       @ visualize](#detrac___visualiz_e_)
        - [detections       @ DETRAC/visualize](#detections___detrac_visualize_)
            - [only_boxes       @ detections/DETRAC/visualize](#only_boxes___detections_detrac_visualiz_e_)
        - [annotations       @ DETRAC/visualize](#annotations___detrac_visualize_)
- [main](#mai_n_)
- [LK_SVM       @ main](#lk_svm___main_)
    - [detrac_0_to_9_40_60       @ LK_SVM](#detrac_0_to_9_40_60___lk_svm_)
        - [test_1       @ detrac_0_to_9_40_60/LK_SVM](#test_1___detrac_0_to_9_40_60_lk_svm_)
        - [5_5       @ detrac_0_to_9_40_60/LK_SVM](#5_5___detrac_0_to_9_40_60_lk_svm_)
    - [detrac_0_59_40_60       @ LK_SVM](#detrac_0_59_40_60___lk_svm_)
    - [detrac_0_59_100_0       @ LK_SVM](#detrac_0_59_100_0___lk_svm_)
        - [lost       @ detrac_0_59_100_0/LK_SVM](#lost___detrac_0_59_100_0_lk_svm_)
            - [24_48_64_128_64_48_24_bn       @ lost/detrac_0_59_100_0/LK_SVM](#24_48_64_128_64_48_24_bn___lost_detrac_0_59_100_0_lk_sv_m_)
                - [test       @ 24_48_64_128_64_48_24_bn/lost/detrac_0_59_100_0/LK_SVM](#test___24_48_64_128_64_48_24_bn_lost_detrac_0_59_100_0_lk_svm_)
                - [continue       @ 24_48_64_128_64_48_24_bn/lost/detrac_0_59_100_0/LK_SVM](#continue___24_48_64_128_64_48_24_bn_lost_detrac_0_59_100_0_lk_svm_)

<!-- /MarkdownTOC -->

<a id="build_"></a>
# build

cmake ..  -DPY_VER=3.6 -DPYTHON_LIBRARIES=/usr/lib/python3.6/dist-packages/numpy/core/include

<a id="getboxstatistic_s_"></a>
# getBoxStatistics

python3 getBoxStatistics.py root_dirs=/data/DETRAC/Images seq_paths=detrac_1_MVI_20011


python3 getBoxStatistics.py root_dirs=/data/DETRAC/Images,/data/GRAM/Images seq_paths=__n__ load_results=0 out_dir=../log/GRAM_DETRAC_all

<a id="visualize_"></a>
# visualize

<a id="detrac___visualiz_e_"></a>
## DETRAC       @ visualize

<a id="detections___detrac_visualize_"></a>
### detections       @ DETRAC/visualize

python3 visualize.py root_dir=/data/DETRAC/Images class_names_path=../labelling_tool/data/predefined_classes_vehicles.txt data_type=detections

<a id="only_boxes___detections_detrac_visualiz_e_"></a>
#### only_boxes       @ detections/DETRAC/visualize

python3 visualize.py root_dir=/data/DETRAC/Images seq_paths=detrac_44_MVI_40871 class_names_path=../labelling_tool/data/predefined_classes_vehicles.txt data_type=detections only_boxes=0 vis.write_frame_id=0 save=1 save_fmt=png


<a id="annotations___detrac_visualize_"></a>
### annotations       @ DETRAC/visualize

python3 visualize.py root_dir=/data/DETRAC/Images class_names_path=../labelling_tool/data/predefined_classes_vehicles.txt data_type=annotations

<a id="mai_n_"></a>
# main

 CUDA_VISIBLE_DEVICES=1 python3 main.py

<a id="lk_svm___main_"></a>
# LK_SVM       @ main

<a id="detrac_0_to_9_40_60___lk_svm_"></a>
## detrac_0_to_9_40_60       @ LK_SVM

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(10)" results_dir=log/detrac_0to9_40_60_lk @test seq_ids="range(10)" @data ratios.detrac="(0.4,0)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10 @train load=0

<a id="test_1___detrac_0_to_9_40_60_lk_svm_"></a>
### test_1       @ detrac_0_to_9_40_60/LK_SVM

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(10)" results_dir=log/detrac_0to9_40_60_lk @test seq_ids="range(10)" @data ratios.detrac="(0.4,-0.01)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10 @train load=1

<a id="5_5___detrac_0_to_9_40_60_lk_svm_"></a>
### 5_5       @ detrac_0_to_9_40_60/LK_SVM

CUDA_VISIBLE_DEVICES=1 python3 main.py @train seq_ids="range(10)" @test seq_ids="range(10)" @data ratios.detrac="(0.05,0.05)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=2

<a id="detrac_0_59_40_60___lk_svm_"></a>
## detrac_0_59_40_60       @ LK_SVM

CUDA_VISIBLE_DEVICES=2 python3 main.py @train seq_ids="range(60)" results_dir=log/detrac_0to59_40_60_lk @test seq_ids="range(60)" @data ratios.detrac="(0.4,0)" @trainer input.convert_to_gs=1 verbose=0 @@target.templates tracker=0 siamese.siam_fc.vis=0 count=10


<a id="detrac_0_59_100_0___lk_svm_"></a>
## detrac_0_59_100_0       @ LK_SVM

<a id="lost___detrac_0_59_100_0_lk_svm_"></a>
### lost       @ detrac_0_59_100_0/LK_SVM

<a id="24_48_64_128_64_48_24_bn___lost_detrac_0_59_100_0_lk_sv_m_"></a>
#### 24_48_64_128_64_48_24_bn       @ lost/detrac_0_59_100_0/LK_SVM

CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg=lk_svm,detrac_0_59_100_0,svm_batch_train:lost

<a id="test___24_48_64_128_64_48_24_bn_lost_detrac_0_59_100_0_lk_svm_"></a>
##### test       @ 24_48_64_128_64_48_24_bn/lost/detrac_0_59_100_0/LK_SVM

CUDA_VISIBLE_DEVICES=0 python3 main.py --cfg=detrac_0_59_100_0,mlp:lost:24_48_64_128_64_48_24:bn,mlp:active:24_48_64_128_64_48_24:bn @train load=1 results_dir=log/detrac_0_59_100_0_mlp_lk_async @data ratios.detrac=1,1 @tester vis=1 @trainer override=1 mode=0 @@target @@lost.mlp save_samples=0 batch.load_path=log/detrac_0_59_100_0_mlp_lk_async/lost/24_48_64_128_64_48_24_bn_batch_train/weights.pt.143 @@@ @@@active.mlp batch.load_path=log/detrac_0_59_100_0_mlp_lk_async/active/24_48_64_128_64_48_24_bn_batch_train/weights.pt.800 save_samples=0

CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg=detrac_0_59_100_0,mlp:lost:24_48_64_128_64_48_24:bn,mlp:active:24_48_64_128_64_48_24:bn,test

<a id="continue___24_48_64_128_64_48_24_bn_lost_detrac_0_59_100_0_lk_svm_"></a>
##### continue       @ 24_48_64_128_64_48_24_bn/lost/detrac_0_59_100_0/LK_SVM

CUDA_VISIBLE_DEVICES=2 python3 main.py --cfg=detrac_0_59_100_0,mlp:lost:24_48_64_128_64_48_24:bn,mlp:active:24_48_64_128_64_48_24:bn,continue:async

