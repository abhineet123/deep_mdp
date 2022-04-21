<!-- MarkdownTOC -->

- [detrac](#detra_c_)
    - [0_9_40_60       @ detrac](#0_9_40_60___detrac_)
- [mnistmot](#mnistmo_t_)
- [ctc](#ctc_)
- [ctmc](#ctm_c_)
    - [0_9_100_100       @ ctmc](#0_9_100_100___ctmc_)
        - [save       @ 0_9_100_100/ctmc](#save___0_9_100_100_ctmc_)
    - [10_46_100_100       @ ctmc](#10_46_100_100___ctmc_)
        - [save_no_show       @ 10_46_100_100/ctmc](#save_no_show___10_46_100_100_ctmc_)

<!-- /MarkdownTOC -->


<a id="detra_c_"></a>
# detrac

<a id="0_9_40_60___detrac_"></a>
## 0_9_40_60       @ detrac-->vis
python3 main.py cfg=_detrac_:s:0_9:d:40_60,_test_:vis:ann @test evaluate=0 load=1

<a id="mnistmo_t_"></a>
# mnistmot       
python3 main.py cfg=_mnistmot_:s:0_9:d:100_100,_test_:vis:ann @test evaluate=0 load=1 @tester use_annotations=0

<a id="ctc_"></a>
# ctc       
python3 main.py cfg=_ctc_:s:0_19:d:100_100,_test_:vis:ann:2 @test evaluate=0 load=1 @tester use_annotations=0

<a id="ctm_c_"></a>
# ctmc       
<a id="0_9_100_100___ctmc_"></a>
## 0_9_100_100       @ ctmc-->vis
python3 main.py cfg=_ctmc_:s:0_9:d:100_100,_test_:vis:ann @test evaluate=0 load=1 @tester use_annotations=0

<a id="save___0_9_100_100_ctmc_"></a>
### save       @ 0_9_100_100/ctmc-->vis

python3 main.py cfg=_ctmc_:s:0_9:d:100_100,_test_:vis:ann:2 @test evaluate=0 load=1 @tester use_annotations=0

<a id="10_46_100_100___ctmc_"></a>
## 10_46_100_100       @ ctmc-->vis

<a id="save_no_show___10_46_100_100_ctmc_"></a>
### save_no_show       @ 10_46_100_100/ctmc-->vis
python3 main.py cfg=_ctmc_:s:10_46:d:100_100,_test_:vis:ann:3 @test evaluate=0 load=1 @tester use_annotations=0