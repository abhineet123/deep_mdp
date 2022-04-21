@tester
	## mot
	save_debug_info=0
	##	
	use_annotations=1
	## traj
		### irange(30)
		min_trajectory_len=__name__
	##
	## trd
		### irange(30)+irange(34,74,10)
		tracked_as_detection=__name__
	##
	## no_gh
		### sort
		sort_targets=0
		### conflict
		resolve_conflicts=0
		### filter
		filter_detections=0	
	##	
	## no_th
		### lost
		lost_heuristics=0
		### tracked
		tracked_heuristics=0
		### reconnect
		reconnect_lost=0	
	##	
	## save_debug
	 ### 0,1,2
	 	save_debug_info=__name__
	## max_lost
		### irange(100)
		max_lost_ratio=__ratio__
	##

	## active_train
	train_active=1
	##

	@@input
		batch_mode=0

	@@@visualizer
		show=1
		mode=(1,1,1)
		pause_after_frame=1
		# lost_cols='none'
	@@@

	@@target
		## lk_gt
		templates.lk.use_gt=1
		## tracked
			### ign
			tracked.ignore_det=1
	@@@
	## vis0
	vis=0
	verbose=0
	@@target
		@@active
			verbose=0
		@@@lost
			verbose=0
			vis=0
			@@cnn
				vis=0
			@@@
		@@@tracked
			verbose=0
			vis=0
			@@cnn
				vis=0
			@@@
		@@@templates
			@@lk
				vis=0
			@@@	
		@@@
	@@@
	##
@

## vis
vis=1
@tester
	vis=1	
	@@input
		batch_mode=0
	@@@visualizer
		show=1
		mode=(1,1,1)
		### ann
		mode=(0,0,1)
		### det
		mode=(0,1,0)
		### ann_det
		mode=(0,1,1)
		### det
		pause_after_frame=1
		# lost_cols=('none',)
		### s
			#### 0
			disp_size=(320,240)
			#### 1
			disp_size=(640,480)
			#### 2
			disp_size=(1280,720)
			#### 3
			disp_size=(1920,1080)
		### r
			#### range(10)
			resize_factor=0.__name__
		### 2
			save=1
		### 3
			show=0
			save=1
@
##


@test
	## mot
	subseq_postfix=0
	## vis
		### 2
		evaluate=0	
	## load
	load=1
	## active_train
	save_prefix+="active_train"
	##
	## max_lost
		### range(100)
		save_prefix+="__full__"
	##
	## traj
		### irange(30)
		save_prefix+="__full__"
	##
	## trd
		### irange(30)+irange(34,74,10)
		save_prefix+="__full__"
	##
	# disable global heuristics
	## no_gh
	save_prefix+="__name__"
		### sort,conflict,filter
		save_prefix+="__name__"
	##	

	# disable target level heuristics
	## no_th
	save_prefix+="__name__"
		### lost,tracked,reconnect
		save_prefix+="__name__"
	##	

## lk_gt
	save_prefix+="__name__"	
## tracked
	### ign
	save_prefix+="__name__"
##

## glob
	save_prefix+="__name__"
##
@

## glob
@tester
	resolve_conflicts=1
	min_trajectory_len=5
##