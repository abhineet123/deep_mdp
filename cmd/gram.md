# detections

python3 tf_api_eval.py frozen_graph_path=trained/isl_intersection_model/frozen_inference_graph.pb labels_path=data/vehicle.pbtxt n_frames=0 batch_size=8 show_img=0 n_classes=1  root_dir=/data/GRAM/Images seq_paths=gram_all.txt sampling_ratio=1.0 random_sampling=0 sleep_time=10  eval_every=0 write_summary=0 write_summary=1 save_det=1 load_det=0 trained_checkpoint_prefix=model.ckpt.data-00000-of-00001 background_path=/data/GRAM/Backgrounds save_dir=/data/GRAM/Detections load_dir=/data/GRAM/Detections

## min_score_thresh=0.01       @ detections

python3 tf_api_eval.py frozen_graph_path=trained/isl_intersection_model/frozen_inference_graph.pb labels_path=data/vehicle.pbtxt n_frames=0 batch_size=8 show_img=0 n_classes=1  root_dir=/data/GRAM/Images seq_paths=gram_all.txt sampling_ratio=1.0 random_sampling=0 sleep_time=10  eval_every=0 write_summary=0 write_summary=1 save_det=1 load_det=0 trained_checkpoint_prefix=model.ckpt.data-00000-of-00001 background_path=/data/GRAM/Backgrounds save_dir=/data/GRAM/Detections load_dir=/data/GRAM/Detections min_score_thresh=0.01


## to_mot       @ detections

python3 csv_to_mot.py img_root_dir=/data/GRAM/Images root_dir=/data/GRAM/Detections 

