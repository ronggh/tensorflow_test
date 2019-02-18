python D:/Tensorflow1.12_src/hub-master/examples/image_retraining/retrain.py ^
--bottleneck_dir ../tf_out/retrain/bottleneck ^
--how_many_training_steps 200 ^
--model_dir ../tf_out/retrain/inception_model/ ^
--output_graph ../tf_out/retrain/output_graph.pb ^
--output_labels ../tf_out/retrain/output_labels.txt ^
--image_dir ../tf_in/retrain/data/train/
pause
