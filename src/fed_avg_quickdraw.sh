python run_fed_avg.py --data_path $3 \
--checkpoint_dir $4 \
--classifier protonets --num_local_epochs 10 --iterations 120 \
--num_clients $1 --num_classes 35 --shots_per_client $2 \
--feature_extractor BiT-M-R50x1-FILM \
--random_seed 1994 --dataset quickdraw --use_npy_data \
--learning_rate 0.006
