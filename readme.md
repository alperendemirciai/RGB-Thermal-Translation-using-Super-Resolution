

cl commands for training and testing: 

python test.py --data ./labeled_rgbt_pairs --save_dir ./outputs/13th_og_run --batch_size 32 --upsampling_method pixelshuffle --gen_filters 32 --random_state 42 --thermal_type thermal8 --sr color

python train_val.py --data ./labeled_rgbt_pairs --save_dir ./outputs/13th_sr3x_run --lr 0.0005 --batch_size 32 --epochs 100 --print_every 2 --upsampling_method pixelshuffle --gen_filters 32 --disc_filters 32 --random_state 42 --thermal_type thermal8 --sr sr_3x --save_freq 3