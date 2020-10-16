#bin/bash

source /home/selenaling/anaconda3/bin/activate stereomag-gpu-2

python train.py \
--cameras_glob 'glob/train/pp/*.txt' \
--image_dir 'PATH/TO/REPLICA/CubemapData/' \
--max_steps 14000 \
--input_type PP \
--which_loss elpips 
--coord_net \
--operation train \
--experiment_name pp_wotemp_elpips_coord
