#bin/bash

source /home/selenaling/anaconda3/bin/activate stereomag-gpu-2

python train.py \
--cameras_glob 'PATH/TO/REALESTATE/TXT/GLOB' \
--image_dir 'PATH/TO/REALESTATE/IMAGE/DATASET' \
--max_steps 14000 \
--input_type REALESTATE_PP \
--which_loss elpips 
--coord_net \
--operation train \
--experiment_name realestate_wotemp_elpips_coord
