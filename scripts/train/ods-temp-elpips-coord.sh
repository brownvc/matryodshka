#bin/bash
source /home/selenaling/anaconda3/bin/activate stereomag-gpu

python train.py \
--cameras_glob 'glob/train/ods/*.txt' \
--image_dir 'PATH/TO/REPLICA/360TrainData/' \
--max_steps 140000 \
--input_type ODS \
--which_loss elpips \
--coord_net \
--transform_inverse_reg \
--experiment_name temporal-elpips-coord \
--operation train \

