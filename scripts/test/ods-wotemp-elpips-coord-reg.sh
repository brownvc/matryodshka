#bin/bash
source /home/selenaling/anaconda3/bin/activate stereomag-gpu-2

python test.py \
--cameras_glob 'glob/test/reg/*.txt' \
--image_dir 'PATH/TO/REPLICA/360TestData/' \
--input_type ODS \
--experiment_name ods-wotemp-elpips-coord \
--coord_net
