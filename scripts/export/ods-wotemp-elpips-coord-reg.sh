#bin/bash
source /home/selenaling/anaconda3/bin/activate stereomag

python export.py \
--coord_net \
--input_type ODS \
--net_only \
--rgba \
--model_name ods-wotemp-elpips-coord \
--pb_output matryodshka \
--operation export

