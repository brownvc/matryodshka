#bin/bash
source /home/selenaling/anaconda3/bin/activate stereomag
python eval.py \
--eval_type on_video \
--videos 'room_0 room_2' \
--model_name ods-wotemp-elpips-coord \
--output_table test/results/ods-wotemp-elpips-coord_video.json
