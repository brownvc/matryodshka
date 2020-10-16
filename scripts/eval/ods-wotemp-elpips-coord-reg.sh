#bin/bash
source /home/selenaling/anaconda3/bin/activate stereomag

python eval.py \
--model_name ods-wotemp-elpips-coord \
--output_table test/results/ods-wotemp-elpips-coord.json
