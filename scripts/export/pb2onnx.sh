#bin/bash
model=$1
source /home/selenaling/anaconda3/bin/activate stereomag36

python -m tf2onnx.convert --input export/$model.pb --output export/$model.onnx --inputs plane_sweep_input:0 --inputs-as-nchw plane_sweep_input:0 --outputs msi_output:0 --opset 9 --verbose --fold_const
