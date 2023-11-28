import onnx
from onnxsim import simplify
from onnxconverter_common import float16

# load your predefined ONNX model
model = onnx.load("onnx_output/VGG-BiLSTM-CTC-gray-20231127.onnx")
# convert model
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

# save model
onnx.save(model_simp, "onnx_output/VGG-BiLSTM-CTC-gray-20231127-sim.onnx")

model = onnx.load("onnx_output/VGG-BiLSTM-CTC-gray-20231127-sim.onnx")
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "onnx_output/VGG-BiLSTM-CTC-gray-20231127-sim-fp16.onnx")
