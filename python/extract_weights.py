import argparse
import numpy as np
import onnx, onnx.numpy_helper
import json

def command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="ONNX file to extract weights from", type=str, required=True)
    # Parameters
    return parser.parse_args()

def layersort(layer):
    tokens = layer.name.split('/')
    if tokens[2] == "MatMul":
        return tokens[1] + '_0'
    elif tokens[2] == "BiasAdd":
        return tokens[1] + '_1'
    else:
        return tokens[1] + "_" + tokens[2]

arguments = command_line()

onnx_input = arguments.filename
model = onnx.load(onnx_input)
layer_list = []
for layer in sorted(model.graph.initializer, key=layersort):
    print(f"Serializing weights for layer {layer.name}...")
    tensor_weights = onnx.numpy_helper.to_array(layer)
    print(tensor_weights.shape)
    layer_list.append({"name": layer.name, "shape": tensor_weights.shape, "weights": tensor_weights.flatten().tolist()})

ofile = '.'.join(arguments.filename.split('.')[:-1] + ["json"])
with open(ofile, 'w') as of:
    json.dump(layer_list, of, indent=4)






#print(weight_list)
#[tensor] = [t for t in model.graph.initializer if t.name == "mobilenetv20_features_conv0_weight"]
#w = numpy_helper.to_array(tensor)