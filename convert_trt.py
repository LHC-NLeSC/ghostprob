from tensorflow.python.compiler.tensorrt import trt_convert as trt

converter = trt.TrtGraphConverterV2(input_saved_model_dir="./tf-models/ghost_nn")
converter.convert()
converter.save("./tf-models/ghost_nn_trt")

print("done")
