from onnx import *
from onnx import optimizer

if __name__ == '__main__':
    all_passes = optimizer.get_available_passes()
    passes = ['fuse_bn_into_conv']
    original_model = onnx.load("../resource/yolov3-tiny.onnx")
    optimized_model = optimizer.optimize(original_model, passes)
    onnx.save(optimized_model, "../resource/yolov3-tiny_new.onnx")