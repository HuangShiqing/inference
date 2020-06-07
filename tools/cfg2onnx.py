import re
import numpy as np

import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

def parse_cfg(file):
    with open(file, "r") as f:
        net = []
        line = f.readline()
        while True:
            flag_end = True
            if "[net]" in line:
                # print("[net]")
                image = {"name": "image", "width": None, "height": None}
                for sub_line in f:
                    if "width" in sub_line:
                        width = int(re.findall(r"\d+", sub_line)[0])
                        image["width"] = width
                    elif "height" in sub_line:
                        height = int(re.findall(r"\d+", sub_line)[0])
                        image["height"] = height
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(image)
            elif "[convolutional]" in line:
                # print("[convolutional]")
                convolutional = {"name": "convolutional", "batch_normalize": None, "filters": None, "size": None,
                                 "stride": None, "pad": None, "activation": None}
                flag_end = True
                for sub_line in f:
                    if "batch_normalize" in sub_line:
                        batch_normalize = int(re.findall(r"\d+", sub_line)[0])
                        convolutional["batch_normalize"] = batch_normalize
                    elif "filters" in sub_line:
                        filters = int(re.findall(r"\d+", sub_line)[0])
                        convolutional["filters"] = filters
                    elif "size" in sub_line:
                        size = int(re.findall(r"\d+", sub_line)[0])
                        convolutional["size"] = size
                    elif "stride" in sub_line:
                        stride = int(re.findall(r"\d+", sub_line)[0])
                        convolutional["stride"] = stride
                    elif "pad" in sub_line:
                        pad = int(re.findall(r"\d+", sub_line)[0])
                        convolutional["pad"] = pad
                    elif "activation" in sub_line:
                        activation = sub_line[sub_line.find('=') + 1:].strip()
                        convolutional["activation"] = activation
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(convolutional)
            elif "[maxpool]" in line:
                # print("[maxpool]")
                maxpool = {"name": "maxpool", "size": None, "stride": None}
                flag_end = True
                for sub_line in f:
                    if "size" in sub_line:
                        size = int(re.findall(r"\d+", sub_line)[0])
                        maxpool["size"] = size
                    elif "stride" in sub_line:
                        stride = int(re.findall(r"\d+", sub_line)[0])
                        maxpool["stride"] = stride
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(maxpool)
            elif "[yolo]" in line:
                # print("[yolo]")
                yolo = {"name": "yolo", "mask": None, "anchors": None, "classes": None, "num": None}
                flag_end = True
                for sub_line in f:
                    if "mask" in sub_line:
                        mask = (list(filter(str.isdigit, sub_line)))
                        mask = [int(i) for i in mask]
                        yolo["mask"] = mask
                    elif "anchors" in sub_line:
                        anchors = re.findall(r"\d+", sub_line)
                        anchors = [int(i) for i in anchors]
                        yolo["anchors"] = anchors
                    elif "classes" in sub_line:
                        classes = int(re.findall(r"\d+", sub_line)[0])
                        yolo["classes"] = classes
                    elif "num" in sub_line:
                        num = int(re.findall(r"\d+", sub_line)[0])
                        yolo["num"] = num
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(yolo)
            elif "[route]" in line:
                # print("[route]")
                route = {"name": "route", "layers": None}
                flag_end = True
                for sub_line in f:
                    if "layers" in sub_line:
                        layers = re.findall(r"\-?\d+", sub_line)
                        layers = [int(i) for i in layers]
                        route["layers"] = layers
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(route)
            elif "[upsample]" in line:
                # print("[upsample]")
                route = {"name": "upsample", "stride": None}
                flag_end = True
                for sub_line in f:
                    if "stride" in sub_line:
                        stride = int(re.findall(r"\d+", sub_line)[0])
                        route["stride"] = stride
                    elif "[" in sub_line and "]" in sub_line:
                        line = sub_line
                        flag_end = False
                        break
                net.append(route)
            if flag_end == True:
                break
        print("finish parese cfg")
        return net

def cfg2onnx(net, file_weights,file_onnx):
    f = open(file_weights, "rb")
    major, minor, revision = np.ndarray(shape=(3,),dtype='int32',buffer=f.read(12))
    if (major*10+minor)>=2 and major<1000 and minor<1000:
        seen = np.ndarray(shape=(1,),dtype='int64',buffer=f.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=f.read(4))
    # print("weights header: ",major,minor,revision)

    layer_index = 0
    nodes = [] # 保存所有层用到的节点
    inputs = [] # 保存所有层的输入
    initializer = [] #保存所有层的输入的值
    outputs = [] # 保存模型的输出
    last_node_output = None #由于有些层不只一个node,因此该参数与layer_output_name[-1]的区别在于该参数是上一个node的输出
    layer_output_name = []#保存每一层的输出的名字,给route层用
    layer_output_size = []#保存每一层的输出的size,给route层用

    for layer in net:
        layer_index += 1
        if layer["name"] == "image":
            layer_index -= 2
            image = helper.make_tensor_value_info('image', TensorProto.FLOAT, [1, 3, layer["width"], layer["height"]])

            layer_output_name.append("image")
            layer_output_size.append([1, 3, layer["width"], layer["height"]])

            inputs.append(image)
        elif layer["name"] == "convolutional":
            padding = 0
            if layer["pad"]:
                padding = layer["size"] // 2
            out_h = int((layer_output_size[-1][2] + 2 * padding - layer["size"]) / layer["stride"] + 1)
            out_w = int((layer_output_size[-1][3] + 2 * padding - layer["size"]) / layer["stride"] + 1)

            layer_Conv_B = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Conv_B", TensorProto.FLOAT,[layer["filters"]])
            layer_Conv_B_initializer = numpy_helper.from_array(np.ndarray(shape=(layer["filters"]), dtype='float32', buffer=f.read(4 * layer["filters"])),"layer_" + str(layer_index) + "_Conv_B")
            if layer["batch_normalize"] == 1:
                layer_Bn_scale = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Bn_scale", TensorProto.FLOAT,[layer["filters"]])
                layer_Bn_scale_initializer = numpy_helper.from_array(np.ndarray(shape=(layer["filters"]), dtype='float32', buffer=f.read(4 * layer["filters"])),"layer_" + str(layer_index) + "_Bn_scale")
                layer_Bn_mean = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Bn_mean",TensorProto.FLOAT, [layer["filters"]])
                layer_Bn_mean_initializer = numpy_helper.from_array(np.ndarray(shape=(layer["filters"]), dtype='float32', buffer=f.read(4 * layer["filters"])),"layer_" + str(layer_index) + "_Bn_mean")
                layer_Bn_var = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Bn_var",TensorProto.FLOAT, [layer["filters"]])
                layer_Bn_var_initializer = numpy_helper.from_array(np.ndarray(shape=(layer["filters"]), dtype='float32', buffer=f.read(4 * layer["filters"])),"layer_" + str(layer_index) + "_Bn_var")
                layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Bn_Y",TensorProto.FLOAT, [1, layer["filters"], out_h, out_w])

            layer_Conv_W = helper.make_tensor_value_info("layer_"+str(layer_index)+"_Conv_W", TensorProto.FLOAT, [layer["filters"], layer_output_size[-1][1], layer["size"], layer["size"]])
            layer_Conv_W_initializer = numpy_helper.from_array(np.ndarray(shape=(layer["filters"], layer_output_size[-1][1], layer["size"], layer["size"]),dtype='float32',buffer=f.read(4*layer["filters"]*layer_output_size[-1][1]*layer["size"]*layer["size"])), "layer_"+str(layer_index)+"_Conv_W")
            layer_Y = helper.make_tensor_value_info("layer_"+str(layer_index)+"_Conv_Y", TensorProto.FLOAT, [1, layer["filters"], out_h, out_w])
            node_layer_Conv = helper.make_node(
                "Conv",
                [layer_output_name[-1], "layer_"+str(layer_index)+"_Conv_W", "layer_"+str(layer_index)+"_Conv_B"] if layer["batch_normalize"] != 1 else [layer_output_name[-1], "layer_"+str(layer_index)+"_Conv_W"],
                ["layer_"+str(layer_index)+"_Conv_Y"],
                kernel_shape=[layer["size"], layer["size"]],
                strides=[layer["stride"], layer["stride"]],
                pads=[padding, padding, padding, padding])

            inputs.append(layer_Conv_W)
            initializer.append(layer_Conv_W_initializer)
            if layer["batch_normalize"] != 1:
                inputs.append(layer_Conv_B)
                initializer.append(layer_Conv_B_initializer)
            nodes.append(node_layer_Conv)
            last_node_output = "layer_"+str(layer_index)+"_Conv_Y"

            if layer["batch_normalize"] == 1:
                node_layer_Bn = onnx.helper.make_node(
                    'BatchNormalization',
                    inputs=["layer_"+str(layer_index)+"_Conv_Y", "layer_" + str(layer_index) + "_Bn_scale", "layer_"+str(layer_index)+"_Conv_B", "layer_"+str(layer_index)+"_Bn_mean","layer_" + str(layer_index) + "_Bn_var"],
                    outputs=["layer_" + str(layer_index) + "_Bn_Y"],
                )
                inputs.append(layer_Bn_scale)
                initializer.append(layer_Bn_scale_initializer)
                inputs.append(layer_Conv_B)
                initializer.append(layer_Conv_B_initializer)
                inputs.append(layer_Bn_mean)
                initializer.append(layer_Bn_mean_initializer)
                inputs.append(layer_Bn_var)
                initializer.append(layer_Bn_var_initializer)
                nodes.append(node_layer_Bn)
                last_node_output = "layer_" + str(layer_index) + "_Bn_Y"

            if layer["activation"] == "leaky":
                layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_leaky_Y", TensorProto.FLOAT, [1, layer["filters"], out_h, out_w])
                node_layer_LeakyRelu = onnx.helper.make_node(
                    'LeakyRelu',
                    inputs=[last_node_output],
                    outputs=["layer_" + str(layer_index) + "_leaky_Y"],
                    alpha=0.1
                )
                nodes.append(node_layer_LeakyRelu)
                last_node_output = "layer_" + str(layer_index) + "_leaky_Y"
            elif layer["activation"] == "linear":
                pass

            if layer_index == 0:
                layer_output_name = layer_output_name[1:]
                layer_output_size = layer_output_size[1:]
            layer_output_name.append(last_node_output)
            layer_output_size.append([1, layer["filters"], out_h, out_w])
        elif layer["name"] == "maxpool":
            # out_w = (last_layer_width + 2 * padding) / layer["stride"]
            out_w = layer_output_size[-1][3] // layer["stride"]
            out_h = layer_output_size[-1][2] // layer["stride"]

            layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Maxpool_Y", TensorProto.FLOAT,[1, layer_output_size[-1][1], out_h, out_w])
            node_layer_MaxPool = onnx.helper.make_node(
                'MaxPool',
                inputs=[layer_output_name[-1]],
                outputs=["layer_" + str(layer_index) + "_Maxpool_Y"],
                kernel_shape=[layer["size"], layer["size"]],
                strides=[layer["stride"], layer["stride"]]
            )
            nodes.append(node_layer_MaxPool)
            last_node_output = "layer_" + str(layer_index) + "_Maxpool_Y"
            layer_output_name.append(last_node_output)
            layer_output_size.append([1, layer_output_size[-1][1], out_h, out_w])
        elif layer["name"] == "yolo":
            layer_Yolo_mask = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Yolo_mask", TensorProto.FLOAT,[3])
            layer_Yolo_mask_initializer = numpy_helper.from_array(np.array(layer["mask"],dtype='int32'),"layer_" + str(layer_index) + "_Yolo_mask")
            layer_Yolo_anchors = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Yolo_anchors", TensorProto.FLOAT,[layer["num"]*2])
            layer_Yolo_anchors_initializer = numpy_helper.from_array(np.array(layer["anchors"], dtype='int32'),"layer_" + str(layer_index) + "_Yolo_anchors")
            layer_Yolo_classes = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Yolo_classes", TensorProto.FLOAT,[1])
            layer_Yolo_classes_initializer = numpy_helper.from_array(np.array(layer["classes"], dtype='int32'),"layer_" + str(layer_index) + "_Yolo_classes")
            layer_Yolo_num = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Yolo_num", TensorProto.FLOAT,[1])
            layer_Yolo_num_initializer = numpy_helper.from_array(np.array(layer["num"], dtype='int32'),"layer_" + str(layer_index) + "_Yolo_num")
            layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Yolo_Y", TensorProto.FLOAT,[1, layer_output_size[-1][1], layer_output_size[-1][2], layer_output_size[-1][3]])
            node_layer_Yolo = helper.make_node(
                "Yolo",
                [layer_output_name[-1], "layer_" + str(layer_index) + "_Yolo_mask","layer_" + str(layer_index) + "_Yolo_anchors","layer_" + str(layer_index) + "_Yolo_classes","layer_" + str(layer_index) + "_Yolo_num"],
                ["layer_" + str(layer_index) + "_Yolo_Y"])

            inputs.append(layer_Yolo_mask)
            initializer.append(layer_Yolo_mask_initializer)
            inputs.append(layer_Yolo_anchors)
            initializer.append(layer_Yolo_anchors_initializer)
            inputs.append(layer_Yolo_classes)
            initializer.append(layer_Yolo_classes_initializer)
            inputs.append(layer_Yolo_num)
            initializer.append(layer_Yolo_num_initializer)
            nodes.append(node_layer_Yolo)
            outputs.append(layer_Y)

            last_node_output = "layer_" + str(layer_index) + "_Yolo_Y"
            layer_output_name.append(last_node_output)
            layer_output_size.append(layer_output_size[-1])
        elif layer["name"] == "route":
            out_c = 0
            for i in layer["layers"]:
                out_c += layer_output_size[i][1]

            layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Concat_Y", TensorProto.FLOAT,[1, out_c, layer_output_size[layer["layers"][0]][2], layer_output_size[layer["layers"][0]][3]])
            node_layer_Concat = helper.make_node(
                "Concat",
                [layer_output_name[i] for i in layer["layers"]],
                ["layer_" + str(layer_index) + "_Concat_Y"])
            nodes.append(node_layer_Concat)

            last_node_output = "layer_" + str(layer_index) + "_Concat_Y"
            layer_output_name.append(last_node_output)
            layer_output_size.append([1, out_c, layer_output_size[layer["layers"][0]][2], layer_output_size[layer["layers"][0]][3]])
        elif layer["name"] == "upsample":
            out_c = layer_output_size[-1][1]
            out_h =layer_output_size[-1][2]*layer["stride"]
            out_w =layer_output_size[-1][3]*layer["stride"]

            layer_Upsample_scales = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Upsample_scales",TensorProto.FLOAT, [4])
            layer_Upsample_scales_initializer = numpy_helper.from_array(np.array([1,1,layer["stride"],layer["stride"]], dtype='int32'),"layer_" + str(layer_index) + "_Upsample_scales")
            layer_Y = helper.make_tensor_value_info("layer_" + str(layer_index) + "_Upsample_Y", TensorProto.FLOAT,[1, out_c, out_h,out_w])
            node_layer_Upsample = onnx.helper.make_node(
                'Upsample',
                inputs=[layer_output_name[-1], "layer_" + str(layer_index) + "_Upsample_scales"],
                outputs=["layer_" + str(layer_index) + "_Upsample_Y"],
                mode='nearest',
            )

            inputs.append(layer_Upsample_scales)
            initializer.append(layer_Upsample_scales_initializer)
            nodes.append(node_layer_Upsample)

            last_node_output = "layer_" + str(layer_index) + "_Upsample_Y"
            layer_output_name.append(last_node_output)
            layer_output_size.append([1,out_c,out_h,out_w])
    f.close()

    graph_def = helper.make_graph(
        nodes,
        file_onnx,
        inputs,
        outputs,
        initializer
    )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    # onnx.checker.check_model(model_def)# 由于有自定义层yolo层因此不能检查
    onnx.save(model_def, file_onnx)


if __name__ == '__main__':
    net = parse_cfg("./resource/yolov3-tiny.cfg")
    cfg2onnx(net,"./resource/yolov3-tiny_120000.weights","./resource/yolov3_tiny.onnx")
    print("finish change cfg")

