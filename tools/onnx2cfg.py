import re
import numpy as np
import struct

import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto


def save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm, file_weights):
    if last_conv_node:
        if last_conv_bn_node:
            np.array(initializer_parm[last_conv_bn_node.input[2]], dtype='float32').tofile(file_weights)  # b
            np.array(initializer_parm[last_conv_bn_node.input[1]], dtype='float32').tofile(file_weights)  # scale
            np.array(initializer_parm[last_conv_bn_node.input[3]], dtype='float32').tofile(file_weights)  # mean
            np.array(initializer_parm[last_conv_bn_node.input[4]], dtype='float32').tofile(file_weights)  # var
            np.array(initializer_parm[last_conv_node.input[1]], dtype='float32').tofile(file_weights)  # w
        else:
            np.array(initializer_parm[last_conv_node.input[2]], dtype='float32').tofile(file_weights)
            np.array(initializer_parm[last_conv_node.input[1]], dtype='float32').tofile(file_weights)
        last_conv_node = None
        last_conv_bn_node = None
    return last_conv_node, last_conv_bn_node


if __name__ == "__main__":
    onnx_name = "yolov3-tiny.onnx"
    cfg_name = "yolov3-tiny-back.cfg"

    model = onnx.load("../resource/" + onnx_name)
    file_cfg = open("../resource/" + cfg_name, "w")
    file_weights = open("../resource/" + cfg_name.split('.')[0] + ".weights", "w")
    initializer_shape = {}  # 用于保存卷积层的filters
    initializer_parm = {}  # 用于保存要写进weights的参数值
    data_type = {1: "float32", 6: "int32", 7: "int64"}
    for initializer in model.graph.initializer:
        initializer_shape[initializer.name] = initializer.dims
        # 数据可能会保存在raw_date里也可能会保存在float_data
        try:
            initializer_parm[initializer.name] = np.ndarray(shape=initializer.dims,
                                                            dtype=data_type[initializer.data_type],
                                                            buffer=initializer.raw_data)
        except:
            initializer_parm[initializer.name] = np.array(list(initializer.float_data), dtype='float32')

    # weights文件写版本信息
    major = 1
    minor = 1
    revision = 1
    seen = 1000
    np.array(major, dtype='int32').tofile(file_weights)
    np.array(minor, dtype='int32').tofile(file_weights)
    np.array(revision, dtype='int32').tofile(file_weights)
    np.array(seen, dtype='int64').tofile(file_weights)

    # cfg文件写net信息
    width = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    height = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    channels = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    node_info = "[net]" \
                "\nwidth={}" \
                "\nheight={}" \
                "\nchannels={}" \
                "\n" \
                "\n".format(width, height, channels)
    file_cfg.writelines(node_info)

    last_conv_info = ""
    last_conv_node = None
    last_conv_bn_node = None
    for node in model.graph.node:
        if node.op_type == "Conv":
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            last_conv_node = node
            node_dict = {"kernel_shape": None, "pads": None, "strides": None}
            attrs = node.attribute
            for attr in attrs:
                if attr.name == "kernel_shape":
                    node_dict["kernel_shape"] = [attr.ints[i] for i in range(2)]
                elif attr.name == "pads":
                    node_dict["pads"] = [attr.ints[i] for i in range(4)]
                elif attr.name == "strides":
                    node_dict["strides"] = [attr.ints[i] for i in range(2)]
            # darknet的pad=1代表padding=size//2,onnx的pads=darknet的padding, padding is used to compute the out_size
            if node_dict["pads"][0] == 0:
                pass
            else:
                node_dict["pads"] = [1, 1, 1, 1]
            inputs = node.input[1]
            last_conv_info = "[convolutional]" \
                             "\nbatch_normalize=0" \
                             "\nfilters={}" \
                             "\nsize={}" \
                             "\nstride={}" \
                             "\npad={}" \
                             "\nactivation=linear" \
                             "\n" \
                             "\n".format(initializer_shape[node.input[1]][0], node_dict["kernel_shape"][0],
                                         node_dict["strides"][0], node_dict["pads"][0])
        elif node.op_type == "BatchNormalization":
            last_conv_bn_node = node
            last_conv_info = last_conv_info.replace("batch_normalize=0", "batch_normalize=1")
        elif node.op_type in ["Relu", "LeakyRelu"]:
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            if node.op_type == "Relu":
                last_conv_info = last_conv_info.replace("activation=linear", "activation=relu")
            elif node.op_type == "LeakyRelu":
                last_conv_info = last_conv_info.replace("activation=linear", "activation=leaky")
            else:
                print("not support activate yet")
                exit()
            file_cfg.writelines(last_conv_info)
            last_conv_info = ""
        elif node.op_type == "MaxPool":
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            node_dict = {"kernel_shape": None, "strides": None}
            attrs = node.attribute
            for attr in attrs:
                if attr.name == "kernel_shape":
                    node_dict["kernel_shape"] = [attr.ints[i] for i in range(2)]
                elif attr.name == "strides":
                    node_dict["strides"] = [attr.ints[i] for i in range(2)]
            node_info = "[maxpool]" \
                        "\nsize={}" \
                        "\nstride={}" \
                        "\n" \
                        "\n".format(node_dict["kernel_shape"][0], node_dict["strides"][0])
            file_cfg.writelines(last_conv_info + node_info)
            last_conv_info = ""
        elif node.op_type == "Upsample":
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            inputs = node.input[1]
            node_info = "[upsample]" \
                        "\nstride={}" \
                        "\n" \
                        "\n".format(initializer_parm[inputs][-1])
            file_cfg.writelines(last_conv_info + node_info)
            last_conv_info = ""
        elif node.op_type == "Concat":
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            node_dict = {"layers": ""}
            for i in node.input:
                node_dict["layers"] += str(int(re.findall(r"\d+", i)[0]) - int(re.findall(r"\d+", node.output[0])[0]))
                node_dict["layers"] += ","
            node_dict["layers"] = node_dict["layers"].strip(',')
            node_info = "[route]" \
                        "\nlayers = {}" \
                        "\n" \
                        "\n".format(node_dict["layers"])
            file_cfg.writelines(last_conv_info + node_info)
            last_conv_info = ""
        elif node.op_type == "Yolo":
            last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                                    file_weights)
            node_dict = {"mask": None, "anchors": None, "classes": None, "num": None}
            inputs = node.input
            node_dict["mask"] = "".join(str(x) + ',' for x in initializer_parm[inputs[1]]).strip(',')
            node_dict["anchors"] = "".join(str(x) + ',' for x in initializer_parm[inputs[2]]).strip(',')
            node_dict["classes"] = str(initializer_parm[inputs[3]])
            node_dict["num"] = str(initializer_parm[inputs[4]])

            node_info = "[yolo]" \
                        "\nmask = {}" \
                        "\nanchors = {}" \
                        "\nclasses = {}" \
                        "\nnum = {}" \
                        "\n" \
                        "\n".format(node_dict["mask"], node_dict["anchors"], node_dict["classes"], node_dict["num"])
            file_cfg.writelines(last_conv_info + node_info)
            last_conv_info = ""
        else:
            print("not support such node type yet")
            exit()
    last_conv_node, last_conv_bn_node = save_last_conv_parm(last_conv_node, last_conv_bn_node, initializer_parm,
                                                            file_weights)
    file_cfg.close()
    file_weights.close()
    print("change onnx to cfg is succeed")
    exit()

    # old version
    # 根据层的类型往weights里写入参数
    # layer_type = ''
    # layer_index = '0'
    # for key in initializer_parm:
    #     successful = False
    #     while not successful:
    #         if str(layer_index) in key:
    #             if "Conv" in key:
    #                 if layer_type == '':
    #                     layer_type = "Conv"
    #             elif "Bn" in key:
    #                 if layer_type == ('' or "Conv"):
    #                     layer_type = "Conv_Bn"
    #             successful = True
    #         else:
    #             if layer_type == "Conv":
    #                 np.array(initializer_parm["layer_{}_Conv_B".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 np.array(initializer_parm["layer_{}_Conv_W".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 layer_type = ''
    #             elif layer_type == "Conv_Bn":
    #                 np.array(initializer_parm["layer_{}_Conv_B".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 np.array(initializer_parm["layer_{}_Bn_scale".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 np.array(initializer_parm["layer_{}_Bn_mean".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 np.array(initializer_parm["layer_{}_Bn_var".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 np.array(initializer_parm["layer_{}_Conv_W".format(layer_index)], dtype='float32').tofile(
    #                     file_weights)
    #                 layer_type = ''
    #             layer_index = re.findall(r"\d+", key)[0]
    # # 写入最后一层的参数
    # if layer_type == "Conv":
    #     np.array(initializer_parm["layer_{}_Conv_B".format(layer_index)], dtype='float32').tofile(file_weights)
    #     np.array(initializer_parm["layer_{}_Conv_W".format(layer_index)], dtype='float32').tofile(file_weights)
    #     layer_type = ''
    # elif layer_type == "Conv_Bn":
    #     np.array(initializer_parm["layer_{}_Conv_B".format(layer_index)], dtype='float32').tofile(file_weights)
    #     np.array(initializer_parm["layer_{}_Bn_scale".format(layer_index)], dtype='float32').tofile(file_weights)
    #     np.array(initializer_parm["layer_{}_Bn_mean".format(layer_index)], dtype='float32').tofile(file_weights)
    #     np.array(initializer_parm["layer_{}_Bn_var".format(layer_index)], dtype='float32').tofile(file_weights)
    #     np.array(initializer_parm["layer_{}_Conv_W".format(layer_index)], dtype='float32').tofile(file_weights)
    #     layer_type = ''
