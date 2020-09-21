#include "network.h"
#include "parser.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"
#include "upsample_layer.h"
#include "route_layer.h"
#include "yolo_layer.h"

// layer_num必须严格等于网络实际的层数
network *init_net(int layer_num, int h, int w, int c) {
	network *net = make_network(layer_num);

	net->h = h;
	net->w = w;
	net->c = c;
	net->batch = 1;
	net->inputs = h * w * c;
	net->input = calloc(net->inputs * net->batch, sizeof(float));
#ifdef GPU
	net->gpu_index = gpu_index;
//	net->output_gpu = out.output_gpu;
	net->input_gpu = cuda_make_array(net->input, net->inputs * net->batch);
//	net->truth_gpu = cuda_make_array(net->truth, net->truths*net->batch);
#endif
	fprintf(stderr,
			"layer     filters    size              input                output\n");
	return net;
}

void finish_net(network *net) {
	//part1.set net's workspace
	//part2.set net's outputs and output

	//part1
	size_t workspace_size = 0;
	int i;
	for (i = net->n - 1; i >= 0; --i) {
		if (net->layers[i].workspace_size > workspace_size) {
			workspace_size = net->layers[i].workspace_size;
		}
	}

	if (workspace_size) {
		//printf("%ld\n", workspace_size);
#ifdef GPU
		if (gpu_index >= 0) {
			net->workspace = cuda_make_array(0,
					(workspace_size - 1) / sizeof(float) + 1);
		} else {
			net->workspace = calloc(1, workspace_size);
		}
#else
		net->workspace = calloc(1, workspace_size);
#endif
	}

	//part2
	layer out = get_network_output_layer(net);
	net->outputs = out.outputs;
	net->output = out.output;
#ifdef GPU
	net->output_gpu = out.output_gpu;
#endif

}

//TODO: 把padding换成SAME或者VALID
int Conv2d(network *net, int filter, int size, int stride, int padding,
		int activation, int batch_normalize, int quantize, int layer_index) {
	fprintf(stderr, "%5d ", layer_index);
	int binary = 0, xnor = 0, adam = 0;
	int h, w, c;
	layer l = { 0 };

	if (layer_index == 0) {
		h = net->h;
		w = net->w;
		c = net->c;
	} else {
		h = net->layers[layer_index - 1].out_h;
		w = net->layers[layer_index - 1].out_w;
		c = net->layers[layer_index - 1].out_c;
	}

	l = make_convolutional_layer(1, h, w, c, filter, 1, size, stride, padding,
			activation, batch_normalize, quantize, xnor, adam);
	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

int DenseLayer(network *net, int output, ACTIVATION activation,
		int batch_normalize, int layer_index) {
	fprintf(stderr, "%5d ", layer_index);
	//TODO:should add workspace?
	int adam = 0;
	int inputs;
	layer l = { 0 };

	if (layer_index == 0) {
		inputs = net->inputs;
	} else {
		inputs = net->layers[layer_index - 1].outputs;
	}

	l = make_connected_layer(1, inputs, output, activation, batch_normalize,
			adam);
	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

int MaxPool(network *net, int size, int stride, int padding, int layer_index) {
	fprintf(stderr, "%5d ", layer_index);
	int h, w, c;
	layer l = { 0 };

	h = net->layers[layer_index - 1].out_h;
	w = net->layers[layer_index - 1].out_w;
	c = net->layers[layer_index - 1].out_c;

	l = make_maxpool_layer(1, h, w, c, size, stride, padding);
	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

int AvgPool(network *net, int layer_index) {
	fprintf(stderr, "%5d ", layer_index);
	int h, w, c;
	layer l = { 0 };

	h = net->layers[layer_index - 1].out_h;
	w = net->layers[layer_index - 1].out_w;
	c = net->layers[layer_index - 1].out_c;

	l = make_avgpool_layer(1, w, h, c);
	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

int UpsampleLayer(network *net, int stride, int scale, int layer_index)
{
	fprintf(stderr, "%5d ", layer_index);
	layer l = { 0 };
	
	int h = net->layers[layer_index - 1].out_h;
	int w = net->layers[layer_index - 1].out_w;
	int c = net->layers[layer_index - 1].out_c;

	l = make_upsample_layer(1, w, h, c, stride);
	// TODO: 优化这里的scale=1时，在forward_upsample_layer里就不需要乘了
	l.scale = scale;
	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

int RouteLayer(network *net, char *layers, int layer_index)
{
	fprintf(stderr, "%5d ", layer_index);
	int len = strlen(layers);
    if(!layers) error("Route Layer must specify input layers");
    int n = 1;//计算有几个输入层
    int i;
    for(i = 0; i < len; ++i){
        if (layers[i] == ',') ++n;
    }

    int *input_layers = calloc(n, sizeof(int));//指向需要进行concat操作的层的序号(绝对序号)
    int *input_sizes = calloc(n, sizeof(int));//指向需要进行concat操作的层的输出尺寸
    for(i = 0; i < n; ++i){
        int index = atoi(layers);
        layers = strchr(layers, ',')+1;
        if(index < 0) index = layer_index + index;
        input_layers[i] = index;
        input_sizes[i] = net->layers[index].outputs;
    }

	layer l = { 0 };
	l = make_route_layer(1, n, input_layers, input_sizes);
	//计算输出的whc
	convolutional_layer first = net->layers[input_layers[0]];
    l.out_w = first.out_w;
    l.out_h = first.out_h;
    l.out_c = first.out_c;
    for(i = 1; i < n; ++i){
        int index = input_layers[i];
        convolutional_layer next = net->layers[index];
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l.out_c += next.out_c;
        }else{
            l.out_h = l.out_w = l.out_c = 0;
        }
    }

	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}

// total=cfg文件yolo层的num,为总共的anchor个数，一般为9;
// anchor为anchor的wh信息的;
// mask_char为描述当前yolo所分配的anchor, 一般最后一个yolo层的l.mask=[0，1，2], 这里用"0,1,2"; 
int YoloLayer(network *net, int total, char *anchor, char *mask_char, int classes, int layer_index)
{
	fprintf(stderr, "%5d ", layer_index);
	layer l = { 0 };
	int h = net->layers[layer_index - 1].out_h;
	int w = net->layers[layer_index - 1].out_w;
	// n为与每个yolo层分配的anchor个数一样,一般为3;
	int n = total;//num初始设置为total没啥用，下面会根据mask的数量变的，一般n为3
	int *mask = parse_yolo_mask(mask_char, &n);
	l = make_yolo_layer(1, w, h, n, total, mask, classes);
	//将anchor字符串转float存储到biases中
	if(anchor){
        int len = strlen(anchor);
        int n = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (anchor[i] == ',') ++n;
        }
        for(i = 0; i < n; ++i){
            float bias = atof(anchor);
            // 可以看出yolo层的biases用来存储anchor的wh信息的
            l.biases[i] = bias;
            anchor = strchr(anchor, ',')+1;
        }
    }
	// l.max_boxes = ;//训练用的参数
	// l.jitter = ;//训练用的参数
	// l.ignore_thresh = ;//训练用的参数
	// l.truth_thresh = ;//训练用的参数
	// l.random = ;//好像没用到的参数

	net->layers[layer_index] = l;
	layer_index++;
	return layer_index;
}
/*
**查看输出示例代码
**float *net_outputs = network_predict(net, im.data);
**int i;
**float b;
**for (i = 0; i < 200; i++) {
**	b = feature2col_get_value(net, net_outputs, i, i, 0);
**}
*/
float feature2col_get_value(network *net, float *net_outputs, int h, int w,
		int c) {
	//TODO:assert the boundary of the h,w,c
	int out_h = net->layers[net->n - 1].out_h;
	int out_w = net->layers[net->n - 1].out_w;
	//	out_c = net->layers[net->n - 1].out_c;
	return net_outputs[w + out_w * h + out_w * out_h * c];
}

