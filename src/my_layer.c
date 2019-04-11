#include "network.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"

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
		int activation, int batch_normalize, int layer_index) {
	fprintf(stderr, "%5d ", layer_index);
	int binary = 0, xnor = 0, adam = 0;
	int h, w, c, n;
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
			activation, batch_normalize, binary, xnor, adam);
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

