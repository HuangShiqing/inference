#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include "parser.h"
#include "network.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "utils.h"
#include "im2col.h"

#include "my_layer.h"

int main() {
//	char *filename = "/home/hsq/DeepLearning/c++/inference/src/test.cfg";
//	network *net = parse_network_cfg(filename);

//net params
	int layer_num = 1;
	int layer_index = 0;
	int input_h = 416, input_w = 416, c = 3;

	image im = load_image_color(
			"/home/hsq/DeepLearning/c++/inference/src/1.jpg", input_h, input_w);

//	float b;
//	int row = 0, col = 0, channel = 0;
//	b = im2col_get_pixel(im.data, input_h, input_w, c, row, col, channel, 0);

	network *net = init_net(layer_num, input_h, input_w, c);

	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, BN, layer_index);
//	layer_index = DenseLayer(net, 2, LINEAR, no_BN, layer_index);

	finish_net(net);

//	save_weights(net, "/home/hsq/DeepLearning/c++/inference/src/test1.weights");
//	load_weights(net, "/home/hsq/DeepLearning/code/model_pruning/test1.weights");

	float *net_outputs = network_predict(net, im.data);

	int i;
	float b;
	for (i = 0; i < 200; i++) {
//		b = net->layers->weights[i];
		b = feature2col_get_value(net, net_outputs, i, i, 0);
//		b = net_outputs[i];
	}

	return 0;
}
