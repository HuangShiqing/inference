#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include "my_macro.h"

#include "parser.h"
#include "network.h"
#include "connected_layer.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "utils.h"
#include "im2col.h"

#include "my_layer.h"


int main() {
//	char *filename = "/home/hsq/DeepLearning/c++/inference/src/test.cfg";
//	network *net = parse_network_cfg(filename);

//net params
	int layer_num = 19;
	int layer_index = 0;
	int input_h = 224, input_w = 224, c = 3;
	struct timeval t1, t2;

	image im = load_image_color(
			"/home/hsq/DeepLearning/c++/inference/src/224-224.jpg", input_h, input_w);

//	float b;
//	int row = 0, col = 0, channel = 0;
//	b = im2col_get_pixel(im.data, input_h, input_w, c, row, col, channel, 0);

	network *net = init_net(layer_num, input_h, input_w, c);

	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 128, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 128, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, no_BN, layer_index);

	layer_index = AvgPool(net, layer_index);
	layer_index = DenseLayer(net, 2, LINEAR, no_BN, layer_index);

	finish_net(net);

//	save_weights(net, "/home/hsq/DeepLearning/c++/inference/src/test1.weights");
//	load_weights(net, "/home/hsq/DeepLearning/code/model_pruning/test1.weights");
	int i = 0, deltaT = 0;
	int start=clock();
	for (i = 0; i < 1000; i++) {
//		gettimeofday(&t1, NULL);
		float *net_outputs = network_predict(net, im.data);
//		gettimeofday(&t2, NULL);
//		deltaT += (t2.tv_sec - t1.tv_sec) * 1000000 + t2.tv_usec - t1.tv_usec;
	}
	int finish=clock();
    int totaltime=(double)(finish-start)/CLOCKS_PER_SEC;
    printf("%d",totaltime);
//	printf("%d",deltaT);

//	int i;
//	float b;
//	for (i = 0; i < 200; i++) {
////		b = net->layers->weights[i];
//		b = feature2col_get_value(net, net_outputs, i, i, 0);
////		b = net_outputs[i];
//	}

	return 0;
}
