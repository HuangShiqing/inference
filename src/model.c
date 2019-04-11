#include "network.h"
#include "my_layer.h"
#include "model.h"

network *vgg16_adjusted(int out_units)
{
	int layer_num = 19;
	int layer_index = 0;
	int input_h = 224, input_w = 224, c = 3;

	network *net = init_net(layer_num, input_h, input_w, c);

	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 128, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 128, 3, 1, 1, RELU, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 256, 3, 1, 1, RELU, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);

	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, RELU, BN, layer_index);

	layer_index = AvgPool(net, layer_index);
	layer_index = DenseLayer(net, out_units, LINEAR, no_BN, layer_index);

	finish_net(net);

	return net;
}
