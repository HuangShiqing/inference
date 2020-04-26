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

network *test_model()
{
	int layer_num = 1;
	int layer_index = 0;
	int input_h = 224, input_w = 224, c = 3;

	network *net = init_net(layer_num, input_h, input_w, c);
	layer_index = Conv2d(net, 64, 3, 1, 1, RELU, no_BN, layer_index);
	finish_net(net);
	return net;
}

network *yolov3_tiny(int classes)
{
	int layer_num = 24;
	int layer_index = 0;
	int input_h = 416, input_w = 416, c = 3;

	network *net = init_net(layer_num, input_h, input_w, c);
	layer_index = Conv2d(net, 16, 3, 1, 1, LEAKY, BN, layer_index);//0
	layer_index = MaxPool(net, 2, 2, 0, layer_index);
	layer_index = Conv2d(net, 32, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);
	layer_index = Conv2d(net, 64, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);//5
	layer_index = Conv2d(net, 128, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);
	layer_index = Conv2d(net, 256, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = MaxPool(net, 2, 2, 0, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, LEAKY, BN, layer_index);//10
	layer_index = MaxPool(net, 2, 1, 0, layer_index);
	layer_index = Conv2d(net, 1024, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = Conv2d(net, 256, 1, 1, 0, LEAKY, BN, layer_index);
	layer_index = Conv2d(net, 512, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = Conv2d(net, 3*(classes+5), 1, 1, 0, LINEAR, no_BN, layer_index);//15
	layer_index = YoloLayer(net, 6, "10,14,23,27,37,58,81,82,135,169,344,319", "3,4,5", classes, layer_index);
	layer_index = RouteLayer(net, "-4", layer_index);
	layer_index = Conv2d(net, 128, 1, 1, 0, LEAKY, BN, layer_index);
	layer_index = UpsampleLayer(net, 2, 1, layer_index);
	layer_index = RouteLayer(net, "-1,8", layer_index);//20
	layer_index = Conv2d(net, 256, 3, 1, 1, LEAKY, BN, layer_index);
	layer_index = Conv2d(net, 3*(classes+5), 1, 1, 0, LINEAR, no_BN, layer_index);
	layer_index = YoloLayer(net, 6, "10,14,23,27,37,58,81,82,135,169,344,319", "0,1,2", classes, layer_index);
	finish_net(net);
	return net;
}
