#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include "my_macro.h"

//#include "parser.h"
#include "network.h"
//#include "connected_layer.h"
//#include "convolutional_layer.h"
//#include "maxpool_layer.h"
#include "utils.h"
#include "im2col.h"

#include "my_layer.h"
#include "model.h"

int main() {
	image im = load_image_color(
			"/home/hsq/DeepLearning/c++/inference/src/224-224.jpg", 224, 224);

	network *net = vgg16_adjusted(2);

//	save_weights(net, "/home/hsq/DeepLearning/c++/inference/src/test1.weights");
	load_weights(net, "/home/hsq/DeepLearning/code/branch/model_pruning/test.weights");

	int i = 0;
	float *net_outputs;
	double time_start=what_time_is_it_now();
	for (i = 0; i < 1000; i++) {
		net_outputs = network_predict(net, im.data);
	}
	double t = what_time_is_it_now()-time_start;
    printf("%f\n",t);

	return 0;
}
