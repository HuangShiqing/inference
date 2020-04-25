#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

#include "parser.h"
#include "network.h"
#include "utils.h"
#include "im2col.h"

#include "my_layer.h"
#include "model.h"

int main() {
	image im = load_image_color("./resource/224-224.jpg", 224, 224);
	
	network *net = test_model();
	// network *net = vgg16_adjusted(2);
	// load_weights(net, "./resource/test.weights");
	// save_weights(net, "./resource/test1.weights");	

	int i = 0;
	float *net_outputs;
	double time_start=what_time_is_it_now();
	printf("start inference\n");
	// for (i = 0; i < 1; i++) {
	net_outputs = network_predict(net, im.data);
	// }
	double t = what_time_is_it_now()-time_start;
    printf("time used: %f s\n",t);

	return 0;
}
