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
	image im = load_image_color("./resource/dog.jpg", 0, 0);
	image sized = letterbox_image(im, 416, 416);

	network *net = yolov3_tiny(80);
	// network *net = test_model();
	// network *net = vgg16_adjusted(2);
	load_weights(net, "./resource/yolov3-tiny.weights");
	// save_weights(net, "./resource/test1.weights");	

	int i = 0;
	double time_start=what_time_is_it_now();
	printf("start inference\n");
	// for (i = 0; i < 1; i++) {
	float *net_outputs = network_predict(net, sized.data);

	int nboxes = 0;
	float thresh = .5;
	float nms = .45;
	layer l = net->layers[net->n-1];
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, 0.5, 0, 1, &nboxes);
	do_nms_sort(dets, nboxes, l.classes, nms);
	// }
	double t = what_time_is_it_now()-time_start;
    printf("time used: %f s\n",t);

	return 0;
}
