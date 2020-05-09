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
#include "gstreamer.h"
#include "gtk_show.h"
#include "event.h"
#include "convert.h"

unsigned char **ringbuffer;//暂存从相机抓取的原始图像
unsigned char **ringbuffer_processed;//暂存处理后的图像即添加了目标框的图像
unsigned int latest_index = 0;//需要处理的原始图像的index
unsigned int latest_index_processed = 0;//添加目标框的图像的index
int ringbuffer_length=2;//ringbuffer的长度
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
event_parameter event1;

int main()
{
	int width = 320;
	int height = 240;
	// int ringbuffer_length = 4;
	gtk_show_init();
	if (gstreamer_init(width, height) != 0)
	{
		printf("gstreamer_init failed\r\n");
		return 0;
	}	
	// image im = load_image_color("./resource/dog.jpg", 0, 0);
	// image sized = letterbox_image(im, 416, 416);

	network *net = yolov3_tiny(80);
	// network *net = test_model();
	// network *net = vgg16_adjusted(2);
	load_weights(net, "./resource/yolov3-tiny.weights");

    while (1)
    {
        event_wait(&event1);//等待图像抓取线程抓到图像后发出的通知
        pthread_mutex_lock(&mutex1); // 上锁失败代表别的线程在使用，则当前线程阻塞
        const unsigned int latest_original = latest_index;
        pthread_mutex_unlock(&mutex1);

        const unsigned int next_index = latest_original;
        // memcpy(ringbuffer_processed[next_index], ringbuffer[latest_original], 320 * 240 * 3);
        rgb2bgr(ringbuffer_processed[next_index], ringbuffer[latest_original], width, height);

        pthread_mutex_lock(&mutex2); // 上锁失败代表别的线程在使用，则当前线程阻塞
        latest_index_processed = next_index;
        pthread_mutex_unlock(&mutex2);
        // event_wake(&event2);
    }

	// int i = 0;
	// double time_start = what_time_is_it_now();
	// printf("start inference\n");
	// // for (i = 0; i < 1; i++) {
	// float *net_outputs = network_predict(net, sized.data);

	// int nboxes = 0;
	// float thresh = .5;
	// float nms = .45;
	// layer l = net->layers[net->n - 1];
	// detection *dets = get_network_boxes(net, im.w, im.h, thresh, 0.5, 0, 1, &nboxes);
	// do_nms_sort(dets, nboxes, l.classes, nms);
	// // }
	// double t = what_time_is_it_now() - time_start;
	// printf("time used: %f s\n", t);

	return 0;
}
