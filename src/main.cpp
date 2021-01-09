extern "C"{
#include "gstreamer.h"
#include "gtk_show.h"
#include "convert.h"
#include "draw.h"
#include "model.h"
#include "parser.h"
#include "network.h"
#include "utils.h"
#include "im2col.h"

#include "my_common.h"
#include "my_layer.h"
// #include "event.h"
}

// #include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h> //sleep
#include <pthread.h>

#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#include "UltraFace.hpp"

unsigned char *buffer_original;	 //暂存从相机抓取的原始图像帧
unsigned char *buffer_processed; //从buffer_original copy过来的要转换后送入inference的图像帧
float *buffer_input;			 //从buffer_processed 转换过来的darkent支持的图像格式
unsigned char *buffer_show;		 //从buffer_processed cpoy过来再加上目标框的图像帧

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

char coco_names[1][20] ={"face"};

int main()
{
	gtk_show_init();
	if (gstreamer_init() != 0)
	{
		printf("gstreamer_init failed\r\n");
		return 0;
	}
	// image im = load_image_color("./resource/dog.jpg", 0, 0);
	// image sized = letterbox_image(im, 416, 416);
	buffer_input = (float*)malloc(WIDTH * HEIGHT * 3 * sizeof(float));
	image **alphabet = load_alphabet();	
	network *net = yolov3_tiny(1, 0);
	// network *net = test_model();
	// network *net = vgg16_adjusted(2);
	load_weights(net, "./resource/yolov3-tiny_120000.weights");
	sleep(1);
	printf("go\r\n");
	while (1)
	{
		// event_wait(&event1);//等待图像抓取线程抓到图像后发出的通知
		pthread_mutex_lock(&mutex1);								   // 上锁失败代表别的线程在使用，则当前线程阻塞
		memcpy(buffer_processed, buffer_original, WIDTH * HEIGHT * 3); // 给inference
		pthread_mutex_unlock(&mutex1);

		// rgb2bgr(buffer_processed, buffer_original, WIDTH, HEIGHT);
		nhwc2nchw_char2float(buffer_input, buffer_processed, WIDTH, HEIGHT); //gst图像转换到darknet图像

		double time_start = what_time_is_it_now();
		network_predict(net, buffer_input);
		int nboxes = 0;
		float thresh = .5;
		float nms = .45;
		layer l = net->layers[net->n - 1];
		detection *dets = get_network_boxes(net, WIDTH, HEIGHT, thresh, 0.5, 0, 1, &nboxes);
		do_nms_obj(dets, nboxes, l.classes, nms);

		double t = what_time_is_it_now() - time_start;
		printf("network_predict used: %f s\n", t);
		my_draw_dets(buffer_processed, WIDTH, HEIGHT, dets, nboxes, thresh, alphabet, coco_names);

		pthread_mutex_lock(&mutex2);							   // 上锁失败代表别的线程在使用，则当前线程阻塞
		memcpy(buffer_show, buffer_processed, WIDTH * HEIGHT * 3); // 准备给gtk
		pthread_mutex_unlock(&mutex2);
		// event_wake(&event2);
	}

	return 0;
}
// int main()
// {
//     image **alphabet = load_alphabet();	
//     gtk_show_init();
//     gstreamer_init();
//     UltraFace ultraface("./resource/RFB-320.mnn", WIDTH, HEIGHT, 4, 0.65); // config model input

//     sleep(1);
//     printf("go\r\n");
//     while(1)
//     {
//         pthread_mutex_lock(&mutex1);// 上锁失败代表别的线程在使用，则当前线程阻塞
// 		memcpy(buffer_processed, buffer_original, WIDTH*HEIGHT*3); // 给inference
// 		pthread_mutex_unlock(&mutex1);

//         // double time_start = what_time_is_it_now();
//         std::vector<FaceInfo> face_info;
//         ultraface.detect(buffer_processed,face_info);
//         // double t = what_time_is_it_now() - time_start;
// 		// printf("network_predict used: %f s\n", t);

//         int nboxes = face_info.size();
//         float thresh = .5;
//         detection *dets = new detection[nboxes];
//         ultraface.faceinfo2det(face_info, dets);
//         my_draw_dets(buffer_processed, WIDTH, HEIGHT, dets, nboxes, thresh, alphabet, coco_names);
//         free(dets);

//         pthread_mutex_lock(&mutex2);							   // 上锁失败代表别的线程在使用，则当前线程阻塞
// 		memcpy(buffer_show, buffer_processed, WIDTH * HEIGHT * 3); // 准备给gtk
// 		pthread_mutex_unlock(&mutex2);
//     }
//     return 0;
// }

