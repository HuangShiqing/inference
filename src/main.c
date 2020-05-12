#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h> //sleep
#include <pthread.h>

#include "parser.h"
#include "network.h"
#include "utils.h"
#include "im2col.h"

#include "my_common.h"
#include "my_layer.h"
#include "model.h"
#include "gstreamer.h"
#include "gtk_show.h"
// #include "event.h"
#include "convert.h"

unsigned char *buffer_original;	 //暂存从相机抓取的原始图像帧
unsigned char *buffer_processed; //从buffer_original copy过来的要转换后送入inference的图像帧
float *buffer_input;			 //从buffer_processed 转换过来的darkent支持的图像格式
unsigned char *buffer_show;		 //从buffer_processed cpoy过来再加上目标框的图像帧

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;
// event_parameter event1;

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
	buffer_input = malloc(WIDTH * HEIGHT * 3 * sizeof(float));
	
	network *net = yolov3_tiny(80);
	// network *net = test_model();
	// network *net = vgg16_adjusted(2);
	load_weights(net, "./resource/yolov3-tiny.weights");
	sleep(1);
	while (1)
	{
		// event_wait(&event1);//等待图像抓取线程抓到图像后发出的通知
		pthread_mutex_lock(&mutex1);								   // 上锁失败代表别的线程在使用，则当前线程阻塞
		memcpy(buffer_processed, buffer_original, WIDTH * HEIGHT * 3); // 给inference
		pthread_mutex_unlock(&mutex1);

		rgb2bgr(buffer_processed, buffer_original, WIDTH, HEIGHT);		

		pthread_mutex_lock(&mutex2);							   // 上锁失败代表别的线程在使用，则当前线程阻塞
		memcpy(buffer_show, buffer_processed, WIDTH * HEIGHT * 3); // 准备给gtk
		pthread_mutex_unlock(&mutex2);
		// event_wake(&event2);
	}

	return 0;
}
