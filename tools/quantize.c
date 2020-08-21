#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h> //sleep
#include <pthread.h>
#include <sys/io.h> //用于遍历
#include <dirent.h> //用于遍历

#include "parser.h"
#include "network.h"
#include "utils.h"
#include "im2col.h"
#include "darknet.h"

#include "my_common.h"
#include "my_layer.h"
#include "model.h"

void quantize_weights_symmetric(network *net)
{
    printf("\r\nstart quantize weight\r\n");
    //层循环
    for (int i = 0; i < net->n; i++)
    {
        if (net->layers[i].type != CONVOLUTIONAL)
            continue;
        float *weight = net->layers[i].weights;
        float absmax = 0;
        for (int j = 0; j < net->layers[i].nweights; j++)
        {
            if (fabsf(weight[j]) > absmax)
                absmax = fabsf(weight[j]);
        }
        // s = max/127
        net->layers[i].weights_quant_multipler = absmax / 127;
        printf("layer %i absmax=%f, s=%f\r\n", i, absmax,
               net->layers[i].weights_quant_multipler);
        for (int j = 0; j < net->layers[i].nweights; j++)
        {
            // int8 = round(float/s)
            net->layers[i].weights_int8[j] = (int8_t)roundf(
                net->layers[i].weights[j] / net->layers[i].weights_quant_multipler);
        }
    }
}

void quantize_featuremap_symmetric(network *net)
{
    //层循环
    for (int i = 0; i < net->n; i++)
    {
        layer l = net->layers[i];
        if (l.type != CONVOLUTIONAL)
            continue;
        // printf("layer %i\r\n",i);
        // channel循环
        for (int j = 0; j < l.c; j++)
        {
            float *out = net->layers[i - 1].output + j * l.h * l.w;
            float *input_quant_multipler = l.input_quant_multipler + j;
            float absmax = *input_quant_multipler * 127; //恢复来自上一张图片的absmax
            for (int k = 0; k < l.h * l.w; k++)
            {
                if (fabsf(out[k]) > absmax)
                    absmax = fabsf(out[k]);
            }
            // s = max/127
            *input_quant_multipler = absmax / 127;
        }
    }
    fprintf(stderr, "*");
    fflush(stderr);
}

int list_dir(char *dir, char **name)
{
    // printf("路径为[%s]\r\n", dir);
    struct dirent *ent = NULL;
    DIR *pDir;
    pDir = opendir(dir);
    // d_reclen：16表示子目录或以.开头的隐藏文件，24表示普通文本文件,28为二进制文件，还有其他……
    int img_num = 0;
    while (NULL != (ent = readdir(pDir)))
    {
        // printf("reclen=%d    type=%d\r\n", ent->d_reclen, ent->d_type);
        if (ent->d_reclen == 24)
        {
            // d_type：4表示为目录，8表示为文件
            if (ent->d_type == 8)
            {
                // printf("普通文件[%s]\r\n", ent->d_name);
            }
        }
        else if (ent->d_reclen == 16)
        {
            // printf("[.]开头的子目录或隐藏文件[%s]\r\n",ent->d_name);
        }
        else
        {
            // printf("其他文件[%s]\n", ent->d_name);
            char *a = (char *)malloc(strlen(ent->d_name) * sizeof(char));
            memcpy(a, ent->d_name, strlen(ent->d_name) * sizeof(char));
            name[img_num++] = a;
        }
    }
    return img_num;
}

int main()
{
    char *name[1000]; //最多可以用1000张图像进行量化
    char *dir = "../resource/calibration/";
    int img_num = list_dir(dir, name);
    char path[100]; //路径名最长为100个字符

    network *net = yolov3_tiny(1);
    load_weights(net, "../resource/yolov3-tiny_120000.weights");
    quantize_weights_symmetric(net);
    printf("\r\nstart quan featuremap\r\n");
    for (int i = 0; i < img_num; i++)
    {
        sprintf(path, "%s%s", dir, name[i]);// 合并字符串得到完整路径
        image im = load_image_color(path, 0, 0);
        image sized = letterbox_image(im, 416, 416);
        network_predict(net, sized.data);
        quantize_featuremap_symmetric(net);        
    }
    printf("\r\n");
    return 0;
}