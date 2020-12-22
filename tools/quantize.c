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
        net->layers[i].weights_quant_multipler[0] = absmax / 127;
        printf("layer %i absmax=%f, s=%f\r\n", i, absmax,
               net->layers[i].weights_quant_multipler[0]);
        for (int j = 0; j < net->layers[i].nweights; j++)
        {
            // int8 = round(float/s)
            net->layers[i].weights_int8[j] = (int8_t)roundf(
                net->layers[i].weights[j] / net->layers[i].weights_quant_multipler[0]);
        }
    }
}

void quantize_featuremap_symmetric(network *net, float *input)
{
    float *input_feature;
    float *input_quant_multipler;
    float absmax;
    //层循环
    for (int i = 0; i < net->n; i++)
    {
        layer l = net->layers[i];
        if (l.type != CONVOLUTIONAL)
            continue;
        // printf("layer %i\r\n",i);
        // channel循环
        input_quant_multipler = l.input_quant_multipler;// + j;
        absmax = *input_quant_multipler * 127; //恢复来自上一张图片的absmax
        for (int j = 0; j < l.c * l.h * l.w; j++)
        {
            if(i==0)
                input_feature = input + j;// * l.h * l.w;
            else                
                input_feature = net->layers[i - 1].output + j; //* l.h * l.w;
            //input_quant_multipler = l.input_quant_multipler;// + j;
            //absmax = *input_quant_multipler * 127; //恢复来自上一张图片的absmax
            //for (int k = 0; k < l.h * l.w; k++)
            {
                if (fabsf(input_feature[0]) > absmax)
                    absmax = fabsf(input_feature[0]);
            }
            // s = max/127
            //*input_quant_multipler = absmax / 127;
        }
        *input_quant_multipler = absmax / 127;
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
            //strlen计算字符串长度，不会包括\0，因此malloc申请 用于拷贝该字符串 来存储的空间时size需要+1
            char *a = (char *)calloc(strlen(ent->d_name)+1, sizeof(char));
            strcpy(a, ent->d_name);
            name[img_num++] = a;
        }
    }
    return img_num;
}

void write_feature(char* str, layer layer)
{
    FILE *fpWrite=fopen(str,"w");
    fprintf(fpWrite,"%d,%d,%d\r\n",layer.out_c,layer.out_h,layer.out_w);
    for(int j=0; j<layer.out_c*layer.out_h*layer.out_w; j++)
    {        
        fprintf(fpWrite,"%f\r\n",layer.output[j]);
    }        
    fclose(fpWrite);
}

int main()
{
    char *name[1000]; //最多可以用1000张图像进行量化
    char *dir = "../resource/calibration/";
    int img_num = list_dir(dir, name);
    char path[300]; //路径名最长为100个字符

    network *net = yolov3_tiny(1, 0);
    load_weights(net, "../resource/yolov3-tiny_120000.weights");
    quantize_weights_symmetric(net);
    printf("\r\nstart quan featuremap\r\n");
    for (int i = 0; i < img_num; i++)
    {
        sprintf(path, "%s%s", dir, name[i]); // 合并字符串得到完整路径
        image im = load_image_color(path, 0, 0);
        image sized = letterbox_image(im, 416, 416);
        network_predict(net, sized.data);
        quantize_featuremap_symmetric(net, sized.data);
    }
    printf("\r\n");
    for(int i=0;i<net->n;i++)
    {
        layer l = net->layers[i];
        if (l.type != CONVOLUTIONAL)
            continue;
        printf("layer %i, s=%f\r\n", i, net->layers[i].input_quant_multipler[0]);
    
    }
    save_weights(net, "../resource/yolov3-tiny_120000_q.weights", 1);
    //FILE *fpWrite=fopen("data.txt","w");
    //for(int j=0;j<net->n;j++)
    //{
    //    if(net->layers[j].type != CONVOLUTIONAL)
    //        continue;
    //    fprintf(fpWrite,"layer%d:\r\n",j);
    //    for(int k=0;k<net->layers[j].c;k++)
    //        fprintf(fpWrite,"%f ",net->layers[j].input_quant_multipler[k]);
    //    fprintf(fpWrite,"\r\n");
    //}        
    //fclose(fpWrite);
    return 0;
}


    
// int main()
// {
//     network *net = yolov3_tiny(1, 1);
//     load_weights(net, "../resource/yolov3-tiny_120000_q.weights");
//     image im = load_image_color("../resource/face.jpg", 0, 0);
//     image sized = letterbox_image(im, 416, 416);
//     printf("start inference\r\n");
//     network_predict(net, sized.data);
//     printf("done\r\n");

//     for(int i=0;i<net->n;i++)
//     {
//         layer l = net->layers[i];
//         if (l.type != CONVOLUTIONAL)
//             continue;
//         printf("layer %i, s=%f\r\n", i, net->layers[i].input_quant_multipler[0]);
    
//     }
//     // char *name = "1.txt";
//     // write_feature(name, net->layers[0]);

//     int nboxes = 0;
//     float thresh = .5;
//     float nms = .45;
//     layer l = net->layers[net->n - 1];
//     detection *dets = get_network_boxes(net, WIDTH, HEIGHT, thresh, 0.5, 0, 1, &nboxes);
//     do_nms_obj(dets, nboxes, l.classes, nms);
//     return 0;
// }