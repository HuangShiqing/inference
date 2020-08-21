#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h> //sleep
#include <pthread.h>
#include <sys/io.h>//用于遍历
#include <dirent.h>//用于遍历

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
    printf("start quantize weight\r\n");
    for(int i=0;i<net->n;i++)//层循环
    {
        if(net->layers[i].type!=CONVOLUTIONAL)
            continue;
        float* weight = net->layers[i].weights;
        float absmax = 0;        
        for(int j=0;j<net->layers[i].nweights;j++)
        {
            if(fabsf(weight[j])>absmax)
                absmax = fabsf(weight[j]);
        }
        //s = max/127
        net->layers[i].weights_quant_multipler = absmax/127;
        printf("layer %i absmax=%f, s=%f\r\n",i,absmax,net->layers[i].weights_quant_multipler);
        for(int j=0;j<net->layers[i].nweights;j++)
        {
            // int8 = round(float/s)
            net->layers[i].weights_int8[j] = (int8_t)roundf(net->layers[i].weights[j]/net->layers[i].weights_quant_multipler);
        }
    }
}


int main()
{
    network *net = yolov3_tiny(1);
    load_weights(net, "../resource/yolov3-tiny_120000.weights");
    quantize_weights_symmetric(net);

    return 0;
}