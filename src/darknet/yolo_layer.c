#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    // 最后一个yolo层l.n=3,l.mask[0]=0,l.mask[1]=1,l.mask[2]=2
    // 第一个yolo层l.n=3,l.mask[0]=6,l.mask[1]=7,l.mask[2]=8
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    // l.cost = calloc(1, sizeof(float));// 用于最后一层计算损失值
    // yolo层的biases用来存储anchor尺度，这里的total=9
    l.biases = calloc(total*2, sizeof(float));
    // 最后一个yolo层的l.mask=[0，1，2],第一个yolo层l.mask=[6,7,8],用来指定9个anchor分配到cell的3个位置上
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    // l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    // l.truths = 90*(4 + 1);// 训练用的参数
    // l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    // l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    // l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    // l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

// void resize_yolo_layer(layer *l, int w, int h)
// {
//     l->w = w;
//     l->h = h;

//     l->outputs = h*w*l->n*(l->classes + 4 + 1);
//     l->inputs = l->outputs;

//     l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
//     l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

// #ifdef GPU
//     cuda_free(l->delta_gpu);
//     cuda_free(l->output_gpu);

//     l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
//     l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
// #endif
// }

box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    // 这里的yolo层的biases是用来存储anchor的大小的，mask就是biases即anchor的索引
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
// {
//     box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
//     float iou = box_iou(pred, truth);

//     float tx = (truth.x*lw - i);
//     float ty = (truth.y*lh - j);
//     float tw = log(truth.w*w / biases[2*n]);
//     float th = log(truth.h*h / biases[2*n + 1]);

//     delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
//     delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
//     delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
//     delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
//     return iou;
// }


// void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
// {
//     int n;
//     if (delta[index]){
//         delta[index + stride*class] = 1 - output[index + stride*class];
//         if(avg_cat) *avg_cat += output[index + stride*class];
//         return;
//     }
//     for(n = 0; n < classes; ++n){
//         delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
//         if(n == class && avg_cat) *avg_cat += output[index + stride*n];
//     }
// }

// feature map = [w1,w2]
// 主要实现对yolo层线性存储的feature map的一个指定w、h、n到线性存储里对应的索引映射
// batch这里无效，为0，location是尺度和w、h位置的结合体，entry是这个(4+l.classes+1)维度的索引
static int entry_index(layer l, int batch, int location, int entry)
{
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    // 从这里可以看出feature map的内存排布，先放w再放下一行的w，当前通道放完后然后放下一个通道的w、h
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    // if(!net.train) return;
    // float avg_iou = 0;
    // float recall = 0;
    // float recall75 = 0;
    // float avg_cat = 0;
    // float avg_obj = 0;
    // float avg_anyobj = 0;
    // int count = 0;
    // int class_count = 0;
    // *(l.cost) = 0;
    // for (b = 0; b < l.batch; ++b) {
    //     for (j = 0; j < l.h; ++j) {
    //         for (i = 0; i < l.w; ++i) {
    //             for (n = 0; n < l.n; ++n) {
    //                 int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
    //                 box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
    //                 float best_iou = 0;
    //                 int best_t = 0;
    //                 for(t = 0; t < l.max_boxes; ++t){
    //                     box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
    //                     if(!truth.x) break;
    //                     float iou = box_iou(pred, truth);
    //                     if (iou > best_iou) {
    //                         best_iou = iou;
    //                         best_t = t;
    //                     }
    //                 }
    //                 int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
    //                 avg_anyobj += l.output[obj_index];
    //                 l.delta[obj_index] = 0 - l.output[obj_index];
    //                 if (best_iou > l.ignore_thresh) {
    //                     l.delta[obj_index] = 0;
    //                 }
    //                 if (best_iou > l.truth_thresh) {
    //                     l.delta[obj_index] = 1 - l.output[obj_index];

    //                     int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
    //                     if (l.map) class = l.map[class];
    //                     int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
    //                     delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
    //                     box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
    //                     delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
    //                 }
    //             }
    //         }
    //     }
    //     for(t = 0; t < l.max_boxes; ++t){
    //         box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

    //         if(!truth.x) break;
    //         float best_iou = 0;
    //         int best_n = 0;
    //         i = (truth.x * l.w);
    //         j = (truth.y * l.h);
    //         box truth_shift = truth;
    //         truth_shift.x = truth_shift.y = 0;
    //         for(n = 0; n < l.total; ++n){
    //             box pred = {0};
    //             pred.w = l.biases[2*n]/net.w;
    //             pred.h = l.biases[2*n+1]/net.h;
    //             float iou = box_iou(pred, truth_shift);
    //             if (iou > best_iou){
    //                 best_iou = iou;
    //                 best_n = n;
    //             }
    //         }

    //         int mask_n = int_index(l.mask, best_n, l.n);
    //         if(mask_n >= 0){
    //             int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
    //             float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

    //             int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
    //             avg_obj += l.output[obj_index];
    //             l.delta[obj_index] = 1 - l.output[obj_index];

    //             int class = net.truth[t*(4 + 1) + b*l.truths + 4];
    //             if (l.map) class = l.map[class];
    //             int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
    //             delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

    //             ++count;
    //             ++class_count;
    //             if(iou > .5) recall += 1;
    //             if(iou > .75) recall75 += 1;
    //             avg_iou += iou;
    //         }
    //     }
    // }
    // *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    // printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

// void backward_yolo_layer(const layer l, network net)
// {
//    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
// }

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            // 获得置信度的索引
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            // 置信度大于阈值则有效框计数加一
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

// void avg_flipped_yolo(layer l)
// {
//     int i,j,n,z;
//     float *flip = l.output + l.outputs;
//     for (j = 0; j < l.h; ++j) {
//         for (i = 0; i < l.w/2; ++i) {
//             for (n = 0; n < l.n; ++n) {
//                 for(z = 0; z < l.classes + 4 + 1; ++z){
//                     int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
//                     int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
//                     float swap = flip[i1];
//                     flip[i1] = flip[i2];
//                     flip[i2] = swap;
//                     if(z == 0){
//                         flip[i1] = -flip[i1];
//                         flip[i2] = -flip[i2];
//                     }
//                 }
//             }
//         }
//     }
//     for(i = 0; i < l.outputs; ++i){
//         l.output[i] = (l.output[i] + flip[i])/2.;
//     }
// }

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    // 把feature map搞了个马甲
    float *predictions = l.output;
    // 应该用不着这句
    // if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        // 证明feature map是按行存储的
        int row = i / l.w;
        int col = i % l.w;
        // yolo层的l.n是等于3，代表3个不同尺度
        for(n = 0; n < l.n; ++n){
            // 指定n为尺度，i指定w、h上的位置，4为指定每个尺度上的第几个数，
            // 按照(x,y,w,h,s,class)排布，第四个数是bbx的置信度
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            // 置信度s的马甲
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            // 获得每个cell最前面那个索引位置
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            // l.w是52那个，netw是416那个，这里将bbox转成在416图像里的位置，
            // 下面那个correct_yolo_boxes才是再转成在输入图像里的位置
            // 和tf版的一个意思，不对，应该说是tf版就是复刻这个的
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            // 这里下面就是赋值，将feature map写到dets里
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                // dest.prob = s*class
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
                if(dets[count].prob[j] > dets[count].max_prob)
                {
                    dets[count].max_prob = dets[count].prob[j];
                    dets[count].max_class = j;
                }
            }
            ++count;
        }
    }
    // 将dets的位置转成在输入图像里的位置，过程大致和tf一样，细节没仔细对比
    // w、h是图像大小，netw、neth是416那个，relative为0，这里无效
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    // 返回处理的框数
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    // if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    // }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

// void backward_yolo_layer_gpu(const layer l, network net)
// {
//     axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
// }
#endif

