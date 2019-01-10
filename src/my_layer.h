/*
 * layer.h
 *
 *  Created on: Dec 29, 2018
 *      Author: hsq
 */

#ifndef LAYER_H_
#define LAYER_H_

#define BN 1
#define no_BN 0
//#define

network *init_net(int layer_num, int h, int w, int c);
int Conv2d(network *net, int filter, int size, int stride, int padding, int activation,
		int batch_normalize, int layer_index);
int DenseLayer(network *net, int output, ACTIVATION activation, int batch_normalize, int layer_index);
void finish_net(network *net);
float feature2col_get_value(network *net, float *net_outputs, int h, int w,
		int c);

#endif /* LAYER_H_ */
