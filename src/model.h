/*
 * model.h
 *
 *  Created on: Apr 11, 2019
 *      Author: hsq
 */

#ifndef MODEL_H_
#define MODEL_H_

network *vgg16_adjusted(int out_units, int quantize);
network *test_model();
network *yolov3_tiny(int classes, int quantize);


#endif /* MODEL_H_ */
