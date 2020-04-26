#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
//void save_weights_double(network net, char *filename);
int *parse_yolo_mask(char *a, int *num);
// list *read_cfg(char *filename);

#endif
