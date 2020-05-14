#ifndef DRWA_H
#define DRAW_H
#include "darknet.h"
void my_draw_box(unsigned char *src, int width, int height, int left, int right, int top, int bot, unsigned char r, unsigned char g, unsigned char b);
void my_draw_dets(unsigned char *src, int width, int height, detection *dets, int nboxes, float thresh, image **alphabet, char names[][20]);
void my_draw_prob(unsigned char *src, int width, int height, int left, int right, int top, int bot, float prob, unsigned char r, unsigned char g, unsigned char b);
void my_draw_label(unsigned char *dst, unsigned char *src, int top, int left, int width, int height, int w, int h, unsigned char r, unsigned char g, unsigned char b);
#endif