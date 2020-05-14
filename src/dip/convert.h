#ifndef CONVERT_H
#define CONVERT_H

void rgb2bgr(unsigned char *dst, unsigned char *src, int width, int height);
void nhwc2nchw(unsigned char *dst, unsigned char *src, int width, int height);
void nchw2nhwc(unsigned char *dst, unsigned char *src, int width, int height);
void char2float(float *dst, unsigned char *src, int width, int height);
void float2char(unsigned char *dst, float *src, int width, int height);
void nhwc2nchw_char2float(float *dst, unsigned char *src, int width, int height);
void nchw2nhwc_float2char(unsigned char *dst, float *src, int width, int height);
#endif
