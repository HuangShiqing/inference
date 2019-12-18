#ifndef IM2COL_H
#define IM2COL_H


void im2col_cpu(float* data_im, int channels, int height, int width, int ksize,
		int stride, int pad, float* data_col);
//added by myself
float im2col_get_pixel(float *im, int height, int width, int channels, int row,
		int col, int channel, int pad);

#ifdef GPU

void im2col_gpu(float *im, int channels, int height, int width, int ksize,
		int stride, int pad, float *data_col);

#endif
#endif
