#include "convert.h"

//适用于内存排布是image[h,w,c]
void rgb2bgr(unsigned char *dst, unsigned char *src, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        int widthx3xi = width * 3 * i;
        for (int j = 0; j < width; j++)
        {
            int jx3 = 3 * j;
            for (int k = 0; k < 3; k++)
            {
                dst[widthx3xi + jx3 + k] = src[widthx3xi + jx3 + (2 - k)];
            }
        }
    }
}

void nhwc2nchw(unsigned char *dst, unsigned char *src, int width, int height)
{
    unsigned char *dst_r = dst;
    unsigned char *dst_g = dst + width * height;
    unsigned char *dst_b = dst + 2 * width * height;
    for (int i = 0; i < width * height; i++)
    {
        dst_r[i] = src[3 * i];
        dst_g[i] = src[3 * i + 1];
        dst_b[i] = src[3 * i + 2];
    }
}

void nchw2nhwc(unsigned char *dst, unsigned char *src, int width, int height)
{
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
        {
            dst[width * 3 * j + 3 * i] = src[width * j + i];
            dst[width * 3 * j + 3 * i + 1] = src[width * j + i + width * height];
            dst[width * 3 * j + 3 * i + 2] = src[width * j + i + 2 * width * height];
        }
}

void char2float(float *dst, unsigned char *src, int width, int height)
{
    for (int i = 0; i < width * height * 3; i++)
        dst[i] = (float)(src[i] / 255.);
}

void float2char(unsigned char *dst, float *src, int width, int height)
{
    for (int i = 0; i < width * height * 3; i++)
        dst[i] = (unsigned char)(src[i] * 255);
}

void nhwc2nchw_char2float(float *dst, unsigned char *src, int width, int height)
{
    float *dst_r = dst;
    float *dst_g = dst + width * height;
    float *dst_b = dst + 2 * width * height;
    for (int i = 0; i < width * height; i++)
    {
        dst_r[i] = (float)(src[3 * i] / 255.);
        dst_g[i] = (float)(src[3 * i + 1] / 255.);
        dst_b[i] = (float)(src[3 * i + 2] / 255.);
    }
}

void nchw2nhwc_float2char(unsigned char *dst, float *src, int width, int height)
{
    for (int j = 0; j < height; j++)
        for (int i = 0; i < width; i++)
        {
            dst[width * 3 * j + 3 * i] = (unsigned char)((src[width * j + i]) * 255);
            dst[width * 3 * j + 3 * i + 1] = (unsigned char)((src[width * j + i + width * height]) * 255);
            dst[width * 3 * j + 3 * i + 2] = (unsigned char)((src[width * j + i + 2 * width * height]) * 255);
        }
}