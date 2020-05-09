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