#include "im2col.h"
#include <stdio.h>
/*
**  从输入的多通道数组im（存储图像数据）中获取指定行、列、、通道数处的元素值
**  输入： im      输入，所有数据存成一个一维数组，例如对于3通道的二维图像而言，
**                每一通道按行存储（每一通道所有行并成一行），三通道依次再并成一行
**        height  每一通道的高度（即输入图像的真正的高度，补0之前）
**        width   每一通道的宽度（即输入图像的宽度，补0之前）
**        channels 输入im的通道数，比如彩色图为3通道，之后每一卷积层的输入的通道数等于上一卷积层卷积核的个数
**        row     要提取的元素所在的行（二维图像补0之后的行数）
**        col     要提取的元素所在的列（二维图像补0之后的列数）
**        channel 要提取的元素所在的通道
**        pad     图像左右上下各补0的长度（四边补0的长度一样）
**  返回： float类型数据，为im中channel通道，row-pad行，col-pad列处的元素值
**  注意：在im中并没有存储补0的元素值，因此height，width都是没有补0时输入图像真正的
**       高、宽；而row与col则是补0之后，元素所在的行列，因此，要准确获取在im中的元素值，
**       首先需要减去pad以获取在im中真实的行列数
*/
float im2col_get_pixel(float *im, int height, int width, int channels, int row,
		int col, int channel, int pad)
{
	row -= pad;
	col -= pad;

	if (row < 0 || col < 0 || row >= height || col >= width)
		return 0;
	return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im, int channels, int height, int width, int ksize,
		int stride, int pad, float* data_col)
{
	int c, h, w;
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;

	int channels_col = channels * ksize * ksize;
	for (c = 0; c < channels_col; ++c)
	{
		int w_offset = c % ksize;
		int h_offset = (c / ksize) % ksize;
		int c_im = c / ksize / ksize;
		for (h = 0; h < height_col; ++h)
		{
			for (w = 0; w < width_col; ++w)
			{
				int im_row = h_offset + h * stride;
				int im_col = w_offset + w * stride;
				int col_index = (c * height_col + h) * width_col + w;
				data_col[col_index] = im2col_get_pixel(data_im, height, width,
						channels, im_row, im_col, c_im, pad);
			}
		}
	}
}

