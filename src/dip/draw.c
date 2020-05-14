#include "draw.h"
#include "convert.h"
#include "my_common.h"

// 越界检查需要在输入前做好
// src:nhwc
void my_draw_box(unsigned char *src, int width, int height, int left, int right, int top, int bot, unsigned char r, unsigned char g, unsigned char b)
{
    int left_top_index = 3 * width * top + 3 * left;
    int right_top_index = 3 * width * top + 3 * right;
    int left_bot_index = 3 * width * bot + 3 * left;
    int right_bot_index = 3 * width * bot + 3 * right;
    for (int i = left_top_index; i < right_top_index; i += 3)
    {
        src[i] = r;
        src[i + 1] = g;
        src[i + 2] = b;
    }
    for (int i = left_bot_index; i < right_bot_index; i += 3)
    {
        src[i] = r;
        src[i + 1] = g;
        src[i + 2] = b;
    }
    for (int i = left_top_index; i < left_bot_index; i += (3 * width))
    {
        src[i] = r;
        src[i + 1] = g;
        src[i + 2] = b;
    }
    for (int i = right_top_index; i < right_bot_index; i += (3 * width))
    {
        src[i] = r;
        src[i + 1] = g;
        src[i + 2] = b;
    }
}

// 越界检查需要在输入前做好
// src:nhwc
void my_draw_prob(unsigned char *src, int width, int height, int left, int right, int top, int bot, float prob, unsigned char r, unsigned char g, unsigned char b)
{
    int h = (bot - top) * prob;
    int right_bot_index = 3 * width * bot + 3 * right;

    int prob_bot_index = right_bot_index;
    int prob_top_index = prob_bot_index - h * 3 * width;
    for (int i = prob_top_index; i < prob_bot_index; i += (3 * width))
    {
        src[i] = r;
        src[i + 1] = g;
        src[i + 2] = b;
    }
}

// 越界检查需要在输入前做好
// src:nhwc
void my_draw_label(unsigned char *dst, unsigned char *src, int top, int left, int width, int height, int w, int h, unsigned char r, unsigned char g, unsigned char b)
{
    int left_top_index = 3 * width * top + 3 * left;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            dst[left_top_index + width * 3 * i + 3 * j] = src[w * 3 * i + 3 * j] * (r / 255.);
            dst[left_top_index + width * 3 * i + 3 * j + 1] = src[w * 3 * i + 3 * j + 1] * (g / 255.);
            dst[left_top_index + width * 3 * i + 3 * j + 2] = src[w * 3 * i + 3 * j + 2] * (b / 255.);
        }
    }
}

void my_draw_dets(unsigned char *src, int width, int height, detection *dets, int nboxes, float thresh, image **alphabet, char names[][20])
{
    for (int i = 0; i < nboxes; i++)
    {
        if (dets[i].max_prob < thresh)
            continue;
        box b = dets[i].bbox;
        int left = (b.x - b.w / 2.) * WIDTH;
        int right = (b.x + b.w / 2.) * WIDTH;
        int top = (b.y - b.h / 2.) * HEIGHT;
        int bot = (b.y + b.h / 2.) * HEIGHT;

        if (left < 0)
            left = 0;
        if (right > WIDTH - 1)
            right = WIDTH - 1;
        if (top < 0)
            top = 0;
        if (bot > HEIGHT - 1)
            bot = HEIGHT - 1;
        my_draw_box(src, width, height, left, right, top, bot, 255, 0, 0);
        my_draw_prob(src, width, height, left, right, top, bot, dets[i].max_prob, 0, 255, 0);
        image label = get_label(alphabet, names[dets[i].max_class], 7);
        unsigned char *l = malloc(label.h * label.w * label.c * sizeof(char));
        nchw2nhwc_float2char(l, label.data, label.w, label.h);
        my_draw_label(src, l, top, left, WIDTH, HEIGHT, label.w, label.h, 255, 0, 0);
        free(l);
        free_image(label);
    }
}