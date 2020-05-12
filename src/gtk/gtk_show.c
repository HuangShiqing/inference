#include <malloc.h>
#include <gtk/gtk.h>
#include <cairo.h> // 绘图所需要的头文件
#include <sys/prctl.h>

#include "my_common.h"

GdkPixbuf *pixbuf_original;
GdkPixbuf *pixbuf_processed;
GdkPixbuf *pixbuf_show;
extern unsigned char *buffer_original;
extern unsigned char *buffer_processed;
extern unsigned char *buffer_show;

extern pthread_mutex_t mutex2;

/* This function is called everytime the video window needs to be redrawn (due to damage/exposure,
 * rescaling, etc). GStreamer takes care of this in the PAUSED and PLAYING states, otherwise,
 * we simply draw a black rectangle to avoid garbage showing up. */
static gboolean draw_cb(GtkWidget *widget, cairo_t *cr)
{
    GtkAllocation allocation;
    /* Cairo is a 2D graphics library which we use here to clean the video window.
     * It is used by GStreamer for other reasons, so it will always be available to us. */
    gtk_widget_get_allocation(widget, &allocation);
    // cairo_set_source_rgb(cr, 255, 255, 255);

    pthread_mutex_lock(&mutex2); // 上锁失败代表别的线程在使用，则当前线程阻塞
    gdk_cairo_set_source_pixbuf(cr, pixbuf_show, 0, 0);
    pthread_mutex_unlock(&mutex2);

    cairo_rectangle(cr, 0, 0, allocation.width, allocation.height);
    cairo_fill(cr);

    return FALSE;
}
static void timer_cb(void *video_window)
{
    gtk_widget_queue_draw((GtkWidget *)video_window);
}

void *gtk_show_thread()
{
    prctl(PR_SET_NAME, "gtk_show"); // 给线程设置名字

    gtk_init(NULL, NULL);
    GtkWidget *video_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(video_window), "rsp csi camera");
    gtk_window_set_default_size(GTK_WINDOW(video_window), WIDTH, HEIGHT);
    gtk_window_set_position(GTK_WINDOW(video_window), GTK_WIN_POS_CENTER);
    // gtk_widget_set_double_buffered(video_window, FALSE);

    // g_signal_connect (video_window, "realize", G_CALLBACK (realize_cb), NULL);
    g_signal_connect(video_window, "draw", G_CALLBACK(draw_cb), NULL);
    gtk_widget_set_app_paintable(video_window, TRUE); // 允许窗口可以绘图

    // 准备图像缓存
    GdkPixbuf *pixbuf_init;

    pixbuf_init = gdk_pixbuf_new_from_file("./resource/dog.jpg",NULL);
    pixbuf_original = gdk_pixbuf_scale_simple(pixbuf_init, WIDTH, HEIGHT, GDK_INTERP_BILINEAR);
    buffer_original = (unsigned char *)malloc(WIDTH*HEIGHT*sizeof(char));
    buffer_original = gdk_pixbuf_get_pixels(pixbuf_original);

    pixbuf_init = gdk_pixbuf_new_from_file("./resource/dog.jpg",NULL);
    pixbuf_processed = gdk_pixbuf_scale_simple(pixbuf_init, WIDTH, HEIGHT, GDK_INTERP_BILINEAR);
    buffer_processed = (unsigned char *)malloc(WIDTH*HEIGHT*sizeof(char));
    buffer_processed = gdk_pixbuf_get_pixels(pixbuf_processed);

    pixbuf_init = gdk_pixbuf_new_from_file("./resource/dog.jpg",NULL);
    pixbuf_show = gdk_pixbuf_scale_simple(pixbuf_init, WIDTH, HEIGHT, GDK_INTERP_BILINEAR);
    buffer_show = (unsigned char *)malloc(WIDTH*HEIGHT*sizeof(char));
    buffer_show = gdk_pixbuf_get_pixels(pixbuf_show);

    // 每40ms启动一次刷新界面
    g_timeout_add(40, (GSourceFunc)timer_cb, (void *)video_window);
    // g_idle_add((GSourceFunc)gtk_widget_queue_draw, (void*)video_window);
    gtk_widget_show(video_window);
    // gtk_widget_queue_draw(video_window);
    // pid_t pid = getpid();
    // pthread_t tid = pthread_self();
    // printf("I am gtk_main, pid: %u, tid: 0x%x\r\n", (unsigned int)pid,(unsigned int)tid);
    gtk_main(); // 里面是个idle循环
    return 0;
}

void gtk_show_init()
{
    pthread_t gtk_show_tid;
    int r = pthread_create(&gtk_show_tid, 0, gtk_show_thread, NULL);
    if (r != 0)
        printf("gtk_show Thread creation failed");
}
