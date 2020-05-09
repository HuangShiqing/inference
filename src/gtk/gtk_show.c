#include <malloc.h>
#include <gtk/gtk.h>
#include <cairo.h> // 绘图所需要的头文件
#include <sys/prctl.h>

GdkPixbuf **pixbuf;
GdkPixbuf **pixbuf_processed;

extern pthread_mutex_t mutex2;

extern unsigned char **ringbuffer;
extern unsigned char **ringbuffer_processed;
extern unsigned int latest_index;
extern unsigned int latest_index_processed;
extern int ringbuffer_length;

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
    const unsigned int next_index = latest_index_processed;
    pthread_mutex_unlock(&mutex2);

    gdk_cairo_set_source_pixbuf(cr, pixbuf_processed[next_index], 0, 0);
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
    int ringbuffer_length = 2;
    prctl(PR_SET_NAME, "gtk_show"); // 给线程设置名字

    gtk_init(NULL, NULL);
    GtkWidget *video_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(video_window), "rsp csi camera");
    gtk_window_set_default_size(GTK_WINDOW(video_window), 320, 240);
    gtk_window_set_position(GTK_WINDOW(video_window), GTK_WIN_POS_CENTER);
    // gtk_widget_set_double_buffered(video_window, FALSE);

    // g_signal_connect (video_window, "realize", G_CALLBACK (realize_cb), NULL);
    g_signal_connect(video_window, "draw", G_CALLBACK(draw_cb), NULL);
    gtk_widget_set_app_paintable(video_window, TRUE); // 允许窗口可以绘图

    // 准备图像缓存
    GdkPixbuf *src_pixbuf;
    pixbuf = (GdkPixbuf **)malloc(ringbuffer_length * sizeof(GdkPixbuf *));
    pixbuf_processed = (GdkPixbuf **)malloc(ringbuffer_length * sizeof(GdkPixbuf *));
    ringbuffer = (unsigned char **)malloc(ringbuffer_length * sizeof(unsigned char *));
    ringbuffer_processed = (unsigned char **)malloc(ringbuffer_length * sizeof(unsigned char *));
    for (int i = 0; i < ringbuffer_length; i++)
    {
        src_pixbuf = gdk_pixbuf_new_from_file("./resource/dog.jpg", NULL);
        // 指定图片大小
        pixbuf[i] = gdk_pixbuf_scale_simple(src_pixbuf, 320, 240, GDK_INTERP_BILINEAR);
        ringbuffer[i] = gdk_pixbuf_get_pixels(pixbuf[i]);
    }
    for (int i = 0; i < ringbuffer_length; i++)
    {
        src_pixbuf = gdk_pixbuf_new_from_file("./resource/dog.jpg", NULL);
        // 指定图片大小
        pixbuf_processed[i] = gdk_pixbuf_scale_simple(src_pixbuf, 320, 240, GDK_INTERP_BILINEAR);
        ringbuffer_processed[i] = gdk_pixbuf_get_pixels(pixbuf_processed[i]);
    }

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
//TODO:把width和height作为参数
void gtk_show_init() //int width, int height, int ringbuffer_length
{
    pthread_t gtk_show_tid;
    int r = pthread_create(&gtk_show_tid, 0, gtk_show_thread, NULL);
    if (r != 0)
        printf("gtk_show Thread creation failed");
}
