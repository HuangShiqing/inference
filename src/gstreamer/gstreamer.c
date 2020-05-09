#include <malloc.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/app/gstappsink.h>

extern unsigned char **ringbuffer;
extern unsigned char **ringbuffer_processed;
extern unsigned int latest_index;
extern unsigned int latest_index_processed;
extern int ringbuffer_length;

#include "event.h"
extern event_parameter event1;
// event_parameter event2;
extern pthread_mutex_t mutex1;
extern pthread_mutex_t mutex2;

GstElement *pipeline;
GstElement *source, *app_sink;

static GstFlowReturn new_sample(GstElement *sink) //, CustomData *data
{
    // block waiting for the buffer
    GstSample *gstSample = gst_app_sink_pull_sample((GstAppSink *)(sink));

    if (!gstSample)
    {
        g_print("gstSample error\r\n");
        return GST_FLOW_ERROR;
    }

    GstBuffer *gstBuffer = gst_sample_get_buffer(gstSample);
    if (!gstBuffer)
    {
        g_print("gstBuffer error\r\n");
        return GST_FLOW_ERROR;
    }
    // retrieve
    GstMapInfo map;
    if (!gst_buffer_map(gstBuffer, &map, GST_MAP_READ))
    {
        g_print("GstMapInfo error\r\n");
        return GST_FLOW_ERROR;
    }
    void *gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
    // const int gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);
    if (!gstData)
    {
        g_print("gstData error\r\n");
        return GST_FLOW_ERROR;
    }

    // copy to next ringbuffer
    const unsigned int next_index = (latest_index + 1) % ringbuffer_length;
    memcpy(ringbuffer[next_index], map.data, 320 * 240 * 3);
    // printf("in new_sample ringbuffer: %p\r\n",ringbuffer);

    pthread_mutex_lock(&mutex1);
    latest_index = next_index;
    pthread_mutex_unlock(&mutex1);
    event_wake(&event1);//发出通知告诉inference线程可以进行处理了

    // gtk_widget_queue_draw(video_window);

    gst_buffer_unmap(gstBuffer, &map);
    gst_sample_unref(gstSample);

    return GST_FLOW_OK;
}

static gboolean print_field(GQuark field, const GValue *value, gpointer pfx)
{
    gchar *str = gst_value_serialize(value);

    g_print("%s  %15s: %s\n", (gchar *)pfx, g_quark_to_string(field), str);
    g_free(str);
    return TRUE;
}

static void print_caps(const GstCaps *caps, const gchar *pfx)
{
    guint i;

    g_return_if_fail(caps != NULL);

    if (gst_caps_is_any(caps))
    {
        g_print("%sANY\n", pfx);
        return;
    }
    if (gst_caps_is_empty(caps))
    {
        g_print("%sEMPTY\n", pfx);
        return;
    }

    for (i = 0; i < gst_caps_get_size(caps); i++)
    {
        GstStructure *structure = gst_caps_get_structure(caps, i);

        g_print("%s%s\n", pfx, gst_structure_get_name(structure));
        gst_structure_foreach(structure, print_field, (gpointer)pfx);
    }
}

static void print_pad_capabilities(GstElement *element, gchar *pad_name)
{
    GstPad *pad = NULL;
    GstCaps *caps = NULL;

    /* Retrieve pad */
    pad = gst_element_get_static_pad(element, pad_name);
    if (!pad)
    {
        g_printerr("Could not retrieve pad '%s'\n", pad_name);
        return;
    }

    /* Retrieve negotiated caps (or acceptable caps if negotiation is not finished yet) */
    caps = gst_pad_get_current_caps(pad);
    if (!caps)
        caps = gst_pad_query_caps(pad, NULL);

    /* Print and free */
    g_print("Caps for the %s pad:\n", pad_name);
    print_caps(caps, "      ");
    gst_caps_unref(caps);
    gst_object_unref(pad);
}

/* This function is called when an error message is posted on the bus */
static void error_cb(GstBus *bus, GstMessage *msg)
{
    GError *err;
    gchar *debug_info;

    /* Print error details on the screen */
    gst_message_parse_error(msg, &err, &debug_info);
    g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error(&err);
    g_free(debug_info);

    /* Set the pipeline to READY (which stops playback) */
    gst_element_set_state(pipeline, GST_STATE_READY);
}
/* This function is called when the pipeline changes states. We use it to
 * keep track of the current state. */
static void state_changed_cb(GstBus *bus, GstMessage *msg)
{
    /* We are only interested in state-changed messages from the pipeline */
    if (GST_MESSAGE_SRC(msg) == GST_OBJECT(pipeline))
    {
        GstState old_state, new_state, pending_state;
        gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
        g_print("\nPipeline state changed from %s to %s:\n",
                gst_element_state_get_name(old_state), gst_element_state_get_name(new_state));
        /* Print the current capabilities of the sink element */
        print_pad_capabilities(source, "src");
        print_pad_capabilities(app_sink, "sink");
    }
}

int gstreamer_init(int width, int height)
{
    GstBus *bus;
    GstStateChangeReturn ret;

    /* Initialize GStreamer */
    gst_init(NULL, NULL);

    /* Create the elements */
    source = gst_element_factory_make("autovideosrc", "source");
    // source = gst_element_factory_make("videotestsrc", "source");
    app_sink = gst_element_factory_make("appsink", "app_sink");

    /* Create the empty pipeline */
    pipeline = gst_pipeline_new("test-pipeline");

    if (!pipeline || !source || !app_sink)
    {
        g_printerr("Not all elements could be created.\n");
        return -1;
    }

    g_object_set(app_sink, "emit-signals", TRUE, NULL);
    g_signal_connect(app_sink, "new-sample", G_CALLBACK(new_sample), NULL);

    /* Build the pipeline */
    gst_bin_add_many(GST_BIN(pipeline), source, app_sink, NULL);

    GstCaps *caps;
    caps = gst_caps_new_simple("video/x-raw",
                               "format", G_TYPE_STRING, "RGB",
                               "width", G_TYPE_INT, width,
                               "height", G_TYPE_INT, height,
                               "framerate", GST_TYPE_FRACTION, 25, 1,
                               NULL);
    if (!gst_element_link_filtered(source, app_sink, caps))
    {
        g_warning("Failed to link element1 and element2!");
        return -1;
    }
    gst_caps_unref(caps);

    bus = gst_element_get_bus(pipeline);
    gst_bus_add_signal_watch(bus);
    g_signal_connect(G_OBJECT(bus), "message::error", (GCallback)error_cb, NULL);
    g_signal_connect(G_OBJECT(bus), "message::state-changed", (GCallback)state_changed_cb, NULL);

    event_init(&event1, 1);// 初始化inference事件

    /* Start playing */
    ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE)
    {
        g_printerr("Unable to set the pipeline to the playing state.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    /* Free resources */
    // gst_object_unref(bus);
    // gst_element_set_state(pipeline, GST_STATE_NULL);
    // gst_object_unref(pipeline);
    return 0;
}
