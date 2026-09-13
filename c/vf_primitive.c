/*
 * vf_primitive.c — native ffmpeg video filter wrapping the primitive-gpu
 * engine via its C ABI. Compile against ffmpeg dev headers (libavfilter-dev,
 * libavutil-dev) and link the Rust static library:
 *
 *   make libfilter_primitive.so
 *
 *   ffmpeg -i in.mp4 -vf "primitive=shapes=8:reuse=1" out.mp4
 *
 * Requires ffmpeg >= 6.0 (filter registration API).
 */
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include "libavfilter/buffersrc.h"
#include "libavfilter/filters.h"
#include "libavfilter/video.h"
#include "primitive_gpu.h"

typedef struct PrimitiveContext {
    const AVClass *class;
    int internal_size;
    int output_size;
    int ss;
    int mode;
    int alpha;
    int shapes_per_frame;
    int64_t seed;
    int reuse;
    int max_shapes;
    struct Session *sess;
    uint8_t *rgba_in;   /* input converted to rgba8 */
    int in_w, in_h;
    int initialized;
} PrimitiveContext;

#define OFFSET(x) offsetof(PrimitiveContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption primitive_options[] = {
    { "internal", "internal optimization resolution", OFFSET(internal_size), AV_OPT_TYPE_INT, { .i64 = 128 }, 32, 512, FLAGS },
    { "output",   "output resolution (long edge)",    OFFSET(output_size),   AV_OPT_TYPE_INT, { .i64 = 0 },  0, 4096, FLAGS },
    { "ss",       "output supersampling",             OFFSET(ss),            AV_OPT_TYPE_INT, { .i64 = 1 },  1, 4,   FLAGS },
    { "mode",     "shape type 0..8",                  OFFSET(mode),          AV_OPT_TYPE_INT, { .i64 = 1 },  0, 8,   FLAGS },
    { "alpha",    "color alpha (0 = auto)",           OFFSET(alpha),         AV_OPT_TYPE_INT, { .i64 = 128 },0, 255, FLAGS },
    { "shapes",   "shapes added per frame",           OFFSET(shapes_per_frame), AV_OPT_TYPE_INT, { .i64 = 8 }, 1, 500, FLAGS },
    { "seed",     "RNG seed",                         OFFSET(seed),          AV_OPT_TYPE_INT64,{ .i64 = 42 },0, INT64_MAX, FLAGS },
    { "reuse",    "warm-start from previous frame",   OFFSET(reuse),         AV_OPT_TYPE_BOOL,{ .i64 = 1 },  0, 1,   FLAGS },
    { "max_shapes", "retained shape cap with reuse",  OFFSET(max_shapes),    AV_OPT_TYPE_INT, { .i64 = 200 },10, 2000, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(primitive);

static av_cold int init(AVFilterContext *ctx)
{
    PrimitiveContext *s = ctx->priv;
    s->sess = NULL;
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    PrimitiveContext *s = ctx->priv;
    if (s->sess) {
        pgpu_session_free(s->sess);
        s->sess = NULL;
    }
    av_freep(&s->rgba_in);
}

static int query_formats(const AVFilterContext *ctx, AVFilterFormatsConfig **cfg)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_RGBA, AV_PIX_FMT_NONE
    };
    return ff_set_common_formats_from_list2(ctx, cfg, pix_fmts);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    PrimitiveContext *s = ctx->priv;

    struct PgpuConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.internal_size    = (uint32_t)s->internal_size;
    /* 0 = match input resolution */
    cfg.output_size      = (uint32_t)(s->output_size > 0 ? s->output_size : inlink->w);
    cfg.ss               = (uint32_t)s->ss;
    cfg.mode             = s->mode;
    cfg.alpha            = s->alpha;
    cfg.shapes_per_frame = (uint32_t)s->shapes_per_frame;
    cfg.seed             = (uint32_t)s->seed;
    cfg.reuse            = s->reuse;
    cfg.max_shapes       = (uint32_t)s->max_shapes;

    if (s->sess) {
        pgpu_session_free(s->sess);
        s->sess = NULL;
    }
    s->sess = pgpu_session_new(&cfg);
    if (!s->sess) {
        av_log(ctx, AV_LOG_ERROR, "primitive-gpu: session creation failed\n");
        return AVERROR_EXTERNAL;
    }
    if (pgpu_session_init(s->sess, inlink->w, inlink->h, &cfg) != 0) {
        av_log(ctx, AV_LOG_ERROR, "primitive-gpu: engine init failed (no Vulkan adapter?)\n");
        pgpu_session_free(s->sess);
        s->sess = NULL;
        return AVERROR_EXTERNAL;
    }
    s->in_w = inlink->w;
    s->in_h = inlink->h;
    s->rgba_in = av_malloc((size_t)inlink->w * inlink->h * 4);
    if (!s->rgba_in) {
        return AVERROR(ENOMEM);
    }
    s->initialized = 1;
    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    PrimitiveContext *s = ctx->priv;

    if (!s->initialized) {
        av_frame_free(&in);
        return AVERROR_EXTERNAL;
    }

    /* RGBA input: copy plane data directly */
    av_image_copy_plane(s->rgba_in, in->linesize[0],
                        in->data[0], in->linesize[0],
                        inlink->w * 4, inlink->h);

    const uint8_t *out = pgpu_process(s->sess, s->rgba_in, inlink->w, inlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR_EXTERNAL;
    }
    uint32_t ow, oh;
    pgpu_output_size(s->sess, &ow, &oh);

    AVFrame *of = ff_get_video_buffer(outlink, ow, oh);
    if (!of) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(of, in);
    /* copy RGBA output row by row (source tightly packed) */
    for (uint32_t y = 0; y < oh; y++) {
        memcpy(of->data[0] + y * of->linesize[0], out + (size_t)y * ow * 4, ow * 4);
    }
    av_frame_free(&in);
    return ff_filter_frame(outlink, of);
}

static const AVFilterPad inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
};

const AVFilter ff_vf_primitive = {
    .name          = "primitive",
    .description   = NULL_IF_CONFIG_SMALL("Reproduce video with geometric primitives (GPU)"),
    .priv_size     = sizeof(PrimitiveContext),
    .priv_class    = &primitive_class,
    .init          = init,
    .uninit        = uninit,
    FILTER_INPUTS(inputs),
    FILTER_OUTPUTS(ff_video_default_filterpad),
    FILTER_QUERY_FUNC2(query_formats),
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};