// Smoke test for the primitive-gpu C ABI: synthesizes moving frames,
// runs them through a session, and checks basic invariants.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "primitive_gpu.h"

static unsigned char lerp(unsigned char a, unsigned char b, float t) {
    return (unsigned char)(a + (b - a) * t);
}

int main(void) {
    struct PgpuConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.internal_size = 128;
    cfg.output_size = 320;
    cfg.ss = 1;
    cfg.mode = 1;
    cfg.alpha = 128;
    cfg.shapes_per_frame = 8;
    cfg.seed = 7;
    cfg.reuse = 1;
    cfg.max_shapes = 120;

    const uint32_t W = 320, H = 180, FRAMES = 6;
    unsigned char *in = malloc(W * H * 4);
    if (!in) return 1;

    struct Session *sess = pgpu_session_new(&cfg);
    if (!sess) { fprintf(stderr, "session_new failed\n"); return 1; }
    if (pgpu_session_init(sess, W, H, &cfg) != 0) {
        fprintf(stderr, "session_init failed\n"); return 1;
    }
    uint32_t ow, oh;
    pgpu_output_size(sess, &ow, &oh);
    printf("output size: %ux%u\n", ow, oh);

    for (uint32_t f = 0; f < FRAMES; f++) {
        float t = (float)f / (FRAMES - 1);
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                uint8_t *px = in + ((y * W + x) * 4);
                // animated gradient + moving blob
                px[0] = lerp(40, 220, x / (float)W);
                px[1] = lerp(60, 180, y / (float)H);
                px[2] = lerp(120, 40, t);
                float cx = (0.2f + 0.6f * t) * W, cy = 0.5f * H;
                float d = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if (d < 40.0f * 40.0f) { px[0] = 255; px[1] = 80; px[2] = 80; }
                px[3] = 255;
            }
        }
        const unsigned char *out = pgpu_process(sess, in, W, H);
        if (!out) { fprintf(stderr, "process failed on frame %u\n", f); return 1; }
        uint32_t shapes = pgpu_shape_count(sess);
        printf("frame %u: %u shapes\n", f, shapes);
        if (shapes == 0) { fprintf(stderr, "no shapes on frame %u\n", f); return 1; }
    }
    pgpu_session_free(sess);
    free(in);
    printf("capi OK\n");
    return 0;
}