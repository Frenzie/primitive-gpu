// Minimal C header for the primitive-gpu C ABI (see c/test_capi.c).
#ifndef PRIMITIVE_GPU_H
#define PRIMITIVE_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct PgpuConfig {
    uint32_t internal_size;
    uint32_t output_size;
    uint32_t ss;
    int32_t mode;
    int32_t alpha;
    uint32_t shapes_per_frame;
    uint32_t seed;
    int32_t reuse;
    uint32_t max_shapes;
};

struct Session;

struct Session *pgpu_session_new(const struct PgpuConfig *config);
void pgpu_session_free(struct Session *sess);
int32_t pgpu_session_init(struct Session *sess, uint32_t w, uint32_t h,
                          const struct PgpuConfig *config);
const uint8_t *pgpu_process(struct Session *sess, const uint8_t *rgba,
                            uint32_t w, uint32_t h);
uint32_t pgpu_shape_count(const struct Session *sess);
void pgpu_output_size(const struct Session *sess, uint32_t *w, uint32_t *h);

#ifdef __cplusplus
}
#endif

#endif