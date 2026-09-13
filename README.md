# primitive-gpu

GPU port of [fogleman/primitive](https://github.com/fogleman/primitive) —
reproduces images with geometric primitives, usable as an ffmpeg video
effect. Vulkan compute shaders via [wgpu](https://wgpu.rs).

ffmpeg can't load third-party filters at runtime, so instead of a plugin this
is a standalone executable: rawvideo in on stdin, rawvideo out on stdout,
between two ffmpeg processes. A C ABI wraps the same engine for use in a
custom ffmpeg build or other C programs.

## Build

Needs Rust and a Vulkan driver (lavapipe works as fallback).

    make            # binary at target/release/primitive-gpu
    make capi       # C libraries: target/release/libprimitive_gpu.{a,so}
    make capi-test  # build + run C ABI smoke test
    make filter     # native ffmpeg filter (needs libavfilter-dev)

## Usage

### Images

    primitive-gpu -i input.png -o output.png -n 200

| Flag | Default | Description |
| --- | --- | --- |
| `-i` / `-o` | — | input / output PNG |
| `-n` | 100 | number of shapes |
| `-m` | 1 | shape: 0=combo, 1=triangle, 2=rect, 3=ellipse, 4=circle, 5=rotated rect, 6=quadratic, 7=rotated ellipse, 8=polygon |
| `-a` | 128 | color alpha; `0` = optimizer picks |
| `-r` | 256 | internal optimization resolution |
| `-s` | 1024 | output size (long edge) |
| `--ss` | 2 | output supersampling |
| `--seed` | 42 | RNG seed |

### Video

Pipe between two ffmpeg processes:

    ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - \
      | primitive-gpu --video --vw 1920 --vh 1080 -n 50 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - \
          -i input.mp4 -map 0:v -map 1:a? -c:v libx264 -crf 18 -c:a copy \
          output.mp4

The source must be scaled to exactly `--vw`×`--vh` rgb24 (`-vf scale=...` if
needed), and `-r` must match the source fps or audio drifts.

Temporal reuse (`--reuse`) carries the previous frame's shapes over: each is
rescored against the new frame, the useful ones are re-committed, then new
shapes are added. Stable output across frames, cheaper per frame. `--max-shapes`
(default 300) caps the retained list.

    ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - \
      | primitive-gpu --video --vw 1920 --vh 1080 -n 10 --reuse --max-shapes 200 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - \
          -i input.mp4 -map 0:v -map 1:a? -c:v libx264 -crf 18 -c:a copy \
          output.mp4

With `-n 10 --reuse --max-shapes 200`: frame 1 gets 10 shapes, later frames
keep up to 200 and add 10 more each.

`make video` and `make video-reuse` run synthetic smoke tests.

### Native ffmpeg filter

ffmpeg filters must be compiled in, so a custom ffmpeg build is required for
`-vf primitive=...`:

    sudo apt install libavfilter-dev libavutil-dev
    make filter
    # link libfilter_primitive.so into an ffmpeg build, then:
    ffmpeg -i in.mp4 -vf "primitive=shapes=8:reuse=1:max_shapes=200" out.mp4

Options mirror the CLI flags (`internal`, `output`, `ss`, `mode`, `alpha`,
`shapes`, `seed`, `reuse`, `max_shapes`). Source in `c/vf_primitive.c`,
targets the ffmpeg 6.0+ filter API. Not compile-tested here (no libavfilter
dev headers on this machine); the C ABI itself is covered by `make capi-test`.

### C ABI

`c/primitive_gpu.h`:

```c
struct Session *pgpu_session_new(const struct PgpuConfig *cfg);
int32_t pgpu_session_init(struct Session *s, uint32_t w, uint32_t h,
                          const struct PgpuConfig *cfg);
const uint8_t *pgpu_process(struct Session *s, const uint8_t *rgba,
                            uint32_t w, uint32_t h);
uint32_t pgpu_shape_count(const struct Session *s);
void pgpu_session_free(struct Session *s);
```

RGBA8 frame in, RGBA8 frame out. `PgpuConfig.reuse` is the same temporal
reuse as `--reuse`. See `c/test_capi.c` for a working example.

## Performance

monalisa.png, 200 triangles, Intel Iris 540 iGPU:

| | Go original (8 cores) | primitive-gpu |
|---|---|---|
| Time | 21.5 s | 47 s |
| RMSE | 0.0395 | 0.0500 |

The iGPU loses to the CPU. The kernel runs hill climbing serially per thread
like the Go original; a workgroup-batched variant (64 threads mutating a
shared incumbent per round) was tried but is unstable on gen9 Mesa — see
below. On better GPUs it should win, and temporal reuse matters more for
video anyway.

## How it works

Compute kernels, everything resident on the GPU:

- **optimize** — one hill-climb chain per thread: 16 random candidates, then
  mutate/accept rounds with an age cutoff. Shapes are scored analytically
  (per-pixel inside tests over a bounding box, no scanlines); the optimal
  color is solved in closed form like the Go `computeColor`. CPU reads back
  one 14-float row per chain and takes the argmin.
- **rescore** — with reuse, rescores last frame's shapes against the new
  frame over a reset canvas.
- **commit** — blends the winning shape into the canvas.
- **render** — every output pixel walks the shape list (bbox-culled) and
  blends analytically.

Shape row: `[score, id, alpha, r, g, b, p0..p7]`, 14 floats.

## Driver notes

Some drivers lose the device compiling large shaders. The engine probes every
adapter with the real optimize shader at startup and uses the first healthy
one; skipped adapters are logged. Update Mesa if yours gets skipped.

The gen9 Mesa ANV driver (development machine) additionally kills
workgroups/barrier-heavy kernels nondeterministically. The shipped kernel
avoids barriers entirely for this reason, dead chains are skipped by the CPU
argmin, and a killed render is retried. If you see
`render produced no output (GPU kill?)` repeatedly, your driver is the
problem — the batched kernel should be fine on healthier hardware.

## Differences from the Go original

- Analytic per-pixel shape tests instead of scanline rasterization.
- PNG output only; the shape list stays on the GPU.
- Temporal reuse for video (new).
- `make go` initializes the Go repo (it predates Go modules) for A/B runs.