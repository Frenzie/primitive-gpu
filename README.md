# primitive-gpu

A GPU (Vulkan compute shader via [wgpu](https://wgpu.rs)) port of
[fogleman/primitive](https://github.com/fogleman/primitive), built to be used
with ffmpeg.

ffmpeg does not support runtime-loaded third-party filters, so the tool is a
standalone executable rather than a plugin: it reads rawvideo on stdin, writes
rawvideo on stdout, and slots between two ffmpeg processes. That constraint
drove the whole design — plain rgb24 pipe I/O, one engine instance reused
across frames, and a C ABI layer so the same engine can later be linked into
a native libavfilter build.

Reproduces images with geometric primitives: starting from a solid background,
the optimizer repeatedly finds the single shape that most reduces the RMSE
against the target image and commits it to the canvas. ~50–200 shapes produce
an abstract, recognizable reproduction.

## Status

Working end-to-end:

- **Image mode**: `primitive-gpu -i input.png -o output.png -n 200`
- **Video pipe mode**: rawvideo stdin → rawvideo stdout between ffmpeg
  processes, with optional **temporal reuse** across frames
- **C ABI library** (`libprimitive_gpu.a` / `.so`) with a passing smoke test
- **Native ffmpeg filter** (`vf_primitive.c`) — compiles anywhere
  `libavfilter-dev` is installed; not verifiable on the dev machine used here

Benchmark on `monalisa.png`, 200 triangles, Intel Iris 540 iGPU (Mesa gen9
ANV, via the automatic adapter fallback — see [Driver notes](#driver-notes)):

| | Go original (8 cores) | primitive-gpu (iGPU) |
|---|---|---|
| Time | 21.5 s | 47 s |
| RMSE vs target | 0.0395 | 0.0500 |

Honest assessment: on this machine the GPU port is slower than the CPU
original. The iGPU is gen9 Skylake, and the current kernel runs hill climbing
serially per thread (mirroring the Go design) because a workgroup-batched
variant — all 64 threads mutating a shared incumbent per round — proved
unstable on this driver (intermittent workgroup kills; details in
[Gen9 driver caveats](#gen9-driver-caveats)). On healthy hardware the batched
kernel is the intended configuration and is expected to win decisively; for
video, temporal reuse is the bigger win anyway.

## Build

Requires Rust (cargo) and a Vulkan-capable driver (lavapipe/llvmpipe works as
a software fallback).

    make            # release binary
    make capi       # C ABI static + shared libraries
    make capi-test  # build and run the C ABI smoke test
    make filter     # native ffmpeg filter (needs libavfilter-dev)
    make clean

The binary lands at `target/release/primitive-gpu`; the C ABI libraries at
`target/release/libprimitive_gpu.{a,so}`.

## Image mode

    primitive-gpu -i input.png -o output.png -n 200

| Flag | Default | Description |
| --- | --- | --- |
| `-i` | required | input image path |
| `-o` | required | output PNG path |
| `-n` | 100 | number of shapes |
| `-m` | 1 | shape type: 0=combo, 1=triangle, 2=rect, 3=ellipse, 4=circle, 5=rotated rect, 6=quadratic bezier, 7=rotated ellipse, 8=polygon |
| `-a` | 128 | color alpha; `0` lets the optimizer mutate alpha per shape |
| `-r` | 256 | internal optimization resolution (long edge) |
| `-s` | 1024 | output image size (long edge) |
| `--ss` | 2 | output supersampling factor |
| `--seed` | 42 | RNG seed |

Diagnostics go to stderr; progress dots print per shape.

## Video pipe mode (ffmpeg)

Transcode a video through the effect:

    ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - \
      | primitive-gpu --video --vw 1920 --vh 1080 -n 50 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - \
          -i input.mp4 -map 0:v -map 1:a? -c:v libx264 -crf 18 -c:a copy \
          output.mp4

Notes:

- The first ffmpeg must output exactly `--vw` × `--vh` rgb24 frames; add
  `-vf scale=1920:1080` before the output if the source differs.
- `-r 30` (fps) must match the source or A/V sync drifts; check with
  `ffprobe -show_entries stream=r_frame_rate input.mp4`.
- `-map 1:a?` copies the source audio track if present.

### Temporal reuse

`--reuse` warm-starts each frame from the previous frame's shape list: every
retained shape is rescored against the new frame, shapes that still improve
the score are re-committed first, and only then are new shapes added. This
makes output temporally stable (no per-frame shape churn) and much faster
per frame for similar quality. Two flags control it:

| Flag | Default | Description |
| --- | --- | --- |
| `--reuse` | off | enable warm-start from the previous frame |
| `--max-shapes` | 300 | cap on retained shapes (oldest/least-useful pruned) |

    ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 - \
      | primitive-gpu --video --vw 1920 --vh 1080 -n 10 --reuse --max-shapes 200 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 1920x1080 -r 30 -i - \
          -i input.mp4 -map 0:v -map 1:a? -c:v libx264 -crf 18 -c:a copy \
          output.mp4

With `-n 10 --reuse`, the first frame gets 10 shapes and later frames add 10
more on top of up to `--max-shapes` retained ones, so quality ramps up and
then stabilizes.

Quick synthetic smoke tests (also `make video` / `make video-reuse`):

    ffmpeg -f lavfi -i testsrc2=size=320x180:rate=10:duration=1 \
             -pix_fmt rgb24 -f rawvideo - \
      | primitive-gpu --video --vw 320 --vh 180 -n 5 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 320x180 -r 10 -i - \
          -c:v libx264 -pix_fmt yuv420p out.mp4

| Video flag | Default | Description |
| --- | --- | --- |
| `--video` | off | enable pipe mode |
| `--vw` / `--vh` | 1920 / 1080 | input frame dimensions |
| `-n` | 100 | shapes per frame |
| `--reuse`, `--max-shapes` | off / 300 | temporal reuse (above) |
| `-m`, `-a`, `-r`, `--ss`, `--seed` | as image mode | per-frame optimization |

## Native ffmpeg filter

For a true `-vf primitive=...` filter, ffmpeg must be rebuilt with the
filter linked in (runtime-loaded third-party filters are not supported by
ffmpeg). This repo ships everything needed:

    sudo apt install libavfilter-dev libavutil-dev
    make filter
    # then either link libfilter_primitive.so into a custom ffmpeg build, or
    # run ffmpeg with:
    ffmpeg -i in.mp4 -vf "primitive=shapes=8:reuse=1:max_shapes=200" out.mp4

Filter options mirror the CLI flags: `internal`, `output`, `ss`, `mode`,
`alpha`, `shapes`, `seed`, `reuse`, `max_shapes`. The filter consumes RGBA
frames and re-encodes whatever ffmpeg negotiates. Source: `c/vf_primitive.c`;
verified API surface: the C ABI test (`make capi-test`). Note the filter
itself could not be compile-tested on the machine this was developed on
(ffmpeg dev headers unavailable); it targets the ffmpeg 6.0+ filter API.

## C ABI

`libprimitive_gpu.{a,so}` expose a small stable interface
(`c/primitive_gpu.h`):

```c
struct Session *pgpu_session_new(const struct PgpuConfig *cfg);
int32_t pgpu_session_init(struct Session *s, uint32_t w, uint32_t h,
                          const struct PgpuConfig *cfg);
const uint8_t *pgpu_process(struct Session *s, const uint8_t *rgba,
                            uint32_t w, uint32_t h);
uint32_t pgpu_shape_count(const struct Session *s);
void pgpu_session_free(struct Session *s);
```

`pgpu_process` takes one RGBA8 frame and returns a pointer to the RGBA8
output (valid until the next call). `PgpuConfig.reuse` enables temporal
reuse internally, exactly like the CLI's `--reuse`. `c/test_capi.c` is a
complete example that animates frames and checks shape counts.

## How it works

Three WGSL compute kernels, all data resident on the GPU:

1. **optimize** — independent hill-climb chains, one thread each. Each chain
   scores 16 random candidate shapes, then runs mutate/accept rounds with a
   `maxAge`-style cutoff (stop after 16 consecutive failures). Scoring is
   analytic: bounding-box pixel pass with per-pixel inside tests (edge
   functions for triangles/polygons, quadratic forms for ellipses,
   sampled-segment distance for béziers) instead of the Go original's
   scanline rasterization. The optimal shape color is solved in closed form
   (same math as the Go `computeColor`), with integer row accumulation for
   exact error sums. The CPU reads back one 14-float row per chain and takes
   the argmin.
2. **rescore** (temporal reuse) — one thread per retained shape; rescores
   the previous frame's shapes against the new target over a reset canvas.
3. **commit** — one thread per pixel; blends the winning shape into the
   canvas at its precomputed optimal color.
4. **render** — once per frame: every output pixel walks the shape list
   (bbox-culled) and blends analytically (equivalent to rendering the SVG
   output), then downsamples.

Shape parameters travel in 14-float rows:
`[score, id, alpha, r, g, b, p0..p7]`.

## Driver notes

Some drivers lose the device when compiling large shaders. At startup the
engine probes **every** enumerated adapter against the real optimize shader
and uses the first healthy one, printing what it picked or skipped:

    skipping adapter: ... (shader compile failed)
    using adapter: Mesa Intel(R) Iris(R) Graphics 540 (SKL GT3) (IntegratedGpu)

Update Mesa if your preferred adapter is being skipped.

### Gen9 driver caveats

Development happened on Intel gen9 (Skylake iGPU) with Mesa's ANV driver,
which exhibited two issues that shaped the current design:

1. **Large-shader device loss** — compiling the workgroup-batched optimizer
   kills the device. Mitigated by the adapter-probe fallback above.
2. **Intermittent workgroup kills under barrier-heavy loops** — the batched
   kernel (64 threads cooperatively mutating a shared incumbent) produced
   dead workgroups nondeterministically, even after eliminating every
   identifiable barrier-divergence hazard. The shipped kernel therefore uses
   barrier-free per-thread chains; a killed chain simply yields a stale row
   that the CPU argmin skips (chains are marked on completion).

If you see `render produced no output (GPU kill?)`, the render dispatch was
killed; it is retried automatically (up to 5 attempts in video mode).

## Differences from the Go original

- No scanlines: shapes are evaluated with analytic per-pixel tests.
- No partial-image-difference scoring: exact integer accumulation over the
  candidate region, which is the right trade on parallel hardware.
- Output is PNG only (no SVG/GIF); the shape list stays on the GPU.
- Temporal reuse for video has no Go counterpart.
- The Go repo predates Go modules; `make go` initializes it for A/B runs.

## Roadmap

- Re-enable the workgroup-batched kernel behind a flag once tested on
  healthy hardware (the design is sound; gen9 ANV is the blocker)
- Native filter verification on a machine with ffmpeg dev headers
- Expose per-chain tuning (rounds, n_random, age) as CLI flags