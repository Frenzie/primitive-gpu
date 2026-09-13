# primitive-gpu

A GPU (Vulkan compute shader via [wgpu](https://wgpu.rs)) port of
[fogleman/primitive](https://github.com/fogleman/primitive), designed to also
work as an **optional effect for ffmpeg**.

Reproduces images with geometric primitives: starting from a solid background,
the optimizer repeatedly finds the single shape that most reduces the RMSE
against the target image, and commits it to the canvas. ~50–200 shapes produce
an abstract, recognizable reproduction.

## Status

Working end-to-end:

- **Image mode**: `primitive-gpu -i input.png -o output.png -n 200`
- **Video pipe mode**: rawvideo on stdin → rawvideo on stdout, for use
  between two ffmpeg processes (see below)

Benchmark on `monalisa.png`, 200 triangles (Intel Iris 540 iGPU, Mesa ANV
fallback adapter — see [Driver notes](#driver-notes)):

| | Go original (8 cores) | primitive-gpu (iGPU) |
|---|---|---|
| Time | 21.5 s | 81.6 s |
| RMSE vs target | 0.0395 | 0.0495 |

Honest assessment: on this machine the GPU port is currently *slower* than the
CPU original. The iGPU is gen9 Skylake and the current kernel runs hill
climbing serially per thread, mirroring the Go design. The redesign that makes
the GPU pay off (batching all mutations of a round across threads instead of
serial try/accept) is sketched in the kernel and is the next big step. On a
discrete GPU the same code is expected to win decisively, and the video mode
is where a GPU actually matters (thousands of runs).

## Build

Requires Rust (cargo) and a Vulkan-capable driver (or lavapipe/llvmpipe as a
software fallback).

    make            # cargo build --release
    make debug      # cargo build
    make clean

The binary lands at `target/release/primitive-gpu`.

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
| `--seed` | 42 | RNG seed for reproducible runs |

Diagnostics go to stderr; progress dots print per shape.

## Video pipe mode (ffmpeg)

The filter ships as a standalone executable reading rawvideo (rgb24) on stdin
and writing rawvideo on stdout, so it works with any ffmpeg build — no ffmpeg
patching required. Create the engine once; frames are processed independently
(no temporal reuse yet).

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

Quick smoke test with a synthetic source:

    ffmpeg -f lavfi -i testsrc2=size=320x180:rate=10:duration=1 \
             -pix_fmt rgb24 -f rawvideo - \
      | primitive-gpu --video --vw 320 --vh 180 -n 5 \
      | ffmpeg -f rawvideo -pix_fmt rgb24 -s 320x180 -r 10 -i - \
          -c:v libx264 -pix_fmt yuv420p out.mp4

`make video` runs exactly this.

| Video flag | Default | Description |
| --- | --- | --- |
| `--video` | off | enable pipe mode |
| `--vw` | 1920 | input frame width |
| `--vh` | 1080 | input frame height |
| `-n` | 100 | shapes per frame |
| `-m`, `-a`, `-r`, `--ss`, `--seed` | as above | per-frame optimization parameters |

## How it works

Three WGSL compute kernels per frame/step, all data resident on the GPU:

1. **optimize** — 64 × 64 independent hill-climb chains (one thread each).
   Each chain scores 16 random candidate shapes, then runs 24
   mutate/accept rounds. Scoring is analytic: bounding-box pixel pass with
   per-pixel inside tests (edge functions for triangles/polygons, quadratic
   forms for ellipses, sampled-segment distance for béziers) instead of the
   Go original's scanline rasterization. The optimal shape color is solved
   in closed form per candidate (same math as the Go `computeColor`).
   CPU reads back one 14-float row per chain (56 KB) and takes the argmin.
2. **commit** — one thread per pixel; blends the winning shape into the
   canvas at its precomputed optimal color.
3. **render** — once per image/frame: every supersampled output pixel walks
   the final shape list and blends analytically (equivalent to rendering the
   SVG output), then boxes down to the target resolution.

Shape parameters travel in 14-float rows: `[score, id, alpha, r, g, b, p0..p7]`
where `p` is the shape's parameters (vertex coordinates, radii, angle, …).

## Driver notes

Some drivers lose the device when compiling large shaders — on this machine,
the Mesa ANV driver for gen9 Intel (plain "Intel Iris 540") and llvmpipe both
fail, while the newer "Mesa Intel Iris 540" ANV build works. The engine
therefore probes **every** enumerated adapter against the real optimize
shader at startup and uses the first healthy one, printing what it picked or
skipped to stderr. If you see

    skipping adapter: ... (shader compile failed)

that is this mechanism working; update Mesa if your preferred adapter is
being skipped.

## Differences from the Go original

- No scanlines: shapes are evaluated with analytic per-pixel tests.
- No partial-image-difference scoring: a full-region pass per candidate,
  which is the right trade on parallel hardware.
- Hill climbing mutates the incumbent chain-best directly (batched
  try/accept per round) rather than Go's serial rollback.
- Output is PNG only (no SVG/GIF); the shape list is retained on the GPU.
- Go repo note: the original predates Go modules; run `go mod init
  github.com/fogleman/primitive` in `../primitive` to build it for A/B tests
  (`make ab` does this for you).

## Roadmap

- Batched per-round mutations (fully parallel hill climbing) — the main
  speedup
- Temporal reuse across video frames (seed frame N from frame N−1's shape
  list), behind a flag since it changes the look
- Native libavfilter (`-vf primitive=...`) via a small C ABI on top of the
  engine, after the pipe mode is tuned