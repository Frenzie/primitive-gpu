// Commit kernel: thread-per-pixel; row 0..3 = winner shape; blends winner
// into canvas using its computed color at row offsets 3..6.

struct Params {
    width: u32,
    height: u32,
    shape_type: i32,
    alpha: i32,
    rounds: u32,
    n_random: u32,
    frame_seed: u32,
    step: u32,
    num_shapes: u32,
    bg: u32,
    out_w: u32,
    out_h: u32,
    ss: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> target: array<u32>;
@group(0) @binding(2) var<storage, read_write> canvas: array<u32>;
@group(0) @binding(3) var<storage, read> winners: array<f32>;
@group(0) @binding(4) var<storage, read> shapes: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let w = params.width;
    let h = params.height;
    if (idx >= w * h) {
        return;
    }
    let x = idx % w;
    let y = idx / w;
    let row = winners[0..14];  // NOTE: replaced below with explicit loop
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i += 1u) {
        p[i] = winners[3 + i];
    }
    let id = u32(winners[1]);
    let alpha = u32(winners[2]);
    let r = winners[3 + 0];
    var inside = inside_of(id, p, f32(x), f32(y));
    if (inside) {
        // optimal color was computed in optimizer; recompute locally for this
        // pixel? No — recompute color globally requires global sums. Instead
        // the optimizer stores color in row[3..6]; we blend with that.
    }
    let cr = f32(canvas[idx] & 0xffu);
    let cg = f32((canvas[idx] >> 8u) & 0xffu);
    let cb = f32((canvas[idx] >> 16u) & 0xffu);
    // Blend (source-over, NRGBA) using the shape color from winners row.
    // winners[3..6] hold the OPTIMAL COLOR (r,g,b) chosen by the optimizer
    // and winners[2] the alpha. For simplicity the optimizer stores its final
    // color in p[0..3]... this design note is resolved in the final port:
    // the optimizer writes color into winners[3..6] and params in p[0..7].
}