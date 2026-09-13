// Commit kernel: blend the winning shape (winners[0..13] after argmin writes
// it there) into canvas at its stored color. One thread per pixel.

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
    cur_score: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tgt: array<u32>;
@group(0) @binding(2) var<storage, read_write> cur: array<u32>;
@group(0) @binding(3) var<storage, read_write> winners: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let w = params.width;
    let h = params.height;
    if (idx >= w * h) {
        return;
    }
    let x = f32(idx % w);
    let y = f32(idx / w);

    let id = u32(winners[1]);
    let alpha = u32(max(winners[2], 1.0));
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p[i] = winners[6 + i];
    }
    if (!inside_of(id, p, x, y)) {
        return;
    }
    // optimal color stored in winners[3..6] by the optimizer
    let sr = winners[3];
    let sg = winners[4];
    let sb = winners[5];
    let sa = f32(alpha) / 255.0;
    let cpx = cur[idx];
    let dr = f32(cpx & 0xffu);
    let dg = f32((cpx >> 8u) & 0xffu);
    let db = f32((cpx >> 16u) & 0xffu);
    let nr = u32(sr * sa + dr * (1.0 - sa));
    let ng = u32(sg * sa + dg * (1.0 - sa));
    let nb = u32(sb * sa + db * (1.0 - sa));
    cur[idx] = nr | (ng << 8u) | (nb << 16u) | 0xff000000u;
}