// Final render: one thread per supersampled output pixel; walks the shape
// list front-to-back and blends analytically (like Model.SVG/Draw).

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
@group(0) @binding(2) var<storage, read> cur: array<u32>;
@group(0) @binding(3) var<storage, read> winners: array<f32>;
@group(0) @binding(4) var<storage, read> shapes: array<f32>;
@group(0) @binding(5) var<storage, read_write> shapes_out_alias: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = params.out_w * params.ss;
    let h = params.out_h * params.ss;
    let px = gid.x;
    let py = gid.y;
    if (px >= w || py >= h) {
        return;
    }
    let idx = py * w + px;

    // map output sample to internal optimization space
    let fx = (f32(px) + 0.5) * f32(params.width) / f32(w);
    let fy = (f32(py) + 0.5) * f32(params.height) / f32(h);

    var r = f32(params.bg & 0xffu);
    var g = f32((params.bg >> 8u) & 0xffu);
    var b = f32((params.bg >> 16u) & 0xffu);

    var i = 0u;
    loop {
        if (i >= params.num_shapes) { break; }
        let row = i * 14u;
        let id = u32(shapes[row + 1u]);
        if (id >= 1u && id <= 8u) {
            // cheap bbox reject: skip shapes whose bbox can't contain px
            var p0: array<f32, 8>;
            for (var j = 0u; j < 8u; j = j + 1u) {
                p0[j] = shapes[row + 6u + j];
            }
            let b4 = bbox_of(id, p0);
            if (fx < b4.x || fx > b4.z || fy < b4.y || fy > b4.w) {
                i = i + 1u;
                continue;
            }
            let alpha = max(shapes[row + 2u], 1.0);
            var p: array<f32, 8>;
            for (var j = 0u; j < 8u; j = j + 1u) {
                p[j] = shapes[row + 6u + j];
            }
            if (inside_of(id, p, fx, fy)) {
                let sa = f32(u32(alpha)) / 255.0;
                let sr = shapes[row + 3u];
                let sg = shapes[row + 4u];
                let sb = shapes[row + 5u];
                r = sr * sa + r * (1.0 - sa);
                g = sg * sa + g * (1.0 - sa);
                b = sb * sa + b * (1.0 - sa);
            }
        }
        i = i + 1u;
    }

    shapes_out_alias[idx] = u32(r) | (u32(g) << 8u) | (u32(b) << 16u) | 0xff000000u;
}