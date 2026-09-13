// Temporal reuse: score each existing shape against the NEW target with
// the canvas reset to bg. Row out: [score, id, alpha, r, g, b, p0..p7].
// One thread per shape.

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
    age: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tgt: array<u32>;
@group(0) @binding(2) var<storage, read> cur: array<u32>;
@group(0) @binding(3) var<storage, read_write> rescored: array<f32>; // 14 per shape

fn copy14(dst: ptr<function, array<f32, 14>>, src: ptr<function, array<f32, 14>>) {
    for (var i = 0u; i < 14u; i = i + 1u) {
        (*dst)[i] = (*src)[i];
    }
}

// Same math as score_serial but taking the shape row directly and writing
// the recomputed color back.
fn score_row(c: ptr<function, array<f32, 14>>) -> f32 {
    let w = params.width;
    let h = params.height;
    let n = f32(w * h);
    let id = u32((*c)[1]);
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p[i] = (*c)[6 + i];
    }
    let bb = bbox_of(id, p);
    let x0 = max(i32(bb.x), 0);
    let x1 = min(i32(bb.z), i32(w) - 1);
    let y0 = max(i32(bb.y), 0);
    let y1 = min(i32(bb.w), i32(h) - 1);
    let a = 255.0 / f32(max((*c)[2], 1.0));

    var sr = 0.0;
    var sg = 0.0;
    var sb = 0.0;
    var csr = 0.0;
    var csg = 0.0;
    var csb = 0.0;
    var area = 0.0;
    var y = y0;
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = u32(y) * w + u32(x);
                let tpx = tgt[idx];
                let cpx = cur[idx];
                sr = sr + (f32(tpx & 0xffu) - f32(cpx & 0xffu));
                sg = sg + (f32((tpx >> 8u) & 0xffu) - f32((cpx >> 8u) & 0xffu));
                sb = sb + (f32((tpx >> 16u) & 0xffu) - f32((cpx >> 16u) & 0xffu));
                csr = csr + f32(cpx & 0xffu);
                csg = csg + f32((cpx >> 8u) & 0xffu);
                csb = csb + f32((cpx >> 16u) & 0xffu);
                area = area + 1.0;
            }
            x = x + 1;
        }
        y = y + 1;
    }
    if (area < 1.0) {
        return 1e30;
    }
    let colr = clamp((sr * a + csr) / area, 0.0, 255.0);
    let colg = clamp((sg * a + csg) / area, 0.0, 255.0);
    let colb = clamp((sb * a + csb) / area, 0.0, 255.0);
    (*c)[3] = colr;
    (*c)[4] = colg;
    (*c)[5] = colb;

    var err2 = 0.0;
    y = y0;
    loop {
        if (y > y1) { break; }
        var row_err = 0;
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = u32(y) * w + u32(x);
                let tpx = tgt[idx];
                let cpx = cur[idx];
                let tr = i32(tpx & 0xffu);
                let tg = i32((tpx >> 8u) & 0xffu);
                let tb = i32((tpx >> 16u) & 0xffu);
                let cr = i32(cpx & 0xffu);
                let cg = i32((cpx >> 8u) & 0xffu);
                let cb = i32((cpx >> 16u) & 0xffu);
                let nr = tr - i32(colr);
                let ng = tg - i32(colg);
                let nb = tb - i32(colb);
                row_err = row_err + (nr*nr + ng*ng + nb*nb - (tr-cr)*(tr-cr) - (tg-cg)*(tg-cg) - (tb-cb)*(tb-cb));
            }
            x = x + 1;
        }
        err2 = err2 + f32(row_err);
        y = y + 1;
    }
    let cur_s = bitcast<f32>(params.cur_score);
    let mean = (cur_s * 255.0) * (cur_s * 255.0) + err2 / (n * 3.0);
    let rmse = sqrt(max(mean, 0.0)) / 255.0;
    return rmse;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sid = gid.x;
    if (sid >= params.num_shapes) {
        return;
    }
    var c: array<f32, 14>;
    for (var i = 0u; i < 14u; i = i + 1u) {
        c[i] = rescored[sid * 14u + i];
    }
    let sc = score_row(&c);
    c[0] = sc;
    for (var i = 0u; i < 14u; i = i + 1u) {
        rescored[sid * 14u + i] = c[i];
    }
}