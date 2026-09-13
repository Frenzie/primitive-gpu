// Independent hill-climb: each thread owns a chain (one random start per
// thread, then mutate-accept rounds). No atomics or shared scoring; the
// per-thread final (score, params) is written to winners[] and a separate
// argmin kernel reduces across threads. This mirrors the Go design of
// BestHillClimbState (many independent chains) with chains per thread.

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
@group(0) @binding(3) var<storage, read_write> winners: array<f32>; // 14 per thread

fn copy_row(dst: ptr<function, array<f32, 14>>, src: ptr<function, array<f32, 14>>) {
    for (var i = 0u; i < 14u; i = i + 1u) {
        (*dst)[i] = (*src)[i];
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;

    rs = mix_in(mix_in(params.frame_seed, params.step), tid * 0x9e3779b9u);

    // round 0: best of n_random random shapes, scored serially by this thread
    var best: array<f32, 14>;
    best[0] = 1e30;
    var k = 0u;
    loop {
        if (k >= params.n_random) { break; }
        var c: array<f32, 14>;
        random_shape(&c);
        let sc = score_serial(&c);
        if (sc < best[0]) {
            copy_row(&best, &c);
            best[0] = sc;
        }
        k = k + 1u;
    }

    // mutate-accept rounds
    var round = 0u;
    loop {
        if (round >= params.rounds) { break; }
        var c: array<f32, 14>;
        copy_row(&c, &best);
        mutate_row(&c);
        let sc = score_serial(&c);
        if (sc < best[0]) {
            copy_row(&best, &c);
            best[0] = sc;
        }
        round = round + 1u;
    }

    for (var i = 0u; i < 14u; i = i + 1u) {
        winners[tid * 14u + i] = best[i];
    }
}

fn random_shape(c: ptr<function, array<f32, 14>>) {
    var t = params.shape_type;
    if (t == 0) {
        t = i32(ru() % 8u) + 1;
    }
    let w = f32(params.width);
    let h = f32(params.height);
    (*c)[0] = 1e30;
    (*c)[1] = f32(t);
    (*c)[2] = f32(select(params.alpha, 128, params.alpha == 0));
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p[i] = 0.0;
    }
    if (t == 1) {
        let x1 = f32(ru() % params.width);
        let y1 = f32(ru() % params.height);
        p[0] = x1;
        p[1] = y1;
        p[2] = x1 + f32(ru() % 31u) - 15.0;
        p[3] = y1 + f32(ru() % 31u) - 15.0;
        p[4] = x1 + f32(ru() % 31u) - 15.0;
        p[5] = y1 + f32(ru() % 31u) - 15.0;
        var g = 0u;
        loop {
            if (tri_valid(p) || g > 8u) { break; }
            mutate_tri(&p);
            g = g + 1u;
        }
    } else if (t == 2) {
        let x1 = f32(ru() % params.width);
        let y1 = f32(ru() % params.height);
        p[0] = x1;
        p[1] = y1;
        p[2] = clamp(x1 + f32(ru() % 32u) + 1.0, 0.0, w - 1.0);
        p[3] = clamp(y1 + f32(ru() % 32u) + 1.0, 0.0, h - 1.0);
    } else if (t == 3 || t == 4) {
        p[0] = f32(ru() % params.width);
        p[1] = f32(ru() % params.height);
        p[2] = f32(ru() % 32u) + 1.0;
        p[3] = f32(ru() % 32u) + 1.0;
        if (t == 4) {
            p[3] = p[2];
        }
    } else if (t == 5) {
        p[0] = f32(ru() % params.width);
        p[1] = f32(ru() % params.height);
        p[2] = f32(ru() % 32u) + 1.0;
        p[3] = f32(ru() % 32u) + 1.0;
        p[4] = f32(ru() % 360u);
    } else if (t == 6) {
        let x1 = rf() * w;
        let y1 = rf() * h;
        p[0] = x1;
        p[1] = y1;
        p[2] = x1 + rf() * 40.0 - 20.0;
        p[3] = y1 + rf() * 40.0 - 20.0;
        p[4] = p[2] + rf() * 40.0 - 20.0;
        p[5] = p[3] + rf() * 40.0 - 20.0;
        p[6] = 1.0;
        var g = 0u;
        loop {
            if (quad_valid(p) || g > 8u) { break; }
            mutate_quad(&p);
            g = g + 1u;
        }
    } else if (t == 7) {
        p[0] = rf() * w;
        p[1] = rf() * h;
        p[2] = rf() * 32.0 + 1.0;
        p[3] = rf() * 32.0 + 1.0;
        p[4] = rf() * 360.0;
    } else {
        let x1 = rf() * w;
        let y1 = rf() * h;
        p[0] = x1;
        p[1] = y1;
        for (var i = 1u; i < 4u; i = i + 1u) {
            p[i * 2u] = x1 + rf() * 40.0 - 20.0;
            p[i * 2u + 1u] = y1 + rf() * 40.0 - 20.0;
        }
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        (*c)[6 + i] = p[i];
    }
}

fn mutate_tri(p: ptr<function, array<f32, 8>>) {
    let w = f32(params.width);
    let h = f32(params.height);
    let which = ru() % 3u;
    let d1 = rns(16.0);
    let d2 = rns(16.0);
    if (which == 0u) {
        (*p)[0] = clamp((*p)[0] + d1, -16.0, w + 15.0);
        (*p)[1] = clamp((*p)[1] + d2, -16.0, h + 15.0);
    } else if (which == 1u) {
        (*p)[2] = clamp((*p)[2] + d1, -16.0, w + 15.0);
        (*p)[3] = clamp((*p)[3] + d2, -16.0, h + 15.0);
    } else {
        (*p)[4] = clamp((*p)[4] + d1, -16.0, w + 15.0);
        (*p)[5] = clamp((*p)[5] + d2, -16.0, h + 15.0);
    }
}

fn mutate_quad(p: ptr<function, array<f32, 8>>) {
    let w = f32(params.width);
    let h = f32(params.height);
    let which = ru() % 3u;
    let d1 = rns(16.0);
    let d2 = rns(16.0);
    if (which == 0u) {
        (*p)[0] = clamp((*p)[0] + d1, -16.0, w + 15.0);
        (*p)[1] = clamp((*p)[1] + d2, -16.0, h + 15.0);
    } else if (which == 1u) {
        (*p)[2] = clamp((*p)[2] + d1, -16.0, w + 15.0);
        (*p)[3] = clamp((*p)[3] + d2, -16.0, h + 15.0);
    } else {
        (*p)[4] = clamp((*p)[4] + d1, -16.0, w + 15.0);
        (*p)[5] = clamp((*p)[5] + d2, -16.0, h + 15.0);
    }
}

fn mutate_row(c: ptr<function, array<f32, 14>>) {
    let t = u32((*c)[1]);
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p[i] = (*c)[6 + i];
    }
    let w = f32(params.width);
    let h = f32(params.height);
    let which = ru() % 3u;
    if (t == 1u) {
        mutate_tri(&p);
        var g = 0u;
        loop {
            if (tri_valid(p) || g > 8u) { break; }
            mutate_tri(&p);
            g = g + 1u;
        }
    } else if (t == 2u) {
        if (which == 0u) {
            p[0] = clamp(p[0] + rns(16.0), 0.0, w - 1.0);
            p[1] = clamp(p[1] + rns(16.0), 0.0, h - 1.0);
        } else {
            p[2] = clamp(p[2] + rns(16.0), 0.0, w - 1.0);
            p[3] = clamp(p[3] + rns(16.0), 0.0, h - 1.0);
        }
    } else if (t == 3u || t == 4u) {
        if (which == 0u) {
            p[0] = clamp(p[0] + rns(16.0), 0.0, w - 1.0);
            p[1] = clamp(p[1] + rns(16.0), 0.0, h - 1.0);
        } else {
            p[2] = clamp(p[2] + rns(16.0), 1.0, w - 1.0);
            if (t == 4u) {
                p[3] = p[2];
            } else {
                p[3] = clamp(p[3] + rns(16.0), 1.0, h - 1.0);
            }
        }
    } else if (t == 5u) {
        if (which == 0u) {
            p[0] = clamp(p[0] + rns(16.0), 0.0, w - 1.0);
            p[1] = clamp(p[1] + rns(16.0), 0.0, h - 1.0);
        } else if (which == 1u) {
            p[2] = clamp(p[2] + rns(16.0), 1.0, w - 1.0);
            p[3] = clamp(p[3] + rns(16.0), 1.0, h - 1.0);
        } else {
            p[4] = p[4] + rns(32.0);
        }
    } else if (t == 6u) {
        mutate_quad(&p);
        var g = 0u;
        loop {
            if (quad_valid(p) || g > 8u) { break; }
            mutate_quad(&p);
            g = g + 1u;
        }
    } else if (t == 7u) {
        if (which == 0u) {
            p[0] = clamp(p[0] + rns(16.0), 0.0, w - 1.0);
            p[1] = clamp(p[1] + rns(16.0), 0.0, h - 1.0);
        } else if (which == 1u) {
            p[2] = clamp(p[2] + rns(16.0), 1.0, w - 1.0);
            p[3] = clamp(p[3] + rns(16.0), 1.0, h - 1.0);
        } else {
            p[4] = p[4] + rns(32.0);
        }
    } else {
        if (rf() < 0.25) {
            let i = ru() % 4u;
            let j = ru() % 4u;
            let tx = p[i * 2u];
            let ty = p[i * 2u + 1u];
            p[i * 2u] = p[j * 2u];
            p[i * 2u + 1u] = p[j * 2u + 1u];
            p[j * 2u] = tx;
            p[j * 2u + 1u] = ty;
        } else {
            let i = ru() % 4u;
            p[i * 2u] = clamp(p[i * 2u] + rns(16.0), -16.0, w + 15.0);
            p[i * 2u + 1u] = clamp(p[i * 2u + 1u] + rns(16.0), -16.0, h + 15.0);
        }
    }
    for (var i = 0u; i < 8u; i = i + 1u) {
        (*c)[6 + i] = p[i];
    }
    if (params.alpha == 0) {
        (*c)[2] = f32(clamp(i32((*c)[2]) + i32(ru() % 21u) - i32(10u), 1, 255));
    }
}

// Serial scoring on one thread. Returns the new global RMSE (3-channel) as
// score; uses cur (score in row 0) to reconstruct the current SSE.
fn score_serial(c: ptr<function, array<f32, 14>>) -> f32 {
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
                area = area + 1.0;
            }
            x = x + 1;
        }
        y = y + 1;
    }
    // optimal alpha-blended color over the region (matches computeColor)
    let cr0 = 0.0; // placeholder replaced below by canvas avg trick:
    // NOTE: exact color formula needs canvas color sums too, so do a second
    // accumulation for canvas sums on the region:
    var csr = 0.0;
    var csg = 0.0;
    var csb = 0.0;
    y = y0;
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = u32(y) * w + u32(x);
                let cpx = cur[idx];
                csr = csr + f32(cpx & 0xffu);
                csg = csg + f32((cpx >> 8u) & 0xffu);
                csb = csb + f32((cpx >> 16u) & 0xffu);
            }
            x = x + 1;
        }
        y = y + 1;
    }
    let colr = clamp((sr * a + csr) / max(area, 1.0), 0.0, 255.0);
    let colg = clamp((sg * a + csg) / max(area, 1.0), 0.0, 255.0);
    let colb = clamp((sb * a + csb) / max(area, 1.0), 0.0, 255.0);
    (*c)[3] = colr;
    (*c)[4] = colg;
    (*c)[5] = colb;

    var err2 = 0.0;
    y = y0;
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = u32(y) * w + u32(x);
                let tpx = tgt[idx];
                let cpx = cur[idx];
                let tr = f32(tpx & 0xffu);
                let tg = f32((tpx >> 8u) & 0xffu);
                let tb = f32((tpx >> 16u) & 0xffu);
                let cr = f32(cpx & 0xffu);
                let cg = f32((cpx >> 8u) & 0xffu);
                let cb = f32((cpx >> 16u) & 0xffu);
                let dr1 = tr - cr;
                let dg1 = tg - cg;
                let db1 = tb - cb;
                let dr2 = tr - colr;
                let dg2 = tg - colg;
                let db2 = tb - colb;
                err2 = err2 + (dr2 * dr2 + dg2 * dg2 + db2 * db2 - dr1 * dr1 - dg1 * dg1 - db1 * db1);
            }
            x = x + 1;
        }
        y = y + 1;
    }
    let cur = bitcast<f32>(params.cur_score);
    let sse = (cur * 255.0) * (cur * 255.0) * n * 3.0 + err2;
    let rmse = sqrt(max(sse, 0.0) / (n * 3.0)) / 255.0;
    return rmse;
}