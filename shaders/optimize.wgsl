// Workgroup-batched hill climbing: one workgroup = one chain with a shared
// incumbent. Each round, all 64 threads mutate a copy of the incumbent, score
// it, and write the candidate to shared memory; thread 0 then installs the
// best candidate if it beats the incumbent. This evaluates 64 mutations per
// round instead of one, and stops after `age` consecutive failed rounds
// (mirroring the Go maxAge cutoff).

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
@group(0) @binding(3) var<storage, read_write> winners: array<f32>; // 14 per workgroup

fn copy_row(dst: ptr<function, array<f32, 14>>, src: ptr<function, array<f32, 14>>) {
    for (var i = 0u; i < 14u; i = i + 1u) {
        (*dst)[i] = (*src)[i];
    }
}


// NOTE: a workgroup-cooperative batched variant (64 threads mutating a
// shared incumbent per round) proved unstable on Intel gen9 ANV (Mesa):
// intermittent workgroup kills / deadlocks around barrier-heavy loops.
// This version runs one independent chain per thread — no barriers in
// the mutation loop — which is deterministic and stable. The batched
// kernel is retained in git history for hardware without this quirk.
@compute @workgroup_size(1, 1, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let tid = gid.x;
    let out_row = tid * 14u;

    rs = mix_in(mix_in(params.frame_seed, params.step), tid * 0x9e3779b9u);

    // round 0: best of n_random random shapes
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

    // mutate-accept rounds with age cutoff (mirrors Go maxAge)
    var round = 0u;
    var age = 0u;
    loop {
        if (round >= params.rounds || age >= params.age) { break; }
        var c: array<f32, 14>;
        copy_row(&c, &best);
        mutate_row(&c);
        let sc = score_serial(&c);
        if (sc < best[0]) {
            copy_row(&best, &c);
            best[0] = sc;
            age = 0u;
        } else {
            age = age + 1u;
        }
        round = round + 1u;
    }

    for (var i = 0u; i < 14u; i = i + 1u) {
        winners[out_row + i] = best[i];
    }
    winners[out_row + 13] = f32(0x1u); // marker: chain completed
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

    // One pass, exact per-channel integer sums.
    var dr = 0i; var dg = 0i; var db = 0i;
    var csr = 0i; var csg = 0i; var csb = 0i;
    var area = 0i;
    var t2r = 0i; var t2g = 0i; var t2b = 0i;
    var c2r = 0i; var c2g = 0i; var c2b = 0i;
    var tcr = 0i; var tcg = 0i; var tcb = 0i;
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
                let tr = i32(tpx & 0xffu);
                let tg = i32((tpx >> 8u) & 0xffu);
                let tb = i32((tpx >> 16u) & 0xffu);
                let cr = i32(cpx & 0xffu);
                let cg = i32((cpx >> 8u) & 0xffu);
                let cb = i32((cpx >> 16u) & 0xffu);
                dr = dr + (tr - cr);
                dg = dg + (tg - cg);
                db = db + (tb - cb);
                csr = csr + cr;
                csg = csg + cg;
                csb = csb + cb;
                t2r = t2r + tr*tr; t2g = t2g + tg*tg; t2b = t2b + tb*tb;
                c2r = c2r + cr*cr; c2g = c2g + cg*cg; c2b = c2b + cb*cb;
                tcr = tcr + tr*cr; tcg = tcg + tg*cg; tcb = tcb + tb*cb;
                area = area + 1;
            }
            x = x + 1;
        }
        y = y + 1;
    }
    if (area < 1) {
        return 1e30;
    }
    let fa = f32(area);
    let colr = clamp((f32(dr) * a + f32(csr)) / fa, 0.0, 255.0);
    let colg = clamp((f32(dg) * a + f32(csg)) / fa, 0.0, 255.0);
    let colb = clamp((f32(db) * a + f32(csb)) / fa, 0.0, 255.0);
    (*c)[3] = colr;
    (*c)[4] = colg;
    (*c)[5] = colb;

    let sse_old = (f32(t2r) + f32(c2r) - 2.0 * f32(tcr))
                + (f32(t2g) + f32(c2g) - 2.0 * f32(tcg))
                + (f32(t2b) + f32(c2b) - 2.0 * f32(tcb));
    let sse_new = (f32(t2r) - 2.0 * colr * (f32(dr) + f32(csr)) + fa * colr * colr)
                + (f32(t2g) - 2.0 * colg * (f32(dg) + f32(csg)) + fa * colg * colg)
                + (f32(t2b) - 2.0 * colb * (f32(db) + f32(csb)) + fa * colb * colb);
    let d_sse = sse_new - sse_old;

    let cur = bitcast<f32>(params.cur_score);
    let cur_sse = (cur * 255.0) * (cur * 255.0) * n * 3.0;
    let new_sse = max(cur_sse + d_sse, 0.0);
    return sqrt(new_sse / (n * 3.0)) / 255.0;
}
