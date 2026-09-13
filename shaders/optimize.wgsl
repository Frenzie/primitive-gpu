// Hill-climb optimizer. One workgroup = one chain. Round 0: best of n_random
// random shapes. Each round: thread 0 mutates the shared best; all threads
// cooperatively score it; accept iff strictly better. This batches the Go
// code's serial try/rollback hill climbing into per-round parallel scoring.

struct Params {
    width: u32,
    height: u32,
    shape_type: i32, // 0 = combo
    alpha: i32,      // 0 = auto (mutated per round)
    rounds: u32,
    n_random: u32,
    frame_seed: u32,
    step: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> target: array<u32>;
@group(0) @binding(2) var<storage, read> canvas: array<u32>;
@group(0) @binding(3) var<storage, read_write> winners: array<f32>; // 14 per wg
@group(0) @binding(4) var<storage, read_write> scratch: array<atomic<u32>>;

var<workgroup> cand: array<f32, 14>;
var<workgroup> best: array<f32, 14>;
var<workgroup> red: array<atomic<i32>, 4>;
var<workgroup> red_area: array<atomic<i32>, 1>;
var<workgroup> red_err: array<atomic<i32>, 2>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let wg = gid.x / 64u;
    let base = wg * 14u;

    // ---- round 0: best of n_random random shapes ----
    if (tid == 0u) {
        best[0] = 1e30;
        var k = 0u;
        loop {
            if (k >= params.n_random) { break; }
            rs = mix_in(mix_in(params.frame_seed, params.step), wg * 0x9e3779b9u + k);
            var c: array<f32, 14>;
            random_shape(&c);
            // thread 0 scores alone (loop over pixels)
            let sc = score_thread0(&c);
            if (sc < best[0]) {
                copy14(&best, &c);
                best[0] = sc;
            }
            k = k + 1u;
        }
    }
    workgroupBarrier();

    // ---- hill-climb rounds ----
    var round = 0u;
    loop {
        if (round >= params.rounds) { break; }
        if (tid == 0u) {
            var c: array<f32, 14>;
            for (var i = 0u; i < 14u; i += 1u) {
                c[i] = best[i];
            }
            mutate_row(&c);
            for (var i = 0u; i < 14u; i += 1u) {
                cand[i] = c[i];
            }
        }
        workgroupBarrier();
        let sc = score_coop(&cand);
        if (tid == 0u && sc < best[0]) {
            for (var i = 0u; i < 14u; i += 1u) {
                best[i] = cand[i];
            }
            best[0] = sc;
        }
        workgroupBarrier();
        round = round + 1u;
    }

    if (tid < 14u) {
        winners[base + tid] = best[tid];
    }
}

fn copy14(dst: ptr<workgroup, array<f32, 14>>, src: ptr<workgroup, array<f32, 14>>) {
    for (var i = 0u; i < 14u; i += 1u) {
        (*dst)[i] = (*src)[i];
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
    (*c)[3] = 0.0;
    (*c)[4] = 0.0;
    (*c)[5] = 0.0;
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i += 1u) {
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
        for (var i = 1u; i < 4u; i += 1u) {
            p[i * 2u] = x1 + rf() * 40.0 - 20.0;
            p[i * 2u + 1u] = y1 + rf() * 40.0 - 20.0;
        }
    }
    for (var i = 0u; i < 8u; i += 1u) {
        (*c)[3 + i] = p[i];
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
    for (var i = 0u; i < 8u; i += 1u) {
        p[i] = (*c)[3 + i];
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
    for (var i = 0u; i < 8u; i += 1u) {
        (*c)[3 + i] = p[i];
    }
    // alpha mutation when auto
    if (params.alpha == 0) {
        (*c)[2] = f32(clampi(i32((*c)[2]) + i32(ru() % 21u) - 10, 1, 255));
    }
}

// Thread-0-only scoring used in round 0 (pixels looped serially).
fn score_thread0(c: ptr<function, array<f32, 14>>) -> f32 {
    let w = params.width;
    let h = params.height;
    let id = u32((*c)[1]);
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i += 1u) {
        p[i] = (*c)[3 + i];
    }
    let bb = bbox_of(id, p);
    let x0 = max(i32(bb.x), 0);
    let x1 = min(i32(bb.z), i32(w) - 1);
    let y0 = max(i32(bb.y), 0);
    let y1 = min(i32(bb.w), i32(h) - 1);
    var rs0 = 0.0;
    var gs0 = 0.0;
    var bs0 = 0.0;
    var area = 0.0;
    var err2 = 0.0;
    let a = f32(max((*c)[2], 1.0)) / 255.0;
    var y = y0;
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = (u32(y) * w + u32(x));
                let tpx = target[idx];
                let cpx = canvas[idx];
                let tr = f32(tpx & 0xffu);
                let tg = f32((tpx >> 8u) & 0xffu);
                let tb = f32((tpx >> 16u) & 0xffu);
                let cr = f32(cpx & 0xffu);
                let cg = f32((cpx >> 8u) & 0xffu);
                let cb = f32((cpx >> 16u) & 0xffu);
                rs0 = rs0 + (tr - cr);
                gs0 = gs0 + (tg - cg);
                bs0 = bs0 + (tb - cb);
                area = area + 1.0;
            }
            x = x + 1u;
        }
        y = y + 1u;
    }
    var col = vec3<f32>(0.0, 0.0, 0.0);
    if (area > 0.0) {
        col.x = clamp((rs0 / area) * a + 0.0, 0.0, 255.0) + 0.0;
        // base is canvas color; the delta formula folds it in below
        col.x = clamp(rs0 * a / area + cr0, 0.0, 255.0);
        col.y = clamp(gs0 * a / area + cg0, 0.0, 255.0);
        col.z = clamp(bs0 * a / area + cb0, 0.0, 255.0);
    }
    // second pass with fixed color
    y = y0;
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = (u32(y) * w + u32(x));
                let tpx = target[idx];
                let cpx = canvas[idx];
                let tr = f32(tpx & 0xffu);
                let tg = f32((tpx >> 8u) & 0xffu);
                let tb = f32((tpx >> 16u) & 0xffu);
                let cr = f32(cpx & 0xffu);
                let cg = f32((cpx >> 8u) & 0xffu);
                let cb = f32((cpx >> 16u) & 0xffu);
                let nr = tr - col.x;
                let ng = tg - col.y;
                let nb = tb - col.z;
                let or = tr - cr;
                let og = tg - cg;
                let ob = tb - cb;
                err2 = err2 + (nr * nr + ng * ng + nb * nb - or * or - og * og - ob * ob);
            }
            x = x + 1u;
        }
        y = y + 1u;
    }
    let n = f32(w * h);
    let total = f32((*c)[14 + 0]); // unused placeholder
    let cur = cur_score; // placeholder
    return score_from(err2, n, cur_score);
}

fn score_from(err2: f32, n: f32, cur: f32) -> f32 {
    // current score was RMSE over 4 channels; we track 3 here. Convert.
    // cur is in [0,1]. Reconstruct the current total SSE over 3 channels:
    let sse = pow(cur * 255.0, 2.0) * n * 3.0 + err2;
    return sqrt(sse / (n * 3.0)) / 255.0;
}