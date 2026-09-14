// Split-round hill climbing without workgroup barriers. The gen9 Mesa ANV
// driver kills barrier-heavy compute, so round-level parallelism is
// expressed as separate dispatches (global sync between them) instead of
// workgroup barriers (intra-dispatch sync):
//
//   ssereset  — clear per-step accumulators
//   propose   — 1 thread per chain: mutate incumbent into candidate buf
//               (or generate a fresh random shape in round 0)
//   eval      — 64 threads per chain stripe-scan the candidate bbox in ONE
//               fused pass (target sums, canvas sums); 4096 chains → 262K
//               threads live at once
//   accept    — 1 thread per chain: reduce partial sums, solve the optimal
//               color in closed form, compute the error analytically from
//               the sums, accept/reject with age tracking
//   winnersel — 1 thread per 64 chains: global argmin for the step
//
// Each dispatch reads only buffers written by the previous one, so no
// barriers are needed inside any kernel.

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
    chains: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tgt: array<u32>;
@group(0) @binding(2) var<storage, read> cur: array<u32>;
@group(0) @binding(3) var<storage, read_write> chain: array<f32>; // 16 per chain
@group(0) @binding(4) var<storage, read_write> cand: array<f32>; // 16 per chain
@group(0) @binding(5) var<storage, read_write> part: array<atomic<u32>>; // partials
@group(0) @binding(6) var<storage, read_write> shapes: array<f32>;
@group(0) @binding(7) var<storage, read_write> sse_buf: array<atomic<u32>>;

// chain row (16 f32): [score, id, alpha, r, g, b, p0..p7, age]
// part layout, 16 u32 per chain:
//   0..3 deltas Σ(t-c) r,g,b
//   3..6 canvas sums Σc r,g,b
//   6 area
//   7..10 Σt² r,g,b
//   10..13 Σc² r,g,b
//   13..16 Σtc r,g,b
//   0..3 target sums r,g,b
//   3..6 canvas sums r,g,b
//   6 area
//   7 Σt² (target square sum, integer, exact)
//   8 Σc² (canvas square sum, integer, exact)
//   9 Σt·c (cross sum, integer, exact)

fn row_score(r: ptr<function, array<f32, 16>>) -> f32 {
    return (*r)[0];
}

// ---------- ssereset ----------
@compute @workgroup_size(64)
fn ssereset(@builtin(global_invocation_id) gid: vec3<u32>) {
    // chain partials: zero; winnersel scratch: +inf sentinel
    let i = gid.x;
    let n_part = params.chains * 16u;
    if (i < n_part) {
        atomicStore(&part[i], 0u);
    }
    let scratch_lo = params.chains * 16u;
    if (i >= scratch_lo && i < scratch_lo + 4096u) {
        atomicStore(&part[i], 0x7f800000u); // +inf
    }
}

// ---------- propose ----------
@compute @workgroup_size(64)
fn propose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.chains) {
        return;
    }
    rs = mix_in(mix_in(mix_in(params.frame_seed, params.step), tid * 0x9e3779b9u), params.rounds * 0x85ebca6bu);
    let base = tid * 16u;
    var c: array<f32, 16>;
    if (params.rounds == 0u) {
        random_shape(&c);
    } else {
        for (var i = 0u; i < 16u; i = i + 1u) {
            c[i] = chain[base + i];
        }
        mutate_row(&c);
    }
    for (var i = 0u; i < 16u; i = i + 1u) {
        cand[base + i] = c[i];
    }
}

// ---------- eval ----------
// 64 threads per chain stripe-scan the candidate's bbox. One fused pass:
// accumulate target sums, canvas sums, and per-row exact error deltas
// atomically into part[].
@compute @workgroup_size(64)
fn eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let cid = tid / 64u; // chain
    let lane = tid % 64u;
    let base = cid * 16u;
    let id = u32(cand[base + 1]);
    if (id == 0u) {
        return;
    }
    var p: array<f32, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        p[i] = cand[base + 6 + i];
    }
    let bb = bbox_of(id, p);
    let w = params.width;
    let h = params.height;
    let x0 = max(i32(bb.x), 0);
    let x1 = min(i32(bb.z), i32(w) - 1);
    let y0 = max(i32(bb.y), 0);
    let y1 = min(i32(bb.w), i32(h) - 1);
    if (x1 < x0 || y1 < y0) {
        return;
    }
    let a = 255.0 / max(cand[base + 2], 1.0);

    // per-lane integer accumulators (exact, two's-complement in u32)
    var sr = 0i;
    var sg = 0i;
    var sb = 0i;
    var csr = 0i;
    var csg = 0i;
    var csb = 0i;
    var area = 0i;
    var t2r = 0i; var t2g = 0i; var t2b = 0i;
    var c2r = 0i; var c2g = 0i; var c2b = 0i;
    var tcr = 0i; var tcg = 0i; var tcb = 0i;
    var y = y0 + i32(lane);
    loop {
        if (y > y1) { break; }
        var x = x0;
        loop {
            if (x > x1) { break; }
            if (inside_of(id, p, f32(x), f32(y))) {
                let idx = u32(y) * params.width + u32(x);
                let tpx = tgt[idx];
                let cpx = cur[idx];
                let trn = i32(tpx & 0xffu);
                let tgn = i32((tpx >> 8u) & 0xffu);
                let tbn = i32((tpx >> 16u) & 0xffu);
                let crn = i32(cpx & 0xffu);
                let cgn = i32((cpx >> 8u) & 0xffu);
                let cbn = i32((cpx >> 16u) & 0xffu);
                sr = sr + (trn - crn);
                sg = sg + (tgn - cgn);
                sb = sb + (tbn - cbn);
                csr = csr + crn;
                csg = csg + cgn;
                csb = csb + cbn;
                t2r = t2r + trn * trn; t2g = t2g + tgn * tgn; t2b = t2b + tbn * tbn;
                c2r = c2r + crn * crn; c2g = c2g + cgn * cgn; c2b = c2b + cbn * cbn;
                tcr = tcr + trn * crn; tcg = tcg + tgn * cgn; tcb = tcb + tbn * cbn;
                area = area + 1;
            }
            x = x + 1;
        }
        y = y + 64;
    }
    let pbase = cid * 16u;
    atomicAdd(&part[pbase + 0], bitcast<u32>(sr));
    atomicAdd(&part[pbase + 1], bitcast<u32>(sg));
    atomicAdd(&part[pbase + 2], bitcast<u32>(sb));
    atomicAdd(&part[pbase + 3], bitcast<u32>(csr));
    atomicAdd(&part[pbase + 4], bitcast<u32>(csg));
    atomicAdd(&part[pbase + 5], bitcast<u32>(csb));
    atomicAdd(&part[pbase + 6], u32(area));
    atomicAdd(&part[pbase + 7], bitcast<u32>(t2r));
    atomicAdd(&part[pbase + 8], bitcast<u32>(t2g));
    atomicAdd(&part[pbase + 9], bitcast<u32>(t2b));
    atomicAdd(&part[pbase + 10], bitcast<u32>(c2r));
    atomicAdd(&part[pbase + 11], bitcast<u32>(c2g));
    atomicAdd(&part[pbase + 12], bitcast<u32>(c2b));
    atomicAdd(&part[pbase + 13], bitcast<u32>(tcr));
    atomicAdd(&part[pbase + 14], bitcast<u32>(tcg));
    atomicAdd(&part[pbase + 15], bitcast<u32>(tcb));
}
fn random_shape(c: ptr<function, array<f32, 16>>) {
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

fn mutate_row(c: ptr<function, array<f32, 16>>) {
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
// ---------- accept ----------
@compute @workgroup_size(64)
fn accept(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.chains) {
        return;
    }
    let base = tid * 16u;
    let pbase = tid * 16u;
    let area = f32(atomicLoad(&part[pbase + 6]));
    if (area < 1.0) {
        chain[base + 15] = chain[base + 15] + 1.0;
        if (chain[base + 15] >= f32(params.age)) {
            chain[base + 1] = 0.0;
        }
        return;
    }
    let id = u32(cand[base + 1]);
    if (id == 0u) {
        return;
    }
    let dr = f32(bitcast<i32>(atomicLoad(&part[pbase + 0])));
    let dg = f32(bitcast<i32>(atomicLoad(&part[pbase + 1])));
    let db = f32(bitcast<i32>(atomicLoad(&part[pbase + 2])));
    let csr = f32(bitcast<i32>(atomicLoad(&part[pbase + 3])));
    let csg = f32(bitcast<i32>(atomicLoad(&part[pbase + 4])));
    let csb = f32(bitcast<i32>(atomicLoad(&part[pbase + 5])));
    let a = 255.0 / max(cand[base + 2], 1.0);
    let colr = clamp((dr * a + csr) / area, 0.0, 255.0);
    let colg = clamp((dg * a + csg) / area, 0.0, 255.0);
    let colb = clamp((db * a + csb) / area, 0.0, 255.0);

    let t2r = f32(bitcast<i32>(atomicLoad(&part[pbase + 7])));
    let t2g = f32(bitcast<i32>(atomicLoad(&part[pbase + 8])));
    let t2b = f32(bitcast<i32>(atomicLoad(&part[pbase + 9])));
    let c2r = f32(bitcast<i32>(atomicLoad(&part[pbase + 10])));
    let c2g = f32(bitcast<i32>(atomicLoad(&part[pbase + 11])));
    let c2b = f32(bitcast<i32>(atomicLoad(&part[pbase + 12])));
    let tcr = f32(bitcast<i32>(atomicLoad(&part[pbase + 13])));
    let tcg = f32(bitcast<i32>(atomicLoad(&part[pbase + 14])));
    let tcb = f32(bitcast<i32>(atomicLoad(&part[pbase + 15])));
    let sse_old = (t2r + c2r - 2.0 * tcr)
                + (t2g + c2g - 2.0 * tcg)
                + (t2b + c2b - 2.0 * tcb);
    let sse_new = (t2r - 2.0 * colr * (dr + csr) + area * colr * colr)
                + (t2g - 2.0 * colg * (dg + csg) + area * colg * colg)
                + (t2b - 2.0 * colb * (db + csb) + area * colb * colb);
    let d_sse = sse_new - sse_old;

    let cur_score = chain[base + 0];
    if (d_sse < 0.0 && cur_score == cur_score && abs(cur_score) != 3.4028235e38) {
        let n = f32(params.width * params.height) * 3.0;
        let cur_sse = cur_score * cur_score * 255.0 * 255.0 * n;
        let new_sse = max(cur_sse + d_sse, 0.0);
        let new_rmse = sqrt(new_sse / n) / 255.0;
        chain[base + 0] = new_rmse;
        chain[base + 3] = colr;
        chain[base + 4] = colg;
        chain[base + 5] = colb;
        chain[base + 1] = cand[base + 1];
        chain[base + 2] = cand[base + 2];
        for (var i = 0u; i < 8u; i = i + 1u) {
            chain[base + 6 + i] = cand[base + 6 + i];
        }
        chain[base + 15] = 0.0;
    } else {
        chain[base + 15] = chain[base + 15] + 1.0;
        if (chain[base + 15] >= f32(params.age)) {
            chain[base + 1] = 0.0;
        }
    }
}

// ---------- chainsinit ----------
// One thread per chain: generate a random shape as the chain's starting
// incumbent (score filled in by the first eval/accept round).
@compute @workgroup_size(64)
fn chainsinit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= params.chains) {
        return;
    }
    rs = mix_in(mix_in(mix_in(params.frame_seed, params.step), tid * 0x9e3779b9u), 0x51633e2du);
    let base = tid * 16u;
    var c: array<f32, 16>;
    random_shape(&c);
    chain[base + 0] = bitcast<f32>(params.cur_score);
    chain[base + 1] = c[1];
    chain[base + 2] = c[2];
    chain[base + 15] = 0.0;
    for (var i = 6u; i < 14u; i = i + 1u) {
        chain[base + i] = c[i];
    }
    chain[base + 3] = 0.0;
    chain[base + 4] = 0.0;
    chain[base + 5] = 0.0;
}

// ---------- winnersel ----------
// One thread per 64 chains: scan for the global best score among live
// chains; write winner row into shapes[num_shapes] and broadcast it back
// to all chains as the new incumbent for the next step.
@compute @workgroup_size(64)
fn winnersel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let wid = gid.x; // handles chains [wid*64, wid*64+64)
    let lo = wid * 64u;
    var best_score = 1e30;
    var best_idx = 0xffffffffu;
    for (var i = lo; i < min(lo + 64u, params.chains); i = i + 1u) {
        let base = i * 16u;
        if (chain[base + 1] == 0.0) {
            continue;
        }
        let sc = chain[base + 0];
        if (sc < best_score) {
            best_score = sc;
            best_idx = i;
        }
    }
    // stage per-workgroup winners into part[] slot area 0..: use part as
    // scratch: slot = params.chains*16 + wid*4
    let wbase = params.chains * 16u + wid * 4u;
    atomicStore(&part[wbase + 0], bitcast<u32>(best_score));
    atomicStore(&part[wbase + 1], bitcast<u32>(f32(best_idx)));
}

@compute @workgroup_size(64)
fn winnersel2(@builtin(global_invocation_id) gid: vec3<u32>) {
    // reduce workgroup winners (wg_count entries) → single winner; write
    // to shapes[num_shapes] and install as chain 0's incumbent; other
    // chains re-init from the winner next propose round? No: they keep
    // hill climbing independently; the step's committed shape is the argmin.
    let wid = gid.x;
    let nwg = (params.chains + 63u) / 64u;
    var best_score = 1e30;
    var best_idx = 0xffffffffu;
    for (var k = 0u; k < nwg; k = k + 1u) {
        let wbase = params.chains * 16u + k * 4u;
        let sc = f32(atomicLoad(&part[wbase + 0]));
        if (sc < best_score) {
            best_score = sc;
            best_idx = u32(atomicLoad(&part[wbase + 1]));
        }
    }
    if (best_idx == 0xffffffffu) {
        return;
    }
    let srow = params.num_shapes * 14u;
    let cbase = u32(best_idx) * 16u;
    // write shape row (14 floats): score,id,alpha,r,g,b,p0..7
    shapes[srow + 0] = best_score;
    shapes[srow + 1] = chain[cbase + 1];
    shapes[srow + 2] = chain[cbase + 2];
    shapes[srow + 3] = chain[cbase + 3];
    shapes[srow + 4] = chain[cbase + 4];
    shapes[srow + 5] = chain[cbase + 5];
    for (var i = 6u; i < 14u; i = i + 1u) {
        shapes[srow + i] = chain[cbase + i];
    }
}
