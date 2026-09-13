// Shared prelude, concatenated before every kernel by shapes.rs.
// Shape row layout (14 f32): [score, id, alpha, r, g, b, p0..p7]
//   id 1 triangle    p0..5  x1,y1,x2,y2,x3,y3
//   id 2 rect        p0..3  x1,y1,x2,y2   (axis-aligned, inclusive)
//   id 3 ellipse     p0..3  cx,cy,rx,ry
//   id 4 circle      p0..3  cx,cy,r,r
//   id 5 rot rect    p0..4  cx,cy,sx,sy,angle(deg)
//   id 6 quadratic   p0..6  x1,y1,cx,cy,x2,y2,width
//   id 7 rot ellipse p0..4  cx,cy,rx,ry,angle(deg)
//   id 8 polygon     p0..7  quad vertices x,y * 4
// Pixels are sampled at integer coordinates to mirror the Go rasterizers.

var<private> rs: u32;

fn pcg() -> u32 {
    rs = rs * 747796405u + 2891336453u;
    let w = ((rs >> ((rs >> 28u) + 4u)) ^ rs) * 277803737u;
    return (w >> 22u) ^ w;
}

fn ru() -> u32 {
    return pcg();
}

fn mix_in(h: u32, v: u32) -> u32 {
    var x = h ^ v;
    x = x * 0x27220a95u;
    x = x ^ (x >> 15u);
    x = x * 0x85ebca6bu;
    x = x ^ (x >> 13u);
    return x;
}

fn rf() -> f32 {
    return f32(ru() >> 8u) * (1.0 / 16777216.0);
}

fn rns(s: f32) -> f32 {
    let u1 = max(rf(), 1e-7);
    let u2 = rf();
    return s * sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

fn clampi(x: i32, lo: i32, hi: i32) -> i32 {
    return clamp(x, lo, hi);
}

fn tri_pt(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32, px: f32, py: f32) -> bool {
    let e1 = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    let e2 = (cx - bx) * (py - by) - (cy - by) * (px - bx);
    let e3 = (ax - cx) * (py - cy) - (ay - cy) * (px - cx);
    return (e1 >= 0.0 && e2 >= 0.0 && e3 >= 0.0) || (e1 <= 0.0 && e2 <= 0.0 && e3 <= 0.0);
}

fn seg_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let ex = bx - ax;
    let ey = by - ay;
    let len2 = ex * ex + ey * ey;
    var t = 0.0;
    if (len2 > 0.0) {
        t = clamp(((px - ax) * ex + (py - ay) * ey) / len2, 0.0, 1.0);
    }
    let dx = ax + ex * t - px;
    let dy = ay + ey * t - py;
    return sqrt(dx * dx + dy * dy);
}

fn tri_valid(p: array<f32, 8>) -> bool {
    let x1 = p[2] - p[0];
    let y1 = p[3] - p[1];
    let x2 = p[4] - p[0];
    let y2 = p[5] - p[1];
    let d1 = sqrt(x1 * x1 + y1 * y1);
    let d2 = sqrt(x2 * x2 + y2 * y2);
    if (d1 <= 0.0 || d2 <= 0.0) {
        return false;
    }
    let a1 = acos(clamp((x1 * x2 + y1 * y2) / (d1 * d2), -1.0, 1.0)) * 57.29577951308232;
    let x3 = p[0] - p[2];
    let y3 = p[1] - p[3];
    let x4 = p[4] - p[2];
    let y4 = p[5] - p[3];
    let d3 = sqrt(x3 * x3 + y3 * y3);
    let d4 = sqrt(x4 * x4 + y4 * y4);
    if (d3 <= 0.0 || d4 <= 0.0) {
        return false;
    }
    let a2 = acos(clamp((x3 * x4 + y3 * y4) / (d3 * d4), -1.0, 1.0)) * 57.29577951308232;
    let a3 = 180.0 - a1 - a2;
    return a1 > 15.0 && a2 > 15.0 && a3 > 15.0;
}

fn quad_valid(p: array<f32, 8>) -> bool {
    let d12 = (p[0] - p[2]) * (p[0] - p[2]) + (p[1] - p[3]) * (p[1] - p[3]);
    let d23 = (p[2] - p[4]) * (p[2] - p[4]) + (p[3] - p[5]) * (p[3] - p[5]);
    let d13 = (p[0] - p[4]) * (p[0] - p[4]) + (p[1] - p[5]) * (p[1] - p[5]);
    return d13 > d12 && d13 > d23;
}

fn inside_poly(p: array<f32, 8>, x: f32, y: f32) -> bool {
    var inside = false;
    var j = 3u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let xi = p[i * 2u];
        let yi = p[i * 2u + 1u];
        let xj = p[j * 2u];
        let yj = p[j * 2u + 1u];
        if ((yi > y) != (yj > y)) {
            let xint = (xj - xi) * (y - yi) / (yj - yi) + xi;
            if (x < xint) {
                inside = !inside;
            }
        }
        j = i;
    }
    return inside;
}

fn inside_of(id: u32, p: array<f32, 8>, px: f32, py: f32) -> bool {
    if (id == 1u) {
        return tri_pt(p[0], p[1], p[2], p[3], p[4], p[5], px, py);
    } else if (id == 2u) {
        return px >= min(p[0], p[2]) && px <= max(p[0], p[2])
            && py >= min(p[1], p[3]) && py <= max(p[1], p[3]);
    } else if (id == 3u || id == 4u) {
        let dx = (px - p[0]) / max(p[2], 0.001);
        let dy = (py - p[1]) / max(p[3], 0.001);
        return dx * dx + dy * dy <= 1.0;
    } else if (id == 5u) {
        let a = radians(p[4]);
        let ca = cos(a);
        let sa = sin(a);
        let dx = px - p[0];
        let dy = py - p[1];
        let tx = dx * ca + dy * sa;
        let ty = -dx * sa + dy * ca;
        return abs(tx) <= p[2] * 0.5 && abs(ty) <= p[3] * 0.5;
    } else if (id == 6u) {
        let hw = max(p[6], 0.25) * 0.5;
        var md = 1e30;
        var qx = p[0];
        var qy = p[1];
        for (var i = 1u; i <= 12u; i = i + 1u) {
            let t = f32(i) / 12.0;
            let mt = 1.0 - t;
            let bx = mt * mt * p[0] + 2.0 * mt * t * p[2] + t * t * p[4];
            let by = mt * mt * p[1] + 2.0 * mt * t * p[3] + t * t * p[5];
            md = min(md, seg_dist(px, py, qx, qy, bx, by));
            qx = bx;
            qy = by;
        }
        return md <= hw;
    } else if (id == 7u) {
        let a = radians(p[4]);
        let ca = cos(a);
        let sa = sin(a);
        let dx = px - p[0];
        let dy = py - p[1];
        let tx = dx * ca + dy * sa;
        let ty = -dx * sa + dy * ca;
        let ux = tx / max(p[2], 0.001);
        let uy = ty / max(p[3], 0.001);
        return ux * ux + uy * uy <= 1.0;
    } else {
        return inside_poly(p, px, py);
    }
}

fn bbox_of(id: u32, p: array<f32, 8>) -> vec4<f32> {
    if (id == 1u) {
        return vec4<f32>(
            min(min(p[0], p[2]), p[4]) - 1.0,
            min(min(p[1], p[3]), p[5]) - 1.0,
            max(max(p[0], p[2]), p[4]) + 1.0,
            max(max(p[1], p[3]), p[5]) + 1.0);
    } else if (id == 2u) {
        return vec4<f32>(
            min(p[0], p[2]) - 1.0,
            min(p[1], p[3]) - 1.0,
            max(p[0], p[2]) + 1.0,
            max(p[1], p[3]) + 1.0);
    } else if (id == 3u || id == 4u) {
        return vec4<f32>(p[0] - p[2] - 1.0, p[1] - p[3] - 1.0, p[0] + p[2] + 1.0, p[1] + p[3] + 1.0);
    } else if (id == 5u) {
        let a = radians(p[4]);
        let ca = cos(a);
        let sa = sin(a);
        let hx = p[2] * 0.5;
        let hy = p[3] * 0.5;
        var mnx = 1e30;
        var mny = 1e30;
        var mxx = -1e30;
        var mxy = -1e30;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let sx = select(-1.0, 1.0, i == 1u || i == 2u);
            let sy = select(-1.0, 1.0, i == 0u || i == 1u);
            let x = hx * sx * ca - hy * sy * sa + p[0];
            let y = hx * sx * sa + hy * sy * ca + p[1];
            mnx = min(mnx, x);
            mny = min(mny, y);
            mxx = max(mxx, x);
            mxy = max(mxy, y);
        }
        return vec4<f32>(mnx - 1.0, mny - 1.0, mxx + 1.0, mxy + 1.0);
    } else if (id == 6u) {
        let m = max(p[6], 0.25) * 0.5 + 1.0;
        return vec4<f32>(
            min(min(p[0], p[2]), p[4]) - m,
            min(min(p[1], p[3]), p[5]) - m,
            max(max(p[0], p[2]), p[4]) + m,
            max(max(p[1], p[3]), p[5]) + m);
    } else if (id == 7u) {
        let a = radians(p[4]);
        let ca = abs(cos(a));
        let sa = abs(sin(a));
        let ex = sqrt(p[2] * p[2] * ca * ca + p[3] * p[3] * sa * sa) + 1.0;
        let ey = sqrt(p[2] * p[2] * sa * sa + p[3] * p[3] * ca * ca) + 1.0;
        return vec4<f32>(p[0] - ex, p[1] - ey, p[0] + ex, p[1] + ey);
    } else {
        var mnx = 1e30;
        var mny = 1e30;
        var mxx = -1e30;
        var mxy = -1e30;
        for (var i = 0u; i < 4u; i = i + 1u) {
            mnx = min(mnx, p[i * 2u]);
            mny = min(mny, p[i * 2u + 1u]);
            mxx = max(mxx, p[i * 2u]);
            mxy = max(mxy, p[i * 2u + 1u]);
        }
        return vec4<f32>(mnx - 1.0, mny - 1.0, mxx + 1.0, mxy + 1.0);
    }
}