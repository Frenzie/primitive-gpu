// Shape row layout (14 f32): [score, id, alpha, r, g, b, p0..p7]
//   id 1 triangle    p0..5  x1,y1,x2,y2,x3,y3
//   id 2 rect        p0..3  x1,y1,x2,y2   (axis-aligned, inclusive)
//   id 3 ellipse     p0..3  cx,cy,rx,ry
//   id 4 circle      p0..3  cx,cy,r,r
//   id 5 rot rect    p0..4  cx,cy,sx,sy,angle(deg)
//   id 6 quadratic   p0..6  x1,y1,cx,cy,x2,y2,width
//   id 7 rot ellipse p0..4  cx,cy,rx,ry,angle(deg)
//   id 8 polygon     p0..7  quad vertices x,y * 4

pub const ROW: usize = 14;
pub const ROW_BYTES: usize = ROW * 4;
pub const MAX_SHAPES: usize = 4096;

pub const MODE_COMBO: i32 = 0;
pub const MODE_TRIANGLE: i32 = 1;
pub const MODE_RECT: i32 = 2;
pub const MODE_ELLIPSE: i32 = 3;
pub const MODE_CIRCLE: i32 = 4;
pub const MODE_ROT_RECT: i32 = 5;
pub const MODE_QUAD: i32 = 6;
pub const MODE_ROT_ELLIPSE: i32 = 7;
pub const MODE_POLYGON: i32 = 8;

pub const COMMON_WGSL: &str = include_str!("../shaders/common.wgsl");
pub const OPTIMIZE_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    include_str!("../shaders/optimize.wgsl")
);
pub const COMMIT_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    include_str!("../shaders/commit.wgsl")
);
pub const RENDER_WGSL: &str = concat!(
    include_str!("../shaders/common.wgsl"),
    include_str!("../shaders/render.wgsl")
);
pub const RESCORE_WGSL: &str = include_str!("../shaders/rescore.wgsl");