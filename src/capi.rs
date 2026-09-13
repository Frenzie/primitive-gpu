//! Minimal C ABI over the engine, for libavfilter integration and other
//! non-Rust hosts. Sessions are opaque; a session owns one GPU engine bound
//! to one frame geometry. For video, create the session once and feed
//! frames; with reuse enabled, each frame warm-starts from the previous
//! frame's shape list.

use crate::engine::Engine;
use crate::io;

/// Opaque session handle.
pub struct Session {
    eng: Option<Engine>,
    reuse: bool,
    frame_idx: u32,
    // pending RGBA output of the last process call
    last: Vec<u8>,
    out_w: u32,
    out_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PgpuConfig {
    /// internal optimization resolution (long edge); typical: 128–256
    pub internal_size: u32,
    /// output long-edge size; output is scaled to this
    pub output_size: u32,
    /// output supersampling factor
    pub ss: u32,
    /// shape type 0..8 (see README)
    pub mode: i32,
    /// color alpha 1..255, 0 = optimizer chooses
    pub alpha: i32,
    /// shapes to add per frame (without reuse)
    pub shapes_per_frame: u32,
    /// RNG seed
    pub seed: u32,
    /// 1 = warm-start each frame from the previous frame's shapes
    pub reuse: i32,
    /// with reuse: cap on retained shapes
    pub max_shapes: u32,
}

impl Default for PgpuConfig {
    fn default() -> Self {
        PgpuConfig {
            internal_size: 128,
            output_size: 640,
            ss: 1,
            mode: 1,
            alpha: 128,
            shapes_per_frame: 8,
            seed: 42,
            reuse: 1,
            max_shapes: 200,
        }
    }
}

/// Create a session. Returns null on failure (no adapter, bad params).
///
/// # Safety
/// `config` must point to a valid PgpuConfig or be null (defaults used).
#[no_mangle]
pub unsafe extern "C" fn pgpu_session_new(config: *const PgpuConfig) -> *mut Session {
    let cfg = if config.is_null() {
        PgpuConfig::default()
    } else {
        *config
    };
    let sess = Box::new(Session {
        eng: None,
        reuse: cfg.reuse != 0,
        frame_idx: 0,
        last: Vec::new(),
        out_w: cfg.output_size,
        out_h: cfg.output_size,
    });
    Box::into_raw(sess)
}

/// Destroy a session.
///
/// # Safety
/// `sess` must be a pointer returned by pgpu_session_new, not yet freed.
#[no_mangle]
pub unsafe extern "C" fn pgpu_session_free(sess: *mut Session) {
    if !sess.is_null() {
        drop(Box::from_raw(sess));
    }
}

/// Process one RGBA8 frame (w×h). Returns a pointer to the RGBA8 output
/// (out_w × out_h, valid until the next pgpu_process call) or null on
/// error. Output size follows the input aspect at output_size long edge.
///
/// # Safety
/// `sess` valid; `rgba` points to w*h*4 readable bytes.
#[no_mangle]
pub unsafe extern "C" fn pgpu_process(
    sess: *mut Session,
    rgba: *const u8,
    w: u32,
    h: u32,
) -> *const u8 {
    let sess = if sess.is_null() {
        return std::ptr::null();
    } else {
        &mut *sess
    };
    if rgba.is_null() || w == 0 || h == 0 {
        return std::ptr::null();
    }
    let src = std::slice::from_raw_parts(rgba, (w * h * 4) as usize);
    if sess.eng.is_none() {
        return std::ptr::null();
    }
    let eng = sess.eng.as_mut().unwrap();

    // downscale input to internal resolution
    let (iw, ih) = if w >= h {
        (eng.w, eng.w * h / w)
    } else {
        (eng.w * w / h, eng.h)
    };
    let target = downscale_rgba(src, w, h, iw, ih);

    let started = sess.reuse && sess.frame_idx > 0;
    let r = if started {
        eng.begin_frame_reuse(target)
    } else {
        eng.reset_for_frame(target);
        Ok(0)
    };
    if r.is_err() {
        return std::ptr::null();
    }
    // run the optimization steps for this frame
    for i in 0..eng.steps_per_frame() {
        let s = eng.seed();
        if eng.step(eng.shape_type(), eng.alpha(), s ^ (sess.frame_idx << 8) ^ i)
            .is_err()
        {
            break;
        }
    }
    eng.trim_to(eng.shape_limit());
    sess.frame_idx += 1;
    match eng.render() {
        Ok(img) => {
            sess.last = img;
            sess.last.as_ptr()
        }
        Err(_) => {
            // retry once; gen9 occasionally kills a dispatch
            match eng.render() {
                Ok(img) => {
                    sess.last = img;
                    sess.last.as_ptr()
                }
                Err(_) => std::ptr::null(),
            }
        }
    }
}

/// Initialize the session's engine for a given frame size. Must be called
/// before pgpu_process. Returns 0 on success.
///
/// # Safety
/// `sess` valid.
#[no_mangle]
pub unsafe extern "C" fn pgpu_session_init(
    sess: *mut Session,
    w: u32,
    h: u32,
    config: *const PgpuConfig,
) -> i32 {
    let sess = if sess.is_null() {
        return -1;
    } else {
        &mut *sess
    };
    let cfg = if config.is_null() {
        PgpuConfig::default()
    } else {
        *config
    };
    sess.reuse = cfg.reuse != 0;
    sess.frame_idx = 0;
    sess.out_w = cfg.output_size;
    sess.out_h = cfg.output_size;

    // average color of first frame becomes bg
    // No frame yet: use neutral gray bg (re-seeded on the first process).
    let (iw, ih) = if w >= h {
        (cfg.internal_size, cfg.internal_size * h / w)
    } else {
        (cfg.internal_size * w / h, cfg.internal_size)
    };
    let (out_w, out_h) = if w >= h {
        (cfg.output_size, cfg.output_size * h / w)
    } else {
        (cfg.output_size * w / h, cfg.output_size)
    };
    sess.out_w = out_w;
    sess.out_h = out_h;
    let fake_target = vec![128u8; (iw * ih * 4) as usize];
    match Engine::new(
        fake_target,
        iw,
        ih,
        out_w,
        out_h,
        cfg.ss.max(1),
        [128, 128, 128],
    ) {
        Ok(mut eng) => {
            eng.set_shape_type(cfg.mode);
            eng.set_alpha(cfg.alpha);
            eng.set_frame_seed(cfg.seed);
            eng.set_steps_per_frame(cfg.shapes_per_frame);
            eng.set_max_shapes(cfg.max_shapes);
            sess.eng = Some(eng);
            0
        }
        Err(_) => -2,
    }
}

/// Number of shapes currently in the model.
///
/// # Safety
/// `sess` valid.
#[no_mangle]
pub unsafe extern "C" fn pgpu_shape_count(sess: *const Session) -> u32 {
    if sess.is_null() {
        return 0;
    }
    (*sess).eng.as_ref().map(|e| e.num_shapes).unwrap_or(0)
}

/// Output dimensions of the last/current render.
///
/// # Safety
/// `sess` valid; out pointers writable.
#[no_mangle]
pub unsafe extern "C" fn pgpu_output_size(sess: *const Session, w: *mut u32, h: *mut u32) {
    if sess.is_null() || w.is_null() || h.is_null() {
        return;
    }
    *w = (*sess).out_w;
    *h = (*sess).out_h;
}

fn downscale_rgba(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh * 4) as usize];
    for y in 0..dh {
        let sy = (y as u64 * sh as u64 / dh as u64) as u32;
        for x in 0..dw {
            let sx = (x as u64 * sw as u64 / dw as u64) as u32;
            let si = ((sy * sw + sx) * 4) as usize;
            let di = ((y * dw + x) * 4) as usize;
            dst[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    dst
}