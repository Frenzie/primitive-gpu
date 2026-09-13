use anyhow::{Context, Result};
use image::{imageops::FilterType, GenericImageView, ImageBuffer, Rgba};
use std::path::Path;

pub struct Frame {
    pub target: Vec<u8>, // internal res, RGBA8, premultiplied? No — plain NRGBA bytes
    pub w: u32,
    pub h: u32,
}

pub fn load_target(path: &str, internal: u32) -> Result<Frame> {
    let img = if path == "-" {
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut std::io::stdin(), &mut buf).context("reading stdin")?;
        image::load_from_memory(&buf)?
    } else {
        image::open(Path::new(path))?
    };
    let small = if internal > 0 {
        img.resize_exact(internal, internal, FilterType::Nearest)
    } else {
        img
    };
    let (w, h) = small.dimensions();
    let target = to_nrgba_bytes(&small);
    Ok(Frame { target, w, h })
}

fn to_nrgba_bytes<I: GenericImageView<Pixel = Rgba<u8>>>(img: &I) -> Vec<u8> {
    let (w, h) = img.dimensions();
    let mut out = vec![0u8; (w * h * 4) as usize];
    for (i, p) in img.pixels().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&p.0);
    }
    out
}

pub fn save_png(path: &str, w: u32, h: u32, data: &[u8]) -> Result<()> {
    let buf: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(w, h, data.to_vec()).context("bad buffer")?;
    buf.save(path)?;
    Ok(())
}

pub fn average_color(target: &[u8], w: u32, h: u32) -> [u8; 3] {
    let n = (w * h) as usize;
    let mut sum = [0u64; 3];
    for px in target.chunks_exact(4) {
        for c in 0..3 {
            sum[c] += px[c] as u64;
        }
    }
    [(sum[0] / n as u64) as u8, (sum[1] / n as u64) as u8, (sum[2] / n as u64) as u8]
}