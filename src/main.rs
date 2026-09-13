mod engine;
mod gpuio;
mod io;
mod shapes;

use anyhow::Result;
use engine::Engine;
use shapes::{MODE_TRIANGLE, ROW};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut input = String::new();
    let mut output = String::new();
    let mut num = 100u32;
    let mut mode = MODE_TRIANGLE;
    let mut alpha = 128i32;
    let mut input_size = 256u32;
    let mut output_size = 1024u32;
    let mut ss = 2u32;
    let mut seed = 42u32;
    let mut video_mode = false;
    let mut vw = 1920u32;
    let mut vh = 1080u32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-i" => {
                i += 1;
                input = args[i].clone();
            }
            "-o" => {
                i += 1;
                output = args[i].clone();
            }
            "-n" => {
                i += 1;
                num = args[i].parse()?;
            }
            "-m" => {
                i += 1;
                mode = args[i].parse()?;
            }
            "-a" => {
                i += 1;
                alpha = args[i].parse()?;
            }
            "-r" => {
                i += 1;
                input_size = args[i].parse()?;
            }
            "-s" => {
                i += 1;
                output_size = args[i].parse()?;
            }
            "--ss" => {
                i += 1;
                ss = args[i].parse()?;
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse()?;
            }
            "--video" => {
                video_mode = true;
            }
            "--vw" => {
                i += 1;
                vw = args[i].parse()?;
            }
            "--vh" => {
                i += 1;
                vh = args[i].parse()?;
            }
            _ => {}
        }
        i += 1;
    }

    if video_mode {
        run_video(vw, vh, num, mode, alpha, input_size, ss, seed)?;
        return Ok(());
    }

    if input.is_empty() || output.is_empty() {
        anyhow::bail!("usage: primitive-gpu -i input -o output [-n count] [-m mode] [-a alpha] [-r size] [-s size] [--ss n] [--seed n] [--video --vw w --vh h]");
    }

    let frame = io::load_target(&input, input_size)?;
    let bg = io::average_color(&frame.target, frame.w, frame.h);
    let out_w = if frame.w >= frame.h { output_size } else { output_size * frame.w / frame.h };
    let out_h = if frame.w >= frame.h { output_size * frame.h / frame.w } else { output_size };

    let mut eng = Engine::new(frame.target.clone(), frame.w, frame.h, out_w, out_h, ss, bg)?;
    eng.set_frame_seed(seed);
    for _ in 0..num {
        let mut ok = None;
        for attempt in 0..3u32 {
            match eng.step(mode, alpha, seed ^ (attempt << 16)) {
                Ok(row) => {
                    ok = Some(row);
                    break;
                }
                Err(_) => continue,
            }
        }
        ok.ok_or_else(|| anyhow::anyhow!("optimizer failed after retries"))?;
        eprint!(".");
    }
    eprintln!();
    let mut img = eng.render()?;
    for _ in 0..4 {
        if let Ok(d) = eng.render() {
            img = d;
            break;
        }
    }
    io::save_png(&output, out_w, out_h, &img)?;
    eprintln!("wrote {} ({} shapes, score {:.4})", output, num, eng.score);
    Ok(())
}

fn run_video(
    vw: u32,
    vh: u32,
    num: u32,
    mode: i32,
    alpha: i32,
    input_size: u32,
    ss: u32,
    seed: u32,
) -> Result<()> {
    use std::io::{Read, Write};
    let bpp = 3usize; // rgb24
    let frame_bytes = vw as usize * vh as usize * bpp;
    let mut stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut frame_idx = 0u32;
    let mut eng: Option<Engine> = None;
    let mut buf = vec![0u8; frame_bytes];
    loop {
        if stdin.read_exact(&mut buf).is_err() {
            break;
        }
        // rgb24 -> internal target
        let (iw, ih) = if vw >= vh { (input_size, input_size * vh / vw) } else { (input_size * vw / vh, input_size) };
        let target = resize_rgb24(&buf, vw, vh, iw, ih);
        if eng.is_none() {
            let bg = io::average_color(&target, iw, ih);
            eng = Some(Engine::new(target.clone(), iw, ih, vw, vh, ss, bg)?);
        }
        let e = eng.as_mut().unwrap();
        e.reset_for_frame(target);
        for s in 0..num {
            e.set_shape_type(mode);
            e.set_alpha(alpha);
            e.set_frame_seed(seed ^ (frame_idx << 8) ^ s);
            e.step(mode, alpha, seed ^ (frame_idx << 8) ^ s)?;
        }
        let img = e.render()?;
        // RGBA -> RGB
        let mut rgb = vec![0u8; frame_bytes];
        for (o, chunk) in img.chunks_exact(4).enumerate() {
            rgb[o * 3..o * 3 + 3].copy_from_slice(&chunk[0..3]);
        }
        stdout.write_all(&rgb)?;
        stdout.flush()?;
        frame_idx += 1;
    }
    Ok(())
}

fn resize_rgb24(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh * 3) as usize];
    for y in 0..dh {
        let sy = (y as u64 * sh as u64 / dh as u64) as u32;
        for x in 0..dw {
            let sx = (x as u64 * sw as u64 / dw as u64) as u32;
            let si = ((sy * sw + sx) * 3) as usize;
            let di = ((y * dw + x) * 3) as usize;
            dst[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    dst
}

// keep ROW import used
#[allow(dead_code)]
const _: [u8; ROW] = [0u8; ROW];