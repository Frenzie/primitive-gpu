#[path = "../engine.rs"]
pub mod engine;
#[path = "../gpuio.rs"]
pub mod gpuio;
#[path = "../shapes.rs"]
pub mod shapes;
#[path = "../io.rs"]
pub mod io;

fn main() {
    use engine::Engine;
    let frame = io::load_target("/home/frans/src/primitive/examples/monalisa.png", 256).unwrap();
    let bg = io::average_color(&frame.target, frame.w, frame.h);
    let mut eng = Engine::new(frame.target.clone(), frame.w, frame.h, 1024, 1024, 1, bg).unwrap();
    for i in 0..25 {
        if eng.step(1, 128, 42).is_err() {
            println!("failed at step {i}");
            return;
        }
    }
    let cb = eng.read_canvas_probe();
    let mut min = [255u8; 3];
    let mut max = [0u8; 3];
    for px in cb.chunks_exact(4) {
        for c in 0..3 {
            if px[c] < min[c] {
                min[c] = px[c];
            }
            if px[c] > max[c] {
                max[c] = px[c];
            }
        }
    }
    println!("canvas after 25 steps: min {:?} max {:?}", min, max);
    println!("score: {:.4}", eng.score);
}
