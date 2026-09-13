#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let src = shapes::OPTIMIZE_WGSL;
    match naga::front::wgsl::parse_str(src) {
        Ok(_) => println!("parse OK"),
        Err(e) => {
            println!("error: {e}");
            let mut src: &dyn std::error::Error = &e;
            while let Some(s) = src.source() {
                println!("caused by: {s}");
                src = s;
            }
        }
    }
}
