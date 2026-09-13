#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    match naga::front::wgsl::parse_str(shapes::RENDER_WGSL) {
        Ok(_) => println!("render OK"),
        Err(e) => println!("render: {e}"),
    }
}
