#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    // Validate via naga directly for a precise error message.
    let src = shapes::OPTIMIZE_WGSL;
    match naga::front::wgsl::parse_str(src) {
        Ok(m) => {
            let info = match naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&m)
            {
                Ok(v) => {
                    println!("naga validate OK");
                    v
                }
                Err(e) => {
                    eprintln!("naga validate error:\n{e}");
                    return;
                }
            };
            let _ = info;
        }
        Err(e) => {
            eprintln!("naga parse error:\n{e}");
        }
    }
}
