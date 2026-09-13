// CI helper: validates all kernel sources with naga.
#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let mut failed = false;
    for (name, src) in [
        ("optimize", shapes::OPTIMIZE_WGSL),
        ("commit", shapes::COMMIT_WGSL),
        ("render", shapes::RENDER_WGSL),
        ("rescore", shapes::RESCORE_WGSL),
    ] {
        match naga::front::wgsl::parse_str(src) {
            Ok(m) => {
                if let Err(e) = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::empty(),
                )
                .validate(&m)
                {
                    println!("{name}: validation error:\n{e}");
                    failed = true;
                } else {
                    println!("{name}: OK");
                }
            }
            Err(e) => {
                println!("{name}: parse error:\n{e}");
                failed = true;
            }
        }
    }
    std::process::exit(if failed { 1 } else { 0 });
}
