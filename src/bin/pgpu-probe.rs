// CI helper: exit 0 iff a healthy Vulkan adapter (compiles the optimize
// shader) exists. Used by CI to skip GPU-dependent smoke tests on runners
// whose software driver can't compile them.
#[path = "../engine_probe.rs"]
mod engine_probe;
#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    match engine_probe::pick_adapter(&instance) {
        Ok(_) => {
            println!("healthy adapter found");
            std::process::exit(0);
        }
        Err(e) => {
            println!("no healthy adapter: {e}");
            std::process::exit(1);
        }
    }
}
