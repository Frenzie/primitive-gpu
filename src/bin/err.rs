#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    for adapter in adapters {
        let info = adapter.get_info();
        let r = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("e"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
        });
        let (device, _) = match r {
            Ok(v) => v,
            Err(e) => {
                println!("{}: request_device failed: {e}", info.name);
                continue;
            }
        };
        let name2 = info.name.clone();
        device.on_uncaptured_error(Box::new(move |e| {
            eprintln!("[{}] ERR: {}", name2, e);
        }));
        let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("opt"),
            source: wgpu::ShaderSource::Wgsl(shapes::OPTIMIZE_WGSL.to_string().into()),
        });
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("opt"),
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let _ = p.get_bind_group_layout(0);
        }));
        let _ = device.poll(wgpu::Maintain::Wait);
        println!("--- {} done", info.name);
    }
}
