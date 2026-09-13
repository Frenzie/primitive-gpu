use anyhow::{anyhow, Result};
use crate::shapes::OPTIMIZE_WGSL;

pub fn pick_adapter(instance: &wgpu::Instance) -> Result<(wgpu::Device, wgpu::Queue)> {
    // Some drivers (Intel gen9 ANV on Mesa) lose the device compiling large
    // shaders; probe each adapter against the real optimize shader and use
    // the first healthy one.
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        let res: Option<(wgpu::Device, wgpu::Queue)> = pollster::block_on(async {
            let dev = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("primitive-gpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await;
            let (d, q) = match dev {
                Ok(v) => v,
                Err(_) => return None,
            };
            let errored = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = errored.clone();
            d.on_uncaptured_error(Box::new(move |_| {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            }));
            let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe"),
                source: wgpu::ShaderSource::Wgsl(OPTIMIZE_WGSL.to_string().into()),
            });
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("probe"),
                    layout: None,
                    module: &m,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            }))
            .is_err();
            let _ = d.poll(wgpu::Maintain::Wait);
            let ok = !panicked && !errored.load(std::sync::atomic::Ordering::SeqCst);
            drop(m);
            if ok { Some((d, q)) } else { None }
        });
        match res {
            Some((d, q)) => {
                eprintln!("using adapter: {} ({:?})", info.name, info.device_type);
                d.on_uncaptured_error(Box::new(|e| {
                    eprintln!("fatal wgpu error: {e}");
                    std::process::exit(2);
                }));
                return Ok((d, q));
            }
            None => eprintln!("skipping adapter: {} (shader compile failed)", info.name),
        }
    }
    Err(anyhow!("no working Vulkan adapter"))
}

