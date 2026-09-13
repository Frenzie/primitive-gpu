#[path = "../gpuio.rs"]
pub mod gpuio;
#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let mut picked = None;
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let r = pollster::block_on(async {
            a.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("d"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()
        });
        if let Some((d, q)) = r {
            let err = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let f = err.clone();
            d.on_uncaptured_error(Box::new(move |_| {
                f.store(true, std::sync::atomic::Ordering::SeqCst);
            }));
            let m = d.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("probe"),
                source: wgpu::ShaderSource::Wgsl(crate::shapes::OPTIMIZE_WGSL.to_string().into()),
            });
            let _ = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("probe"),
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let _ = d.poll(wgpu::Maintain::Wait);
            if !err.load(std::sync::atomic::Ordering::SeqCst) {
                picked = Some((d, q));
                break;
            }
        }
    }
    let (device, queue) = picked.expect("no adapter");
    // shapes buffer with ONE synthetic triangle row covering half canvas:
    // [score, id=1, alpha=255, r=255,g=0,b=0, x1,y1,x2,y2,x3,y3,0,0]
    let row: [f32; 14] = [0.5, 1.0, 255.0, 255.0, 0.0, 0.0, 10.0, 10.0, 250.0, 10.0, 10.0, 250.0, 0.0, 0.0];
    let shapes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shapes"),
        size: 4096 * 56,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&shapes_buf, 0, bytemuck::bytes_of(&row));
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("d1"),
        size: 256 * 256 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 1024 * 1024 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct U { width: u32, height: u32, shape_type: i32, alpha: i32, rounds: u32, n_random: u32, frame_seed: u32, step: u32, num_shapes: u32, bg: u32, out_w: u32, out_h: u32, ss: u32, cur_score: u32, age: u32, pad2: u32 }
    let bg = 110u32 | (99 << 8) | (62 << 16) | (0xff << 24);
    let u = U { width: 256, height: 256, shape_type: 1, alpha: 128, rounds: 0, n_random: 0, frame_seed: 0, step: 0, num_shapes: 1, bg, out_w: 1024, out_h: 1024, ss: 1, cur_score: 0, age: 0, pad2: 0 };
    queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&u));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("r"),
        source: wgpu::ShaderSource::Wgsl(shapes::RENDER_WGSL.to_string().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("r"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bgl = pipeline.get_bind_group_layout(0);
    let bg2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dummy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: dummy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: dummy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: shapes_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("out"),
                size: 1024 * 1024 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }).as_entire_binding() },
        ],
    });
    let _ = bg2;
    let _ = queue;
}
