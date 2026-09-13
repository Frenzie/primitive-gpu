#[path = "../gpuio.rs"]
pub mod gpuio;
#[path = "../shapes.rs"]
pub mod shapes;
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let mut picked = None;
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let nm = a.get_info().name.clone();
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
            let errored = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let f = errored.clone();
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
            let ok = !errored.load(std::sync::atomic::Ordering::SeqCst);
            drop(m);
            println!("adapter {nm}: {}", if ok { "ok" } else { "lost" });
            if ok {
                picked = Some((d, q));
                break;
            }
        } else {
            println!("adapter {nm}: request failed");
        }
    }
    let (device, queue) = picked.expect("no adapter");
    let winners = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("w"),
        size: 64 * shapes::ROW_BYTES as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let tgt = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("t"),
        size: 256 * 256 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let cur = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("c"),
        size: 256 * 256 * 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    // uniform: width 256, height 256, type 1, alpha 128, rounds 8, n_random 64,
    // seed 42, step 0, num_shapes 0, bg 0, out 1024x1024 ss2, cur_score bits(1.0), age 2
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct U { width: u32, height: u32, shape_type: i32, alpha: i32, rounds: u32, n_random: u32, frame_seed: u32, step: u32, num_shapes: u32, bg: u32, out_w: u32, out_h: u32, ss: u32, cur_score: u32, age: u32, pad2: u32 }
    let u = U { width: 256, height: 256, shape_type: 1, alpha: 128, rounds: 256, n_random: 1000, frame_seed: 42, step: 0, num_shapes: 0, bg: 0, out_w: 1024, out_h: 1024, ss: 2, cur_score: (1.0f32).to_bits(), age: 16, pad2: 0 };
    let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p"),
        size: 64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&pbuf, 0, bytemuck::bytes_of(&u));
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("opt"),
        source: wgpu::ShaderSource::Wgsl(shapes::OPTIMIZE_WGSL.to_string().into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("opt"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bgl = pipeline.get_bind_group_layout(0);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: pbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: tgt.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cur.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: winners.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(2, 1, 1);
    }
    queue.submit(Some(enc.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);
    let rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rb"),
        size: 64 * shapes::ROW_BYTES as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc2 = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc2.copy_buffer_to_buffer(&winners, 0, &rb, 0, 64 * shapes::ROW_BYTES as u64);
    queue.submit(Some(enc2.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);
    let data = gpuio::read_bytes(&device, &rb);
    let rows: &[f32] = bytemuck::cast_slice(&data);
    for wg in 0..2 {
        println!("wg{wg}: {:?}", &rows[wg * 14..wg * 14 + 14]);
    }
}
