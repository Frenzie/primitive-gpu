use crate::shapes::*;
use anyhow::{anyhow, Result};
use wgpu::util::DeviceExt;

pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_buf: wgpu::Buffer,
    canvas_buf: wgpu::Buffer,
    winners_buf: wgpu::Buffer,
    shapes_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    optimize_bg: wgpu::BindGroup,
    commit_bg: wgpu::BindGroup,
    optimize_pipeline: wgpu::ComputePipeline,
    commit_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::ComputePipeline,
    rescore_pipeline: wgpu::ComputePipeline,
    rescore_bg: wgpu::BindGroup,
    render_out_buf: wgpu::Buffer,
    render_bg: wgpu::BindGroup,
    readback_buf: wgpu::Buffer,
    pub w: u32,
    pub h: u32,
    pub out_w: u32,
    pub out_h: u32,
    pub ss: u32,
    pub bg: [u8; 3],
    pub num_shapes: u32,
    pub score: f64,
    shape_type: i32,
    alpha: i32,
    frame_seed: u32,
    age: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Uniform {
    width: u32,
    height: u32,
    shape_type: i32,
    alpha: i32,
    rounds: u32,
    n_random: u32,
    frame_seed: u32,
    step: u32,
    num_shapes: u32,
    bg: u32,
    out_w: u32,
    out_h: u32,
    ss: u32,
    cur_score: u32,
    age: u32,
    pad2: u32,
}

const WG_COUNT: u32 = 1024; // independent hill-climb chains (one thread each)
const WG_SIZE: u32 = 64; // threads per workgroup (parallel mutations per round)
const ROUNDS: u32 = 24;
const N_RANDOM: u32 = 16;
const AGE: u32 = 16;
fn pick_adapter(instance: &wgpu::Instance) -> Result<(wgpu::Device, wgpu::Queue)> {
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

impl Engine {
    pub fn new(
        target: Vec<u8>,
        w: u32,
        h: u32,
        out_w: u32,
        out_h: u32,
        ss: u32,
        bg: [u8; 3],
    ) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let (device, queue) = pick_adapter(&instance)?;

        let n_px = (w * h) as u64;
        let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target"),
            contents: &target,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let canvas_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas"),
            size: n_px * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let winners_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("winners"),
            size: WG_COUNT as u64 * ROW_BYTES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shapes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shapes"),
            size: ROW_BYTES as u64 * MAX_SHAPES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: WG_COUNT as u64 * ROW_BYTES as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Explicit layouts: naga prunes unused bindings from auto layouts,
        // which made binding counts vary per kernel. Declare them exactly.
        let uni = || wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let stor = |binding: u32, ro: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let optimize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("optimize-bgl"),
            entries: &[uni(), stor(1, true), stor(2, true), stor(3, false)],
        });
        let commit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("commit-bgl"),
            entries: &[uni(), stor(1, true), stor(2, false), stor(3, false)],
        });
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render-bgl"),
            entries: &[
                uni(),
                stor(1, true),
                stor(2, true),
                stor(3, true),
                stor(4, true),
                stor(5, false),
            ],
        });

        let mk = |label: &'static str, src: &str, bgl: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.to_string().into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let optimize_pipeline = mk("optimize", OPTIMIZE_WGSL, &optimize_bgl);
        let commit_pipeline = mk("commit", COMMIT_WGSL, &commit_bgl);
        let render_pipeline = mk("render", RENDER_WGSL, &render_bgl);
        let rescore_pipeline = mk("rescore", RESCORE_WGSL, &optimize_bgl);

        let pb = params_buf.as_entire_binding();
        let tb = target_buf.as_entire_binding();
        let cb = canvas_buf.as_entire_binding();
        let wb = winners_buf.as_entire_binding();
        let sb = shapes_buf.as_entire_binding();
        let mk_bg = |bgl: &wgpu::BindGroupLayout, extra: Option<wgpu::BindGroupEntry>| {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: pb.clone() },
                wgpu::BindGroupEntry { binding: 1, resource: tb.clone() },
                wgpu::BindGroupEntry { binding: 2, resource: cb.clone() },
                wgpu::BindGroupEntry { binding: 3, resource: wb.clone() },
            ];
            if let Some(e) = extra {
                entries.push(e);
            }
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: bgl,
                entries: &entries,
            })
        };
        let optimize_bg = mk_bg(&optimize_bgl, None);
        let commit_bg = mk_bg(&commit_bgl, None);
        let rescore_bg = mk_bg(&optimize_bgl, None);
        // render needs bindings 4 (shapes) and 5 (output): a persistent buffer.
        let render_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-out"),
            size: (out_w as u64) * (out_h as u64) * (ss as u64) * (ss as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pb.clone() },
                wgpu::BindGroupEntry { binding: 1, resource: tb.clone() },
                wgpu::BindGroupEntry { binding: 2, resource: cb.clone() },
                wgpu::BindGroupEntry { binding: 3, resource: wb.clone() },
                wgpu::BindGroupEntry { binding: 4, resource: sb.clone() },
                wgpu::BindGroupEntry { binding: 5, resource: render_out_buf.as_entire_binding() },
            ],
        });

        // seed canvas with bg color
        let bg_u = bg_packed(bg);
        let canvas = vec![bg_u; (w * h) as usize];
        queue.write_buffer(&canvas_buf, 0, bytemuck::cast_slice(&canvas));

        let score = initial_score(&target, w, h, bg);
        Ok(Engine {
            score,
            device,
            queue,
            target_buf,
            canvas_buf,
            winners_buf,
            shapes_buf,
            params_buf,
            optimize_bg,
            commit_bg,
            optimize_pipeline,
            commit_pipeline,
            render_pipeline,
            rescore_pipeline,
            rescore_bg,
            render_out_buf,
            render_bg,
            readback_buf,
            w,
            h,
            out_w,
            out_h,
            ss,
            bg,
            num_shapes: 0,
            shape_type: MODE_TRIANGLE,
            alpha: 128,
            frame_seed: 0,
            age: AGE,
        })
    }

    fn uniform(&self) -> Uniform {
        Uniform {
            width: self.w,
            height: self.h,
            shape_type: self.shape_type,
            alpha: self.alpha,
            rounds: ROUNDS,
            n_random: N_RANDOM,
            frame_seed: self.frame_seed,
            step: self.num_shapes,
            num_shapes: self.num_shapes,
            bg: bg_packed(self.bg),
            out_w: self.out_w,
            out_h: self.out_h,
            ss: self.ss,
            cur_score: (self.score as f32).to_bits(),
            age: self.age,
            pad2: 0,
        }
    }

    /// One optimization step: find the best shape and commit it to the canvas.
    pub fn step(&mut self, _shape_type: i32, _alpha: i32, _frame_seed: u32) -> Result<[f32; ROW]> {
        let u = self.uniform();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&u));

        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("step") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("optimize"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.optimize_pipeline);
            pass.set_bind_group(0, &self.optimize_bg, &[]);
            pass.dispatch_workgroups(WG_COUNT, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.winners_buf,
            0,
            &self.readback_buf,
            0,
            WG_COUNT as u64 * ROW_BYTES as u64,
        );
        self.queue.submit(Some(enc.finish()));

        let raw = crate::gpuio::read_bytes(&self.device, &self.readback_buf);
        let rows: &[f32] = bytemuck::cast_slice(&raw);
        let mut best = [0f32; ROW];
        best[0] = f32::INFINITY;
        for wg in 0..WG_COUNT as usize {
            let r = &rows[wg * ROW..(wg + 1) * ROW];
            // skip never-completed chains (id 0): killed/hung workgroups
            if r[1] == 0.0 {
                continue;
            }
            if r[0] < best[0] {
                best.copy_from_slice(r);
            }
        }
        if !best[0].is_finite() || best[1] == 0.0 {
            return Err(anyhow!("optimizer produced no valid shape"));
        }

        self.queue.write_buffer(
            &self.shapes_buf,
            self.num_shapes as u64 * ROW_BYTES as u64,
            bytemuck::bytes_of(&best),
        );
        self.num_shapes += 1;
        self.queue
            .write_buffer(&self.winners_buf, 0, bytemuck::bytes_of(&best));

        let u2 = self.uniform();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&u2));
        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("commit") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("commit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.commit_pipeline);
            pass.set_bind_group(0, &self.commit_bg, &[]);
            pass.dispatch_workgroups((self.w * self.h + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        if std::env::var("PG_TRACE_STEP").is_ok() {
            eprintln!("step{}: score={:.6} id={} p=({:.1},{:.1},{:.1},{:.1},{:.1},{:.1})",
                self.num_shapes, best[0], best[1], best[6], best[7], best[8], best[9], best[10], best[11]);
        }
        self.score = best[0] as f64;
        Ok(best)
    }

    /// High-quality analytic render at out_w x out_h (supersampled).
    pub fn render(&self) -> Result<Vec<u8>> {
        let w = self.out_w * self.ss;
        let h = self.out_h * self.ss;
        let n_px = (w * h) as u64;
        let rb_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-read"),
            size: n_px * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let u = self.uniform();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&u));
        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.render_bg, &[]);
            pass.dispatch_workgroups((w + 7) / 8, (h + 7) / 8, 1);
        }
        // copy from the persistent render-out buffer (binding 5)
        enc.copy_buffer_to_buffer(&self.render_out_buf, 0, &rb_buf, 0, n_px * 4);
        self.queue.submit(Some(enc.finish()));
        let data = crate::gpuio::read_bytes(&self.device, &rb_buf);
        if std::env::var("PG_DUMP_ROWS").is_ok() {
            let sz = (self.num_shapes as u64) * ROW_BYTES as u64;
            let rbb = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rows-read"),
                size: (self.num_shapes as u64) * ROW_BYTES as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut e2 = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            e2.copy_buffer_to_buffer(&self.shapes_buf, 0, &rbb, 0, (self.num_shapes as u64) * ROW_BYTES as u64);
            self.queue.submit(Some(e2.finish()));
            let rows = crate::gpuio::read_bytes(&self.device, &rbb);
            let rf: &[f32] = bytemuck::cast_slice(&rows);
            for r in rf.chunks(14).take(3) {
                eprintln!("row: {:.3} id={} a={} c=({:.0},{:.0},{:.0})", r[0], r[1], r[2], r[3], r[4], r[5]);
            }
        }

        if std::env::var("PG_DUMP_RAW").is_ok() {
            io_dump(&data, w, h);
        }
        if self.num_shapes > 0 {
            let mut nonzero = 0u64;
            for c in data.chunks_exact(4) {
                if c[0] + c[1] + c[2] > 10 {
                    nonzero += 1;
                }
            }
            if nonzero == 0 {
                return Err(anyhow!("render produced no output (GPU kill?)"));
            }
        }
        // box-downsample ss x ss to out_w x out_h
        let mut out = vec![0u8; (self.out_w * self.out_h * 4) as usize];
        for oy in 0..self.out_h {
            for ox in 0..self.out_w {
                let (mut r, mut g, mut b) = (0u32, 0u32, 0u32);
                for sy in 0..self.ss {
                    for sx in 0..self.ss {
                        let idx = (((oy * self.ss + sy) * w + ox * self.ss + sx) * 4) as usize;
                        r += data[idx] as u32;
                        g += data[idx + 1] as u32;
                        b += data[idx + 2] as u32;
                    }
                }
                let cnt = self.ss * self.ss;
                let oidx = ((oy * self.out_w + ox) * 4) as usize;
                out[oidx] = (r / cnt) as u8;
                out[oidx + 1] = (g / cnt) as u8;
                out[oidx + 2] = (b / cnt) as u8;
                out[oidx + 3] = 255;
            }
        }
        Ok(out)
    }

    /// Debug helper: read back the canvas buffer.
    pub fn read_canvas_probe(&self) -> Vec<u8> {
        let size = (self.w * self.h * 4) as u64;
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas-probe"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.canvas_buf, 0, &rb, 0, size);
        self.queue.submit(Some(enc.finish()));
        crate::gpuio::read_bytes(&self.device, &rb)
    }

    pub fn set_shape_type(&mut self, t: i32) {
        self.shape_type = t;
    }

    pub fn set_alpha(&mut self, a: i32) {
        self.alpha = a;
    }

    pub fn set_frame_seed(&mut self, s: u32) {
        self.frame_seed = s;
    }

    pub fn set_age(&mut self, a: u32) {
        self.age = a;
    }

    /// Start a fresh optimization for a new video frame.
    pub fn reset_for_frame(&mut self, target: Vec<u8>) {
        self.num_shapes = 0;
        self.score = 1.0;
        self.queue.write_buffer(&self.target_buf, 0, &target);
        let bg_u = bg_packed(self.bg);
        let canvas = vec![bg_u; (self.w * self.h) as usize];
        self.queue.write_buffer(&self.canvas_buf, 0, bytemuck::cast_slice(&canvas));
        self.score = initial_score(&target, self.w, self.h, self.bg);
    }

    /// Begin a new frame reusing the previous frame's shape list: reset the
    /// canvas to bg, rescore every retained shape against the new target,
    /// then re-commit those that still improve the score. Returns the number
    /// of retained shapes.
    pub fn begin_frame_reuse(&mut self, target: Vec<u8>) -> Result<u32> {
        let n_prev = self.num_shapes;
        self.num_shapes = 0;
        self.queue.write_buffer(&self.target_buf, 0, &target);
        let bg_u = bg_packed(self.bg);
        let canvas = vec![bg_u; (self.w * self.h) as usize];
        self.queue.write_buffer(&self.canvas_buf, 0, bytemuck::cast_slice(&canvas));
        let score0 = initial_score(&target, self.w, self.h, self.bg);
        self.score = score0;

        if n_prev == 0 {
            return Ok(0);
        }

        // copy the previous shape list into winners (rescored in place)
        self.copy_prev_shapes_to_winners(n_prev)?;

        // dispatch rescore: one thread per shape, in place on winners
        let mut u = self.uniform();
        u.num_shapes = n_prev;
        u.cur_score = (score0 as f32).to_bits();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&u));
        let mut enc =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rescore") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rescore"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.rescore_pipeline);
            pass.set_bind_group(0, &self.rescore_bg, &[]);
            pass.dispatch_workgroups((n_prev + 63) / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.winners_buf,
            0,
            &self.readback_buf,
            0,
            n_prev as u64 * ROW_BYTES as u64,
        );
        self.queue.submit(Some(enc.finish()));
        let raw = crate::gpuio::read_bytes(&self.device, &self.readback_buf);
        let rows: &[f32] = bytemuck::cast_slice(&raw);

        // sort by score (CPU): improving rows (score < score0) first
        let mut list: Vec<[f32; ROW]> = (0..n_prev as usize)
            .filter(|i| {
                let r = &rows[i * ROW..(i + 1) * ROW];
                r[1] != 0.0 && r[0] < score0 as f32
            })
            .map(|i| {
                let mut r = [0f32; ROW];
                r.copy_from_slice(&rows[i * ROW..(i + 1) * ROW]);
                r
            })
            .collect();
        list.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());

        if std::env::var("PG_TRACE_REUSE").is_ok() {
            let mut scs: Vec<String> = list.iter().map(|r| format!("{:.4}", r[0])).collect();
            scs.truncate(8);
            eprintln!("  score0={score0:.4} best-rescored: {}", scs.join(" "));
        }
        // re-commit retained shapes in score order (greedy: keep if it
        // still improves the running score)
        let mut kept = 0u32;
        for row in &list {
            if row[0] >= score0 as f32 {
                break;
            }
            self.queue.write_buffer(
                &self.shapes_buf,
                self.num_shapes as u64 * ROW_BYTES as u64,
                bytemuck::bytes_of(row),
            );
            self.queue
                .write_buffer(&self.winners_buf, 0, bytemuck::bytes_of(row));
            let u2 = self.uniform();
            self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&u2));
            let mut enc =
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("commit") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("commit"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.commit_pipeline);
                pass.set_bind_group(0, &self.commit_bg, &[]);
                pass.dispatch_workgroups((self.w * self.h + 63) / 64, 1, 1);
            }
            self.queue.submit(Some(enc.finish()));
            self.score = row[0] as f64;
            self.num_shapes += 1;
            kept += 1;
        }
        Ok(kept)
    }

    /// Keep only the first n shapes (lowest rescored scores come first
    /// after begin_frame_reuse, so this drops the least useful ones).
    pub fn trim_to(&mut self, n: u32) {
        if self.num_shapes > n {
            self.num_shapes = n;
        }
    }

    fn copy_prev_shapes_to_winners(&self, n: u32) -> Result<()> {
        let sz = n as u64 * ROW_BYTES as u64;
        let rb = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prev-shapes"),
            size: sz,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut e1 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        e1.copy_buffer_to_buffer(&self.shapes_buf, 0, &rb, 0, sz);
        self.queue.submit(Some(e1.finish()));
        let data = crate::gpuio::read_bytes(&self.device, &rb);
        self.queue.write_buffer(&self.winners_buf, 0, &data);
        Ok(())
    }
}

fn io_dump(data: &[u8], w: u32, h: u32) {
    let mut nonzero = 0u64;
    for c in data.chunks_exact(4) {
        if c[0] + c[1] + c[2] > 10 { nonzero += 1; }
    }
    eprintln!("raw render {}x{} nonzero-px {}", w, h, nonzero);
}

fn initial_score(target: &[u8], w: u32, h: u32, bg: [u8; 3]) -> f64 {
    let mut total = 0u64;
    for px in target.chunks_exact(4) {
        for c in 0..3 {
            let d = px[c] as i64 - bg[c] as i64;
            total += (d * d) as u64;
        }
    }
    let n = (w * h) as u64;
    (f64::sqrt(total as f64 / (w * h * 3) as f64) / 255.0) as f64
}

fn bg_packed(bg: [u8; 3]) -> u32 {
    (bg[0] as u32) | ((bg[1] as u32) << 8) | ((bg[2] as u32) << 16) | (0xff << 24)
}