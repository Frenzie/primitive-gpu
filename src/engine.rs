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
    render_bg: wgpu::BindGroup,
    optimize_pipeline: wgpu::ComputePipeline,
    commit_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::ComputePipeline,
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
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const WG_COUNT: u32 = 64; // hill-climb chains per step
const ROUNDS: u32 = 24; // mutations per chain
const N_RANDOM: u32 = 16; // random candidates per chain start

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
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no Vulkan-capable adapter found"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("primitive-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;

        let n_px = (w * h) as u64;
        let target_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("target"),
            contents: &target,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let canvas_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("canvas"),
            size: n_px * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let winners_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("winners"),
            size: WG_COUNT as u64 * ROW_BYTES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let shapes_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shapes"),
            size: ROW_BYTES as u64 * MAX_SHAPES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        let mk_mod = |src: &str, label: &'static str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.to_string().into()),
            })
        };
        let optimize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("optimize"),
            layout: None,
            module: &mk_mod(OPTIMIZE_WGSL, "optimize"),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let commit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("commit"),
            layout: None,
            module: &mk_mod(COMMIT_WGSL, "commit"),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let render_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("render"),
            layout: None,
            module: &mk_mod(RENDER_WGSL, "render"),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind = |pipeline: &wgpu::ComputePipeline| {
            let layout = pipeline.get_bind_group_layout(0);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: target_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: canvas_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: winners_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: shapes_buf.as_entire_binding() },
                ],
            })
        };
        let optimize_bg = bind(&optimize_pipeline);
        let commit_bg = bind(&commit_pipeline);
        let render_bg = bind(&render_pipeline);

        // seed canvas with bg color
        let bg_u = bg_packed(bg);
        let canvas = vec![bg_u; (w * h) as usize];
        queue.write_buffer(&canvas_buf, 0, bytemuck::cast_slice(&canvas));

        Ok(Engine {
            device,
            queue,
            target_buf,
            canvas_buf,
            winners_buf,
            shapes_buf,
            params_buf,
            optimize_bg,
            commit_bg,
            render_bg,
            optimize_pipeline,
            commit_pipeline,
            render_pipeline,
            readback_buf,
            w,
            h,
            out_w,
            out_h,
            ss,
            bg,
            num_shapes: 0,
            score: 1.0,
            shape_type: MODE_TRIANGLE,
            alpha: 128,
            frame_seed: 0,
        })
    }

    /// One optimization step: find the best shape and commit it to the canvas.
    pub fn step(&mut self, shape_type: i32, alpha: i32, frame_seed: u32) -> Result<[f32; ROW]> {
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
            pass.dispatch_workgroups((w_h(self.w, self.h) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(enc.finish()));
        self.score = best[0] as f64;
        Ok(best)
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
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    /// High-quality analytic render at out_w x out_h (supersampled).
    pub fn render(&self) -> Result<Vec<u8>> {
        let w = self.out_w * self.ss;
        let h = self.out_h * self.ss;
        let n_px = (w * h) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-out"),
            size: n_px * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rb_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-read"),
            size: n_px * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let render_bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.target_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.canvas_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.winners_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.shapes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buf.as_entire_binding() },
            ],
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
            pass.set_bind_group(0, &render_bg2, &[]);
            pass.dispatch_workgroups(((n_px + 63) / 64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &rb_buf, 0, n_px * 4);
        self.queue.submit(Some(enc.finish()));
        let data = crate::gpuio::read_bytes(&self.device, &rb_buf);

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

    pub fn set_shape_type(&mut self, t: i32) {
        self.shape_type = t;
    }

    pub fn set_alpha(&mut self, a: i32) {
        self.alpha = a;
    }

    pub fn set_frame_seed(&mut self, s: u32) {
        self.frame_seed = s;
    }

    /// Start a fresh optimization for a new video frame: reset shape count
    /// and re-seed the canvas to bg. (Temporal reuse is a later flag.)
    pub fn reset_for_frame(&mut self, target: Vec<u8>) {
        self.num_shapes = 0;
        self.score = 1.0;
        self.queue.write_buffer(
            &self.target_buf,
            0,
            &target,
        );
        let bg_u = bg_packed(self.bg);
        let canvas = vec![bg_u; (self.w * self.h) as usize];
        self.queue.write_buffer(&self.canvas_buf, 0, bytemuck::cast_slice(&canvas));
    }
}

fn w_h(w: u32, h: u32) -> u32 {
    w * h
}

fn bg_packed(bg: [u8; 3]) -> u32 {
    (bg[0] as u32) | ((bg[1] as u32) << 8) | ((bg[2] as u32) << 16) | (0xff << 24)
}