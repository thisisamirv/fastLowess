//! GPU-accelerated execution engine for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the GPU-accelerated smoothing function for LOWESS
//! operations. It leverages `wgpu` to execute local regression fits in parallel
//! on the GPU, providing maximum throughput for large-scale data processing.
//!
//! ## Design notes
//!
//! * **Implementation**: Uses `wgpu` for cross-platform GPU compute.
//! * **Shader**: Implements weighted linear regression in WGSL.
//! * **Precision**: Currently optimized for f32 (standard GPU precision).
//! * **Generics**: Generic over `Float` types for API compatibility.
//!
//! ## Key concepts
//!
//! * **Workgroup Dispatch**: Parallelizes fitting by distributing points across GPU threads.
//! * **State Management**: Orchestrates buffer uploads, shader execution, and downloads.
//! * **Performance**: Minimizes data transfer between CPU and GPU.
//!
//! ## Invariants
//!
//! * Input arrays must have finite values.
//! * Output buffer reflects the same length as input data.
//! * GPU support must be available at runtime.
//!
//! ## Non-goals
//!
//! * This module does not provide a CPU fallback (handled by other engines).
//! * This module is not optimized for small datasets due to transfer overhead.

use bytemuck::{Pod, Zeroable};
use num_traits::Float;
use std::fmt::Debug;

// Export dependencies from lowess crate
use lowess::internals::engine::executor::LowessConfig;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

/// GPU shader for LOWESS fitting.
const SHADER_SOURCE: &str = r#"
struct Config {
    n: u32,
    window_size: u32,
    weight_function: u32,
    zero_weight_fallback: u32,
    fraction: f32,
    delta: f32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> robustness_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> y_smooth: array<f32>;
@group(0) @binding(5) var<storage, read_write> residuals: array<f32>;
@group(0) @binding(6) var<storage, read_write> w_config: WeightConfig;
@group(0) @binding(7) var<storage, read_write> reduction: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) {
        return;
    }

    // Window logic (simplified for GPU parallelization)
    let n = config.n;
    let window_size = config.window_size;
    
    var left = 0u;
    if (i > window_size / 2u) {
        left = i - window_size / 2u;
    }
    
    if (left + window_size > n) {
        if (n > window_size) {
            left = n - window_size;
        } else {
            left = 0u;
        }
    }
    let right = left + window_size - 1;

    let x_i = x[i];
    let bandwidth = max(abs(x_i - x[left]), abs(x_i - x[right]));

    if (bandwidth <= 0.0) {
        y_smooth[i] = y[i];
        residuals[i] = 0.0;
        return;
    }

    // Weighted linear regression
    var sum_w = 0.0;
    var sum_wx = 0.0;
    var sum_wxx = 0.0;
    var sum_wy = 0.0;
    var sum_wxy = 0.0;

    for (var j = left; j <= right; j = j + 1u) {
        let dist = abs(x[j] - x_i);
        let u = dist / bandwidth;
        
        // Tricube kernel
        var w = 0.0;
        if (u < 1.0) {
            let tmp = 1.0 - u * u * u;
            w = tmp * tmp * tmp;
        }
        
        let rw = robustness_weights[j];
        let combined_w = w * rw;

        let xj = x[j];
        let yj = y[j];
        
        sum_w += combined_w;
        sum_wx += combined_w * xj;
        sum_wxx += combined_w * xj * xj;
        sum_wy += combined_w * yj;
        sum_wxy += combined_w * xj * yj;
    }

    if (sum_w <= 0.0) {
        y_smooth[i] = y[i];
        residuals[i] = 0.0;
    } else {
        let det = sum_w * sum_wxx - sum_wx * sum_wx;
        if (abs(det) < 1e-10) {
            y_smooth[i] = sum_wy / sum_w;
        } else {
            let a = (sum_wy * sum_wxx - sum_wxy * sum_wx) / det;
            let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
            y_smooth[i] = a + b * x_i;
        }
        residuals[i] = y[i] - y_smooth[i];
    }
}

struct WeightConfig {
    n: u32,
    scale: f32,
}

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum_abs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let i = global_id.x;
    var val = 0.0;
    if (i < config.n) {
        val = abs(residuals[i]);
    }
    
    scratch[local_id.x] = val;
    workgroupBarrier();
    
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (local_id.x < s) {
            scratch[local_id.x] += scratch[local_id.x + s];
        }
        workgroupBarrier();
    }
    
    if (local_id.x == 0u) {
        reduction[workgroup_id.x] = scratch[0];
    }
}

// Final reduction pass to compute MAR and set scale in w_config
@compute @workgroup_size(1)
fn finalize_scale() {
    var total_sum = 0.0;
    let num_workgroups = (config.n + 255u) / 256u;
    for (var i = 0u; i < num_workgroups; i = i + 1u) {
        total_sum += reduction[i];
    }
    
    let mar = total_sum / f32(config.n);
    
    // Robust scale fallback logic for f32
    let scale_threshold = max(1e-4 * mar, 1e-10);
    w_config.scale = max(mar, scale_threshold);
}

@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= w_config.n) {
        return;
    }

    let r = abs(residuals[i]);
    let tuned_scale = w_config.scale * 6.0; // DEFAULT_BISQUARE_C = 6.0

    if (tuned_scale <= 1e-12) {
        robustness_weights[i] = 1.0;
    } else {
        let u = r / tuned_scale;
        if (u < 1.0) {
            let tmp = 1.0 - u * u;
            robustness_weights[i] = tmp * tmp;
        } else {
            robustness_weights[i] = 0.0;
        }
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuConfig {
    n: u32,
    window_size: u32,
    weight_function: u32,
    zero_weight_fallback: u32,
    fraction: f32,
    delta: f32,
    padding: [u32; 2], // Pad to 8-byte alignment (total 32 bytes)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WeightConfig {
    n: u32,
    scale: f32,
}

/// GPU-accelerated executor for LOWESS operations.
#[cfg(feature = "gpu")]
struct GpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fit_pipeline: wgpu::ComputePipeline,
    weight_pipeline: wgpu::ComputePipeline,

    // Persistent buffers
    config_buffer: Option<wgpu::Buffer>,
    weight_config_buffer: Option<wgpu::Buffer>,
    x_buffer: Option<wgpu::Buffer>,
    y_buffer: Option<wgpu::Buffer>,
    weights_buffer: Option<wgpu::Buffer>,
    output_buffer: Option<wgpu::Buffer>,
    residuals_buffer: Option<wgpu::Buffer>,
    reduction_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,

    // Bind group
    bind_group: Option<wgpu::BindGroup>,
    weight_bind_group: Option<wgpu::BindGroup>,
    mar_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,

    n: u32,
}

#[cfg(feature = "gpu")]
impl GpuExecutor {
    /// Initialize the GPU executor.
    async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter_res = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter_res {
            Ok(a) => a,
            Err(e) => return Err(format!("No suitable GPU adapter found: {:?}", e)),
        };

        let device_res = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("LOWESS Compute Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await;

        let (device, queue): (wgpu::Device, wgpu::Queue) = match device_res {
            Ok(d) => d,
            Err(e) => return Err(format!("Device request failed: {:?}", e)),
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LOWESS Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Explicit Bind Group Layout for all 7 bindings
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LOWESS Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LOWESS Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });

        let fit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LOWESS Fit Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let weight_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LOWESS Weight Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_weights"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Finalize Scale Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("finalize_scale"),
            compilation_options: Default::default(),
            cache: None,
        });

        let mar_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MAR Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_sum_abs"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            fit_pipeline,
            weight_pipeline,
            mar_pipeline,
            finalize_pipeline,
            config_buffer: None,
            weight_config_buffer: None,
            x_buffer: None,
            y_buffer: None,
            weights_buffer: None,
            output_buffer: None,
            residuals_buffer: None,
            reduction_buffer: None,
            staging_buffer: None,
            bind_group: None,
            weight_bind_group: None,
            n: 0,
        })
    }

    /// Reset buffers for a new dataset.
    fn reset_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        robustness_weights: &[f32],
        config: GpuConfig,
    ) {
        let n = x.len() as u32;
        self.n = n;
        let size = (n as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

        // 1. Create/Recreate Buffers
        self.config_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: bytemuck::cast_slice(&[config]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        ));

        let weight_config = WeightConfig { n, scale: 0.0 };
        self.weight_config_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Weight Config Buffer"),
                contents: bytemuck::cast_slice(&[weight_config]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));

        self.x_buffer = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("X Buffer"),
                    contents: bytemuck::cast_slice(x),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );

        self.y_buffer = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Y Buffer"),
                    contents: bytemuck::cast_slice(y),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );

        self.weights_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Weights Buffer"),
                contents: bytemuck::cast_slice(robustness_weights),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            },
        ));

        self.output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.residuals_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residuals Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.reduction_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Buffer"),
            size: (n.div_ceil(256) as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // 2. Create Bind Groups (Shared Layout 0-6)
        let bind_group_layout = self.fit_pipeline.get_bind_group_layout(0);

        let common_entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: self.config_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.x_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.y_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.weights_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: self.output_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: self.residuals_buffer.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: self
                    .weight_config_buffer
                    .as_ref()
                    .unwrap()
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: self.reduction_buffer.as_ref().unwrap().as_entire_binding(),
            },
        ];

        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOWESS Bind Group"),
            layout: &bind_group_layout,
            entries: &common_entries,
        }));

        self.weight_bind_group = None; // We now use a single shared bind group

        // Wait, the weight shader bindings are DIFFERENT.
        // @group(0) @binding(0) var<uniform> w_config: WeightConfig;
        // binding 3, 5 are shared.
        // Let's re-verify the shader group(0) bindings for update_weights.
    }

    /// Dispatch the fit kernel.
    fn run_fit(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fit Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fit Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.fit_pipeline);
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            let workgroups = self.n.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Compute the robust scale (MAR) entirely on the GPU.
    fn run_compute_scale(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Scale Computation Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MAR Reduction Pass 1"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.mar_pipeline);
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            let num_workgroups = self.n.div_ceil(256);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Finalize Scale Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.finalize_pipeline);
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            compute_pass.dispatch_workgroups(1, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Update robustness weights on the GPU.
    /// If `scale` is provided, it is uploaded to the GPU first.
    /// Otherwise, it uses the scale already computed on the GPU.
    fn run_update_weights(&self, scale: Option<f32>) {
        if let Some(s) = scale {
            let w_config = WeightConfig {
                n: self.n,
                scale: s,
            };
            self.queue.write_buffer(
                self.weight_config_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&[w_config]),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Weight Update Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Weight Update Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.weight_pipeline);
            compute_pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            let workgroups = self.n.div_ceil(64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Download a buffer to the CPU.
    async fn download_buffer(&self, buffer: &wgpu::Buffer) -> Result<Vec<f32>, String> {
        let size = (self.n as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Download Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, self.staging_buffer.as_ref().unwrap(), 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.staging_buffer.as_ref().unwrap().slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.as_ref().unwrap().unmap();
            Ok(result)
        } else {
            Err("Failed to map staging buffer".to_string())
        }
    }

    async fn download_y_smooth(&self) -> Result<Vec<f32>, String> {
        self.download_buffer(self.output_buffer.as_ref().unwrap())
            .await
    }

    async fn download_weights(&self) -> Result<Vec<f32>, String> {
        self.download_buffer(self.weights_buffer.as_ref().unwrap())
            .await
    }
}

/// Perform LOWESS fitting on the GPU.
pub fn fit_pass_gpu<T>(
    x: &[T],
    y: &[T],
    config: &LowessConfig<T>,
) -> (Vec<T>, Option<Vec<T>>, usize, Vec<T>)
where
    T: Float + Debug + Send + Sync + 'static,
{
    #[cfg(feature = "gpu")]
    {
        use pollster::block_on;

        let mut executor = match block_on(GpuExecutor::new()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("GPU initialization failed: {}. Falling back to CPU.", e);
                return (Vec::new(), None, 0, Vec::new());
            }
        };

        let n = x.len();
        let x_f32: Vec<f32> = x.iter().map(|&v| v.to_f32().unwrap()).collect();
        let y_f32: Vec<f32> = y.iter().map(|&v| v.to_f32().unwrap()).collect();
        let initial_weights = vec![1.0f32; n];

        let window_size = (config
            .fraction
            .unwrap_or_else(|| T::from(0.67).unwrap())
            .to_f64()
            .unwrap()
            * n as f64)
            .ceil() as u32;

        let gpu_config = GpuConfig {
            n: n as u32,
            window_size,
            weight_function: 0,
            zero_weight_fallback: 0,
            fraction: config
                .fraction
                .unwrap_or_else(|| T::from(0.67).unwrap())
                .to_f32()
                .unwrap(),
            delta: config.delta.to_f32().unwrap(),
            padding: [0, 0],
        };

        executor.reset_buffers(&x_f32, &y_f32, &initial_weights, gpu_config);

        for iter in 0..=config.iterations {
            // 1. Fit on GPU
            executor.run_fit();

            if iter < config.iterations {
                // 2. Compute scale on GPU (Pure GPU path)
                executor.run_compute_scale();

                // 3. Update weights on GPU
                executor.run_update_weights(None);
            }
        }

        // Final download
        let y_smooth_f32 = match block_on(executor.download_y_smooth()) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("GPU final download failed: {}. Returning empty.", e);
                return (Vec::new(), None, 0, Vec::new());
            }
        };

        let rob_weights_f32 = match block_on(executor.download_weights()) {
            Ok(w) => w,
            Err(_) => vec![1.0f32; n],
        };

        // Convert back to T
        let smoothed: Vec<T> = y_smooth_f32.iter().map(|&v| T::from(v).unwrap()).collect();
        let rob_weights: Vec<T> = rob_weights_f32
            .iter()
            .map(|&v| T::from(v).unwrap())
            .collect();

        (smoothed, None, config.iterations, rob_weights)
    }

    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature not enabled")
    }
}
