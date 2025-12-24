//! GPU-accelerated execution engine for LOWESS smoothing.
//!
//! ## Purpose
//!
//! This module provides the GPU-accelerated smoothing function for LOWESS
//! operations. It leverages `wgpu` to execute local regression fits in parallel
//! on the GPU, providing maximum throughput for large-scale data processing.
//!
//! ## Optimizations
//!
//! * **Delta Interpolation**: Uses "anchors" (subset of points) for fitting,
//!   then interpolates remaining points. This reduces complexity from O(N^2)
//!   to O(Anchors * Width + N).
//!

use bytemuck::{Pod, Zeroable};
use num_traits::Float;
use std::fmt::Debug;

// Export dependencies from lowess crate
use lowess::internals::engine::executor::LowessConfig;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

// -----------------------------------------------------------------------------
// Shader Source (WGSL)
// -----------------------------------------------------------------------------
const SHADER_SOURCE: &str = r#"
struct Config {
    n: u32,
    window_size: u32,
    weight_function: u32, // Unused in this simplified shader (always Tricube)
    zero_weight_fallback: u32, // Unused
    fraction: f32,
    delta: f32,
}

struct WeightConfig {
    n: u32,
    scale: f32,
}

// Group 0: Constants & Input Data
@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> anchor_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> anchor_output: array<f32>;

// Group 1: Topology
@group(1) @binding(0) var<storage, read> interval_map: array<u32>;

// Group 2: State (Weights, Output, Residuals)
@group(2) @binding(0) var<storage, read_write> robustness_weights: array<f32>;
@group(2) @binding(1) var<storage, read_write> y_smooth: array<f32>;
@group(2) @binding(2) var<storage, read_write> residuals: array<f32>;

// Group 3: Aux (Reduction & Weight Config)
@group(3) @binding(0) var<storage, read_write> w_config: WeightConfig;
@group(3) @binding(1) var<storage, read_write> reduction: array<f32>;

// -----------------------------------------------------------------------------
// Kernel 1: Fit at Anchors
// Dispatched with num_anchors threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn fit_anchors(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let anchor_id = global_id.x;
    // Bounds check against number of anchors (passed implicitly by dispatch size)
    // We strictly need to know num_anchors, but safe if buffer is large enough/dispatch exact.
    // However, robust way implies we check bounds. We'll rely on dispatch dims matching buffer.
    if (anchor_id >= arrayLength(&anchor_indices)) {
        return;
    }

    let i = anchor_indices[anchor_id];
    
    // Window logic
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
    let right = left + window_size - 1u;

    let x_i = x[i];
    let bandwidth = max(abs(x_i - x[left]), abs(x_i - x[right]));

    if (bandwidth <= 0.0) {
        anchor_output[anchor_id] = y[i];
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
        anchor_output[anchor_id] = y[i];
    } else {
        let det = sum_w * sum_wxx - sum_wx * sum_wx;
        if (abs(det) < 1e-10) {
            anchor_output[anchor_id] = sum_wy / sum_w;
        } else {
            let a = (sum_wy * sum_wxx - sum_wxy * sum_wx) / det;
            let b = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
            anchor_output[anchor_id] = a + b * x_i;
        }
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: Interpolate
// Dispatched with N threads
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn interpolate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= config.n) { return; }

    let k = interval_map[i];
    
    // Bounds check for last segment
    let num_anchors = arrayLength(&anchor_indices);
    let idx_l_ptr = k;
    var idx_r_ptr = k + 1u;
    if (idx_r_ptr >= num_anchors) {
        idx_r_ptr = k; // Fallback to flat if at end
    }

    let idx_l = anchor_indices[idx_l_ptr];
    let idx_r = anchor_indices[idx_r_ptr];
    
    let y_l = anchor_output[idx_l_ptr];
    let y_r = anchor_output[idx_r_ptr];
    
    let x_i = x[i];
    let x_l = x[idx_l];
    let x_r = x[idx_r];

    var fitted = 0.0;
    
    if (idx_l == idx_r) {
        fitted = y_l;
    } else {
        // Linear interpolation
        let t = (x_i - x_l) / (x_r - x_l);
        fitted = y_l + (y_r - y_l) * t;
    }

    y_smooth[i] = fitted;
    residuals[i] = y[i] - fitted;
}

// -----------------------------------------------------------------------------
// Kernel 3: MAR Reduction
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Kernel 4: Finalize Scale
// -----------------------------------------------------------------------------
@compute @workgroup_size(1)
fn finalize_scale() {
    var total_sum = 0.0;
    let num_workgroups = (config.n + 255u) / 256u;
    for (var i = 0u; i < num_workgroups; i = i + 1u) {
        total_sum += reduction[i];
    }
    let mar = total_sum / f32(config.n);
    w_config.scale = max(mar, 1e-10);
}

// -----------------------------------------------------------------------------
// Kernel 5: Update Weights
// -----------------------------------------------------------------------------
@compute @workgroup_size(64)
fn update_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= w_config.n) { return; }

    let r = abs(residuals[i]);
    let tuned_scale = w_config.scale * 6.0;

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
    padding: [u32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct WeightConfig {
    n: u32,
    scale: f32,
}

#[cfg(feature = "gpu")]
struct GpuExecutor {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Pipelines
    fit_pipeline: wgpu::ComputePipeline,
    interpolate_pipeline: wgpu::ComputePipeline,
    weight_pipeline: wgpu::ComputePipeline,
    mar_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,

    // Buffers - Group 0
    config_buffer: Option<wgpu::Buffer>,
    x_buffer: Option<wgpu::Buffer>,
    y_buffer: Option<wgpu::Buffer>,
    anchor_indices_buffer: Option<wgpu::Buffer>,
    anchor_output_buffer: Option<wgpu::Buffer>,

    // Buffers - Group 1
    interval_map_buffer: Option<wgpu::Buffer>,

    // Buffers - Group 2
    weights_buffer: Option<wgpu::Buffer>,
    y_smooth_buffer: Option<wgpu::Buffer>,
    residuals_buffer: Option<wgpu::Buffer>,

    // Buffers - Group 3
    w_config_buffer: Option<wgpu::Buffer>,
    reduction_buffer: Option<wgpu::Buffer>,

    // Staging
    staging_buffer: Option<wgpu::Buffer>,

    // Bind Groups
    bg0_data: Option<wgpu::BindGroup>,
    bg1_topo: Option<wgpu::BindGroup>,
    bg2_state: Option<wgpu::BindGroup>,
    bg3_aux: Option<wgpu::BindGroup>,

    n: u32,
    num_anchors: u32,
}
impl GpuExecutor {
    async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await;

        // Handle Option or Result depending on wgpu version?
        // Error message said: `Result<Adapter, RequestAdapterError>`
        // So we just use `?` or map_err.
        let adapter = adapter.map_err(|_| "No GPU adapter found")?;

        let (device, queue): (wgpu::Device, wgpu::Queue) = adapter
            .request_device(&Default::default())
            .await
            .map_err(|e| format!("Device error: {:?}", e))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LOWESS Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        // Layouts
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG0 Data"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });

        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG1 Topo"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group_layout_2 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG2 State"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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

        let bind_group_layout_3 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("BG3 Aux"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
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
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[
                &bind_group_layout_0,
                &bind_group_layout_1,
                &bind_group_layout_2,
                &bind_group_layout_3,
            ],
            ..Default::default()
        });

        let create_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Ok(Self {
            fit_pipeline: create_pipeline("fit_anchors"),
            interpolate_pipeline: create_pipeline("interpolate"),
            mar_pipeline: create_pipeline("reduce_sum_abs"),
            finalize_pipeline: create_pipeline("finalize_scale"),
            weight_pipeline: create_pipeline("update_weights"),
            device,
            queue,
            config_buffer: None,
            x_buffer: None,
            y_buffer: None,
            anchor_indices_buffer: None,
            anchor_output_buffer: None,
            interval_map_buffer: None,
            weights_buffer: None,
            y_smooth_buffer: None,
            residuals_buffer: None,
            w_config_buffer: None,
            reduction_buffer: None,
            staging_buffer: None,
            bg0_data: None,
            bg1_topo: None,
            bg2_state: None,
            bg3_aux: None,
            n: 0,
            num_anchors: 0,
        })
    }

    fn reset_buffers(
        &mut self,
        x: &[f32],
        y: &[f32],
        anchors: &[u32],
        intervals: &[u32],
        rob_w: &[f32],
        config: GpuConfig,
    ) {
        let n = x.len() as u32;
        let num_anchors = anchors.len() as u32;
        self.n = n;
        self.num_anchors = num_anchors;

        // Group 0
        self.config_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Config"),
                contents: bytemuck::cast_slice(&[config]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        ));
        self.x_buffer = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("X"),
                    contents: bytemuck::cast_slice(x),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );
        self.y_buffer = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Y"),
                    contents: bytemuck::cast_slice(y),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );
        self.anchor_indices_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Anchors"),
                contents: bytemuck::cast_slice(anchors),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        self.anchor_output_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AnchorOutput"),
            size: (num_anchors as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        self.bg0_data = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BG0"),
                layout: &self.fit_pipeline.get_bind_group_layout(0),
                entries: &[
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
                        resource: self
                            .anchor_indices_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self
                            .anchor_output_buffer
                            .as_ref()
                            .unwrap()
                            .as_entire_binding(),
                    },
                ],
            }),
        );

        // Group 1
        self.interval_map_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("IntervalMap"),
                contents: bytemuck::cast_slice(intervals),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));
        self.bg1_topo = Some(
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BG1"),
                layout: &self.fit_pipeline.get_bind_group_layout(1),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self
                        .interval_map_buffer
                        .as_ref()
                        .unwrap()
                        .as_entire_binding(),
                }],
            }),
        );

        // Group 2
        let n_bytes = (n as usize * 4) as u64;
        self.weights_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Weights"),
                contents: bytemuck::cast_slice(rob_w),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        ));
        self.y_smooth_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("YSmooth"),
            size: n_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.residuals_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residuals"),
            size: n_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.bg2_state = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG2"),
            layout: &self.fit_pipeline.get_bind_group_layout(2),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weights_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.y_smooth_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.residuals_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));

        // Group 3
        let w_conf = WeightConfig { n, scale: 0.0 };
        self.w_config_buffer = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("WConfig"),
                contents: bytemuck::cast_slice(&[w_conf]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));
        self.reduction_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction"),
            size: (n.div_ceil(256) as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.bg3_aux = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BG3"),
            layout: &self.fit_pipeline.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.w_config_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.reduction_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));

        self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: n_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }

    fn run_pipeline(&self, pipeline: &wgpu::ComputePipeline, dispatch_size: u32, label: &str) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, self.bg0_data.as_ref().unwrap(), &[]);
            pass.set_bind_group(1, self.bg1_topo.as_ref().unwrap(), &[]);
            pass.set_bind_group(2, self.bg2_state.as_ref().unwrap(), &[]);
            pass.set_bind_group(3, self.bg3_aux.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(dispatch_size, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    fn run_iteration(&self) {
        // 1. Fit at Anchors
        self.run_pipeline(
            &self.fit_pipeline,
            self.num_anchors.div_ceil(64),
            "FitAnchors",
        );
        // 2. Interpolate
        self.run_pipeline(
            &self.interpolate_pipeline,
            self.n.div_ceil(64),
            "Interpolate",
        );
        // 3. Compute Scale (MAR)
        self.run_pipeline(&self.mar_pipeline, self.n.div_ceil(256), "MAR");
        self.run_pipeline(&self.finalize_pipeline, 1, "FinalizeScale");
        // 4. Update Weights
        self.run_pipeline(&self.weight_pipeline, self.n.div_ceil(64), "UpdateWeights");
    }

    async fn download_buffer(&self, buf: &wgpu::Buffer) -> Option<Vec<f32>> {
        let size = (self.n as usize * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, 0, self.staging_buffer.as_ref().unwrap(), 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.as_ref().unwrap().slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Some(Ok(())) = rx.receive().await {
            let data = slice.get_mapped_range();
            let ret = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            self.staging_buffer.as_ref().unwrap().unmap();
            Some(ret)
        } else {
            None
        }
    }
}

/// Perform a GPU-accelerated LOWESS fit pass.
///
/// This function executes the LOWESS algorithm on the GPU using the
/// optimized anchor-based interpolation method.
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
        let mut exec = match block_on(GpuExecutor::new()) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("GPU Err: {}", e);
                return (Vec::new(), None, 0, Vec::new());
            }
        };

        let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32().unwrap()).collect();
        let y_f32: Vec<f32> = y.iter().map(|v| v.to_f32().unwrap()).collect();
        let n = x.len();
        let delta = config.delta.to_f32().unwrap();

        // Compute Anchors & Intervals
        let mut anchors = Vec::with_capacity(n / 10);
        let mut intervals = vec![0u32; n];

        let mut last_idx = 0;
        anchors.push(0);
        let mut current_anchor_idx = 0;

        for i in 0..n {
            if i > 0 && x_f32[i] - x_f32[last_idx] > delta {
                last_idx = i;
                anchors.push(i as u32);
                current_anchor_idx += 1;
            }
            if current_anchor_idx >= 1 {
                intervals[i] = current_anchor_idx - 1;
            } else {
                intervals[i] = 0;
            }
        }
        // Ensure last point is anchor if not already
        if *anchors.last().unwrap() != (n - 1) as u32 {
            anchors.push((n - 1) as u32);
            // Fix intervals for tail? Logic above is simple "closest left anchor".
            // Correct interval logic for interpolation between A[k] and A[k+1]:
            // Point i must have intervals[i] = k where A[k] <= i <= A[k+1].
            // Re-run interval mapping strictly.
        }

        // Strict Interval Mapping
        let mut anchor_ptr = 0;
        for (i, interval) in intervals.iter_mut().enumerate() {
            while anchor_ptr + 1 < anchors.len() && (i as u32) >= anchors[anchor_ptr + 1] {
                anchor_ptr += 1;
            }
            // For i between A[ptr] and A[ptr+1], interval is ptr.
            *interval = anchor_ptr as u32;
        }

        let gpu_config = GpuConfig {
            n: n as u32,
            window_size: (config.fraction.unwrap().to_f32().unwrap() * n as f32) as u32,
            weight_function: 0,
            zero_weight_fallback: 0,
            fraction: config.fraction.unwrap().to_f32().unwrap(),
            delta,
            padding: [0, 0],
        };

        exec.reset_buffers(
            &x_f32,
            &y_f32,
            &anchors,
            &intervals,
            &vec![1.0; n],
            gpu_config,
        );

        for _ in 0..=config.iterations {
            exec.run_iteration();
        }

        let y_res = block_on(exec.download_buffer(exec.y_smooth_buffer.as_ref().unwrap())).unwrap();
        let w_res = block_on(exec.download_buffer(exec.weights_buffer.as_ref().unwrap())).unwrap();

        let y_out: Vec<T> = y_res.into_iter().map(|v| T::from(v).unwrap()).collect();
        let w_out: Vec<T> = w_res.into_iter().map(|v| T::from(v).unwrap()).collect();

        (y_out, None, config.iterations, w_out)
    }
    #[cfg(not(feature = "gpu"))]
    {
        unimplemented!("GPU feature disabled")
    }
}
