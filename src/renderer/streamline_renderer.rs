use wgpu::util::DeviceExt;

use crate::data::trx_data::TrxGpuData;

/// GPU resources for streamline rendering.
pub struct StreamlineResources {
    pub pipeline: wgpu::RenderPipeline,
    /// Pipeline for slice views (depth test Always, so streamlines aren't hidden behind NIfTI quad).
    pub slice_pipeline: wgpu::RenderPipeline,
    pub position_buffer: wgpu::Buffer,
    pub color_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    /// Per-viewport uniform buffers: [3D, axial, coronal, sagittal].
    pub uniform_buffers: [wgpu::Buffer; 4],
    /// Per-viewport bind groups.
    pub bind_groups: [wgpu::BindGroup; 4],
    pub num_indices: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    /// Slab clipping axis: 0=X, 1=Y, 2=Z, 3=disabled.
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
    _pad: u32,
}

impl StreamlineResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        data: &TrxGpuData,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("streamline_shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/streamline.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("streamline_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create 4 uniform buffers and bind groups (3D + 3 slice viewports)
        let default_uniforms = Uniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            slab_axis: 3, // disabled
            slab_min: 0.0,
            slab_max: 0.0,
            _pad: 0,
        };

        let uniform_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("streamline_uniforms_{i}")),
                contents: bytemuck::bytes_of(&default_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });

        let bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("streamline_bind_group_{i}")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffers[i].as_entire_binding(),
                }],
            })
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("streamline_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Separate position and color vertex buffer layouts
        let position_layout = wgpu::VertexBufferLayout {
            array_stride: (3 * std::mem::size_of::<f32>()) as wgpu::BufferAddress, // 12 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        };

        let color_layout = wgpu::VertexBufferLayout {
            array_stride: (4 * std::mem::size_of::<f32>()) as wgpu::BufferAddress, // 16 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            }],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamline_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[position_layout, color_layout],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        // Slice view pipeline: depth test Always so streamlines aren't hidden behind NIfTI quad
        let slice_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("streamline_slice_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        }],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        }],
                    },
                ],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        let position_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_positions"),
            contents: bytemuck::cast_slice(&data.positions),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_colors"),
            contents: bytemuck::cast_slice(&data.colors),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("streamline_indices"),
            contents: bytemuck::cast_slice(&data.all_indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            slice_pipeline,
            position_buffer,
            color_buffer,
            index_buffer,
            uniform_buffers,
            bind_groups,
            num_indices: data.all_indices.len() as u32,
        }
    }

    /// Update uniforms for a specific viewport.
    /// `viewport`: 0=3D, 1=axial, 2=coronal, 3=sagittal.
    /// `slab_axis`: 0=X, 1=Y, 2=Z, 3=disabled.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        viewport: usize,
        view_proj: glam::Mat4,
        slab_axis: u32,
        slab_min: f32,
        slab_max: f32,
    ) {
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            slab_axis,
            slab_min,
            slab_max,
            _pad: 0,
        };
        queue.write_buffer(&self.uniform_buffers[viewport], 0, bytemuck::bytes_of(&uniforms));
    }

    /// Re-upload only the color buffer.
    pub fn update_colors(&self, queue: &wgpu::Queue, colors: &[[f32; 4]]) {
        queue.write_buffer(&self.color_buffer, 0, bytemuck::cast_slice(colors));
    }

    /// Replace the index buffer contents and update the draw count.
    pub fn update_indices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        indices: &[u32],
    ) {
        let new_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
        if new_size <= self.index_buffer.size() {
            queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(indices));
        } else {
            self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("streamline_indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            });
        }
        self.num_indices = indices.len() as u32;
    }
}
