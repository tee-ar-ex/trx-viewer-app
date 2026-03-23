use wgpu::util::DeviceExt;

use crate::data::gifti_data::GiftiSurfaceData;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshUniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
    camera_pos: [f32; 3],
    shininess: f32,
    specular_strength: f32,
    ambient_strength: f32,
    map_opacity: f32,
    map_threshold: f32,
    scalar_min: f32,
    scalar_max: f32,
    scalar_enabled: u32,
    colormap: u32,
    _pad: [u32; 1],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

pub struct MeshDrawStyle {
    pub color: [f32; 4],
    pub scalar_min: f32,
    pub scalar_max: f32,
    pub scalar_enabled: bool,
    pub colormap: SurfaceColormap,
    pub ambient_strength: f32,
    pub gloss: f32,
    pub map_opacity: f32,
    pub map_threshold: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SurfaceColormap {
    BlueWhiteRed = 0,
    Viridis = 1,
    Inferno = 2,
}

pub struct MeshResources {
    pub pipeline: wgpu::RenderPipeline,
    surfaces: Vec<GpuSurface>,
}

struct GpuSurface {
    vertex_buffer: wgpu::Buffer,
    scalar_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffers: [wgpu::Buffer; 4],
    bind_groups: [wgpu::BindGroup; 4],
}

impl MeshResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/mesh.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mesh_bind_group_layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                    ],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<f32>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        offset: 0,
                        shader_location: 2,
                        format: wgpu::VertexFormat::Float32,
                    }],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
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

        Self {
            pipeline,
            surfaces: Vec::new(),
        }
    }

    pub fn add_surface(
        &mut self,
        device: &wgpu::Device,
        surface: &GiftiSurfaceData,
    ) -> usize {
        let vertices: Vec<MeshVertex> = surface
            .vertices
            .iter()
            .zip(surface.normals.iter())
            .map(|(position, normal)| MeshVertex {
                position: *position,
                normal: *normal,
            })
            .collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_indices"),
            contents: bytemuck::cast_slice(&surface.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let scalar_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gifti_surface_scalars"),
            contents: bytemuck::cast_slice(&vec![0.0f32; surface.vertices.len()]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let default_uniforms = MeshUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            color: [0.7, 0.7, 0.7, 1.0],
            camera_pos: [0.0, 0.0, 1.0],
            shininess: 24.0,
            specular_strength: 0.18,
            ambient_strength: 0.42,
            map_opacity: 1.0,
            map_threshold: 0.0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            scalar_enabled: 0,
            colormap: SurfaceColormap::BlueWhiteRed as u32,
            _pad: [0],
        };

        let uniform_buffers: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("mesh_uniforms_{i}")),
                contents: bytemuck::bytes_of(&default_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        });
        let bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("mesh_bind_group_{i}")),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffers[i].as_entire_binding(),
                }],
            })
        });

        self.surfaces.push(GpuSurface {
            vertex_buffer,
            scalar_buffer,
            index_buffer,
            num_indices: surface.indices.len() as u32,
            uniform_buffers,
            bind_groups,
        });
        self.surfaces.len() - 1
    }

    pub fn update_surface_uniforms(
        &self,
        queue: &wgpu::Queue,
        surface_index: usize,
        viewport: usize,
        view_proj: glam::Mat4,
        style: &MeshDrawStyle,
        camera_pos: glam::Vec3,
    ) {
        if let Some(surface) = self.surfaces.get(surface_index) {
            let uniforms = MeshUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                color: style.color,
                camera_pos: camera_pos.into(),
                shininess: 8.0 + 80.0 * style.gloss.clamp(0.0, 1.0),
                specular_strength: 0.02 + 0.28 * style.gloss.clamp(0.0, 1.0),
                ambient_strength: style.ambient_strength.clamp(0.0, 1.0),
                map_opacity: style.map_opacity.clamp(0.0, 1.0),
                map_threshold: style.map_threshold.clamp(0.0, 1.0),
                scalar_min: style.scalar_min,
                scalar_max: style.scalar_max,
                scalar_enabled: if style.scalar_enabled { 1 } else { 0 },
                colormap: style.colormap as u32,
                _pad: [0],
            };
            queue.write_buffer(
                &surface.uniform_buffers[viewport],
                0,
                bytemuck::bytes_of(&uniforms),
            );
        }
    }

    pub fn update_surface_scalars(
        &self,
        queue: &wgpu::Queue,
        surface_index: usize,
        scalars: &[f32],
    ) {
        if let Some(surface) = self.surfaces.get(surface_index) {
            queue.write_buffer(&surface.scalar_buffer, 0, bytemuck::cast_slice(scalars));
        }
    }

    pub fn paint(
        &self,
        render_pass: &mut wgpu::RenderPass<'static>,
        viewport: usize,
        draw_calls: &[(usize, MeshDrawStyle)],
    ) {
        render_pass.set_pipeline(&self.pipeline);
        for (surface_index, style) in draw_calls {
            if let Some(surface) = self.surfaces.get(*surface_index) {
                if style.color[3] <= 0.001 {
                    continue;
                }
                render_pass.set_bind_group(0, &surface.bind_groups[viewport], &[]);
                render_pass.set_vertex_buffer(0, surface.vertex_buffer.slice(..));
                render_pass.set_vertex_buffer(1, surface.scalar_buffer.slice(..));
                render_pass.set_index_buffer(
                    surface.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..surface.num_indices, 0, 0..1);
            }
        }
    }
}
