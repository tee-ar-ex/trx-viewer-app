use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::app::AppSceneLightingParams as SceneLightingParams;
use crate::data::bundle_mesh::{BundleMesh, BundleMeshVertex};
use crate::data::gifti_data::GiftiSurfaceData;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MeshUniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
    camera_pos: [f32; 3],
    shininess: f32,
    map_opacity: f32,
    map_threshold: f32,
    scalar_min: f32,
    scalar_max: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    scalar_enabled: u32,
    colormap: u32,
    _pad: [u32; 3],
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BundleUniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    opacity: f32,
    ambient_strength: f32,
    key_strength: f32,
    fill_strength: f32,
    headlight_mix: f32,
    specular_strength: f32,
    _pad: [f32; 3],
}

struct BundleGpuSurface {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct MeshResources {
    pub pipeline: wgpu::RenderPipeline,
    surfaces: Vec<GpuSurface>,
    // Bundle surfaces — keyed by file_id, one Vec per TRX file
    pub bundle_pipeline: wgpu::RenderPipeline,
    bundle_surfaces: HashMap<usize, Vec<BundleGpuSurface>>,
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
                buffers: &[
                    wgpu::VertexBufferLayout {
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
                    },
                ],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
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

        // ── Bundle mesh pipeline ──────────────────────────────────────────────
        let bundle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bundle_mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bundle_mesh.wgsl").into()),
        });

        let bundle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bundle_mesh_bgl"),
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

        let bundle_pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bundle_mesh_pl_layout"),
            bind_group_layouts: &[&bundle_bgl],
            push_constant_ranges: &[],
        });

        let bundle_stride = std::mem::size_of::<BundleMeshVertex>() as wgpu::BufferAddress;
        let bundle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bundle_mesh_pipeline"),
            layout: Some(&bundle_pl_layout),
            vertex: wgpu::VertexState {
                module: &bundle_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: bundle_stride,
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
                        wgpu::VertexAttribute {
                            offset: 24,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // two-sided (handled in shader)
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
                module: &bundle_shader,
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
            bundle_pipeline,
            bundle_surfaces: HashMap::new(),
        }
    }

    pub fn add_surface(&mut self, device: &wgpu::Device, surface: &GiftiSurfaceData) -> usize {
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
            map_opacity: 1.0,
            map_threshold: 0.0,
            scalar_min: 0.0,
            scalar_max: 1.0,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            scalar_enabled: 0,
            colormap: SurfaceColormap::BlueWhiteRed as u32,
            _pad: [0; 3],
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
        scene_lighting: SceneLightingParams,
    ) {
        if let Some(surface) = self.surfaces.get(surface_index) {
            let uniforms = MeshUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                color: style.color,
                camera_pos: camera_pos.into(),
                shininess: 8.0 + 80.0 * style.gloss.clamp(0.0, 1.0),
                map_opacity: style.map_opacity.clamp(0.0, 1.0),
                map_threshold: style.map_threshold.clamp(0.0, 1.0),
                scalar_min: style.scalar_min,
                scalar_max: style.scalar_max,
                ambient_strength: scene_lighting.ambient_strength(),
                key_strength: scene_lighting.key_strength(),
                fill_strength: scene_lighting.fill_strength(),
                headlight_mix: scene_lighting.headlight_mix(),
                specular_strength: scene_lighting.specular_strength()
                    * (0.15 + 0.85 * style.gloss.clamp(0.0, 1.0)),
                scalar_enabled: if style.scalar_enabled { 1 } else { 0 },
                colormap: style.colormap as u32,
                _pad: [0; 3],
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
                render_pass
                    .set_index_buffer(surface.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..surface.num_indices, 0, 0..1);
            }
        }
    }

    // ── Bundle meshes (voxel density surfaces) ───────────────────────────────

    fn make_bundle_gpu_surface(
        &self,
        device: &wgpu::Device,
        mesh: &BundleMesh,
        label: &str,
    ) -> BundleGpuSurface {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_verts_{label}")),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_idx_{label}")),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let default_uniforms = BundleUniforms {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            camera_pos: [0.0; 3],
            opacity: 0.5,
            ambient_strength: 0.46,
            key_strength: 0.34,
            fill_strength: 0.18,
            headlight_mix: 0.18,
            specular_strength: 0.14,
            _pad: [0.0; 3],
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("bundle_uni_{label}")),
            contents: bytemuck::bytes_of(&default_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bgl = self.bundle_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("bundle_bg_{label}")),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        BundleGpuSurface {
            vertex_buffer,
            index_buffer,
            num_indices: mesh.indices.len() as u32,
            uniform_buffer,
            bind_group,
        }
    }

    /// Replace the bundle surfaces for one TRX file.
    /// `meshes` is a slice of `(mesh, label)` pairs — one entry per source
    /// (e.g. one for all streamlines, or one per group).
    pub fn set_bundle_meshes(
        &mut self,
        file_id: usize,
        device: &wgpu::Device,
        meshes: &[(BundleMesh, String)],
    ) {
        let surfaces = meshes
            .iter()
            .map(|(m, label)| self.make_bundle_gpu_surface(device, m, label))
            .collect();
        self.bundle_surfaces.insert(file_id, surfaces);
    }

    /// Update per-frame uniforms for one TRX file's bundle surfaces.
    pub fn update_bundle_uniforms(
        &self,
        file_id: usize,
        queue: &wgpu::Queue,
        view_proj: glam::Mat4,
        camera_pos: glam::Vec3,
        opacity: f32,
        scene_lighting: SceneLightingParams,
    ) {
        if let Some(surfaces) = self.bundle_surfaces.get(&file_id) {
            let uniforms = BundleUniforms {
                view_proj: view_proj.to_cols_array_2d(),
                camera_pos: camera_pos.into(),
                opacity,
                ambient_strength: scene_lighting.ambient_strength(),
                key_strength: scene_lighting.key_strength(),
                fill_strength: scene_lighting.fill_strength(),
                headlight_mix: scene_lighting.headlight_mix(),
                specular_strength: scene_lighting.specular_strength(),
                _pad: [0.0; 3],
            };
            for bs in surfaces {
                queue.write_buffer(&bs.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
            }
        }
    }

    /// Draw bundle surfaces only for the active file ids in the current frame.
    pub fn paint_bundle(&self, render_pass: &mut wgpu::RenderPass<'static>, file_ids: &[usize]) {
        if self.bundle_surfaces.is_empty() || file_ids.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.bundle_pipeline);
        for file_id in file_ids {
            if let Some(surfaces) = self.bundle_surfaces.get(file_id) {
                for bs in surfaces {
                    render_pass.set_bind_group(0, &bs.bind_group, &[]);
                    render_pass.set_vertex_buffer(0, bs.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(bs.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..bs.num_indices, 0, 0..1);
                }
            }
        }
    }

    pub fn clear_bundle_mesh(&mut self, file_id: usize) {
        self.bundle_surfaces.remove(&file_id);
    }
}
