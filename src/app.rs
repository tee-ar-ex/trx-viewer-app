use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use glam::Vec3;

use crate::data::bundle_mesh::{build_bundle_mesh, BundleMesh};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::{colormap_bwr, ColorMode, RenderStyle, scalar_auto_range, TrxGpuData};
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::mesh_renderer::{MeshDrawStyle, MeshResources, SurfaceColormap};
use crate::renderer::slice_renderer::{SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::StreamlineResources;

/// Source for the bundle surface mesh.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BundleMeshSource {
    /// All streamlines in the file.
    All,
    /// Currently filtered / visible selection.
    Selection,
    /// One mesh per TRX group.
    PerGroup,
}

struct LoadedGiftiSurface {
    name: String,
    path: PathBuf,
    data: GiftiSurfaceData,
    gpu_index: usize,
    visible: bool,
    opacity: f32,
    color: [f32; 3],
    show_projection_map: bool,
    map_opacity: f32,
    map_threshold: f32,
    surface_ambient: f32,
    surface_gloss: f32,
    projection_mode: SurfaceProjectionMode,
    projection_dps: Option<String>,
    projection_depth_mm: f32,
    projection_colormap: SurfaceColormap,
    auto_range: bool,
    range_min: f32,
    range_max: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SurfaceProjectionMode {
    Density,
    MeanDps,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct SurfaceProjectionCacheKey {
    surface_idx: usize,
    selection_revision: u64,
    depth_bin: i32,
    mode: SurfaceProjectionMode,
    dps_name: Option<String>,
}

#[derive(Clone)]
struct SurfaceProjectionCacheValue {
    density: Vec<f32>,
    mean_dps: Vec<f32>,
    data_min: f32,
    data_max: f32,
}

/// Main application state.
pub struct TrxViewerApp {
    trx_data: Option<TrxGpuData>,
    nifti_volume: Option<NiftiVolume>,
    camera_3d: OrbitCamera,
    slice_cameras: [OrthoSliceCamera; 3],
    /// Current slice indices [axial(k), coronal(j), sagittal(i)].
    slice_indices: [usize; 3],
    slices_dirty: bool,
    has_streamlines: bool,
    has_slices: bool,
    /// Loaded file paths for display.
    trx_path: Option<PathBuf>,
    nifti_path: Option<PathBuf>,
    /// Volume center/extent for camera reset.
    volume_center: Vec3,
    volume_extent: f32,
    /// Error message to display.
    error_msg: Option<String>,
    /// Loaded GIFTI surfaces.
    gifti_surfaces: Vec<LoadedGiftiSurface>,
    /// Optional streamline filter based on distance to a selected surface.
    surface_query_active: bool,
    surface_query_surface: usize,
    surface_query_result: Option<HashSet<u32>>,
    /// Monotonic revision of streamline selection/filter state.
    selection_revision: u64,
    /// Cached surface projection maps keyed by active parameters.
    surface_projection_cache: HashMap<SurfaceProjectionCacheKey, SurfaceProjectionCacheValue>,
    /// Whether projections should be recomputed from current filters/settings.
    surface_projection_dirty: bool,
    /// Current coloring mode.
    color_mode: ColorMode,
    /// Whether vertex colors need re-upload.
    colors_dirty: bool,
    /// Global streamline visibility toggle.
    show_streamlines: bool,
    /// Group visibility (one bool per group, all visible by default).
    group_visible: Vec<bool>,
    /// Whether the index buffer needs rebuild.
    indices_dirty: bool,
    /// Uniform color picker value.
    uniform_color: [f32; 4],
    /// Volume intensity windowing.
    window_center: f32,
    window_width: f32,
    /// Max number of streamlines to display.
    max_streamlines: usize,
    /// Whether to randomize which streamlines are shown.
    use_random_subset: bool,
    /// Ordering of streamline indices (identity or shuffled).
    streamline_order: Vec<u32>,
    /// Slab half-width for slice view streamline clipping (mm).
    slab_half_width: f32,
    /// Active rendering style for streamlines.
    render_style: RenderStyle,
    /// Tube radius in mm (for Tubes mode).
    tube_radius: f32,
    /// Per-slice visibility in the 3D viewport [axial, coronal, sagittal].
    slice_visible: [bool; 3],
    /// World-space slice positions used when no NIfTI is loaded [axial(Z), coronal(Y), sagittal(X)].
    slice_world_offsets: [f32; 3],
    /// Whether scalar colormap range is auto-computed from data.
    scalar_auto_range: bool,
    /// Scalar colormap range (min, max) — used for DPV/DPS coloring.
    scalar_range_min: f32,
    scalar_range_max: f32,
    /// Sphere query state.
    sphere_query_active: bool,
    sphere_center: Vec3,
    sphere_radius: f32,
    sphere_query_result: Option<HashSet<u32>>,
    /// Bundle surface (voxel density mesh).
    show_bundle_mesh: bool,
    bundle_mesh_source: BundleMeshSource,
    bundle_mesh_voxel_size: f32,
    bundle_mesh_threshold: f32,
    bundle_mesh_smooth: f32,
    bundle_mesh_opacity: f32,
    bundle_mesh_ambient: f32,
    /// CPU copy of the last-built bundle meshes (used for slice contour drawing).
    bundle_meshes_cpu: Vec<BundleMesh>,
    /// Pending background rebuild: receives (mesh, label) pairs when ready.
    bundle_mesh_pending: Option<std::sync::mpsc::Receiver<Vec<(BundleMesh, String)>>>,
    /// Instant at which the last rebuild was requested (for debounce).
    bundle_mesh_dirty_at: Option<std::time::Instant>,
}

impl TrxViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        trx_path: Option<String>,
        nifti_path: Option<String>,
    ) -> Self {
        let mut app = Self {
            trx_data: None,
            nifti_volume: None,
            camera_3d: OrbitCamera::new(Vec3::ZERO, 200.0),
            slice_cameras: [
                OrthoSliceCamera::new(SliceAxis::Axial, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Coronal, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Sagittal, Vec3::ZERO, 200.0),
            ],
            slice_indices: [0; 3],
            slices_dirty: false,
            has_streamlines: false,
            has_slices: false,
            trx_path: None,
            nifti_path: None,
            volume_center: Vec3::ZERO,
            volume_extent: 200.0,
            error_msg: None,
            gifti_surfaces: Vec::new(),
            surface_query_active: false,
            surface_query_surface: 0,
            surface_query_result: None,
            selection_revision: 0,
            surface_projection_cache: HashMap::new(),
            surface_projection_dirty: false,
            color_mode: ColorMode::DirectionRgb,
            colors_dirty: false,
            show_streamlines: true,
            group_visible: Vec::new(),
            indices_dirty: false,
            uniform_color: [1.0, 1.0, 1.0, 1.0],
            window_center: 0.5,
            window_width: 1.0,
            max_streamlines: 30_000,
            use_random_subset: false,
            streamline_order: Vec::new(),
            slab_half_width: 5.0,
            render_style: RenderStyle::Flat,
            tube_radius: 0.4,
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            sphere_query_active: false,
            sphere_center: Vec3::ZERO,
            sphere_radius: 10.0,
            sphere_query_result: None,
            show_bundle_mesh: false,
            bundle_mesh_source: BundleMeshSource::All,
            bundle_mesh_voxel_size: 2.0,
            bundle_mesh_threshold: 3.0,
            bundle_mesh_smooth: 1.5,
            bundle_mesh_opacity: 0.5,
            bundle_mesh_ambient: 0.35,
            bundle_meshes_cpu: Vec::new(),
            bundle_mesh_pending: None,
            bundle_mesh_dirty_at: None,
        };

        if let Some(render_state) = &cc.wgpu_render_state {
            if let Some(path) = trx_path {
                app.load_trx(&PathBuf::from(path), render_state);
            }
            if let Some(path) = nifti_path {
                app.load_nifti(&PathBuf::from(path), render_state);
            }
        }

        app
    }

    fn load_trx(&mut self, path: &PathBuf, rs: &egui_wgpu::RenderState) {
        match TrxGpuData::load(path) {
            Ok(data) => {
                self.volume_center = data.center();
                self.volume_extent = data.extent();
                self.camera_3d =
                    OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
                self.reset_slice_cameras();

                let resources =
                    StreamlineResources::new(&rs.device, rs.target_format, &data);
                {
                    let mut renderer = rs.renderer.write();
                    renderer.callback_resources.insert(resources);
                    if renderer.callback_resources.get::<MeshResources>().is_none() {
                        let mr = MeshResources::new(&rs.device, rs.target_format);
                        renderer.callback_resources.insert(mr);
                    }
                }

                // Initialise free-roam slice positions to the bounding-box centre.
                self.slice_world_offsets = [
                    self.volume_center.z,  // axial
                    self.volume_center.y,  // coronal
                    self.volume_center.x,  // sagittal
                ];
                self.has_streamlines = true;
                self.trx_path = Some(path.clone());
                self.group_visible = vec![true; data.groups.len()];
                self.color_mode = ColorMode::DirectionRgb;
                self.max_streamlines = data.nb_streamlines.min(30_000);
                self.streamline_order = (0..data.nb_streamlines as u32).collect();
                self.use_random_subset = false;
                self.sphere_query_active = false;
                self.sphere_query_result = None;
                self.surface_query_active = false;
                self.surface_query_result = None;
                self.selection_revision = self.selection_revision.wrapping_add(1);
                self.surface_projection_cache.clear();
                self.surface_projection_dirty = true;
                // Apply max streamline limit on initial load
                if data.nb_streamlines > self.max_streamlines {
                    self.indices_dirty = true;
                }
                self.trx_data = Some(data);
                self.error_msg = None;
                // Trigger bundle mesh rebuild if it was already shown.
                if self.show_bundle_mesh {
                    self.bundle_mesh_dirty_at = Some(std::time::Instant::now());
                }
            }
            Err(e) => {
                self.error_msg = Some(format!("Failed to load TRX: {e}"));
            }
        }
    }

    fn load_nifti(&mut self, path: &PathBuf, rs: &egui_wgpu::RenderState) {
        match NiftiVolume::load(path) {
            Ok(vol) => {
                self.slice_indices = [
                    vol.dims[2] / 2,
                    vol.dims[1] / 2,
                    vol.dims[0] / 2,
                ];

                if !self.has_streamlines {
                    self.volume_center = vol.voxel_to_world(Vec3::new(
                        vol.dims[0] as f32 / 2.0,
                        vol.dims[1] as f32 / 2.0,
                        vol.dims[2] as f32 / 2.0,
                    ));
                    self.volume_extent = (vol.voxel_to_world(Vec3::new(
                        vol.dims[0] as f32,
                        vol.dims[1] as f32,
                        vol.dims[2] as f32,
                    )) - vol.voxel_to_world(Vec3::ZERO))
                    .length();
                    self.camera_3d =
                        OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
                }
                self.reset_slice_cameras();

                let slice_resources =
                    SliceResources::new(&rs.device, &rs.queue, rs.target_format, &vol);
                slice_resources.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], &vol);
                slice_resources.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], &vol);
                slice_resources.update_slice(
                    &rs.queue,
                    SliceAxis::Sagittal,
                    self.slice_indices[2],
                    &vol,
                );

                rs.renderer.write().callback_resources.insert(slice_resources);

                self.has_slices = true;
                self.nifti_path = Some(path.clone());
                self.nifti_volume = Some(vol);
                self.slices_dirty = false;
                self.error_msg = None;
            }
            Err(e) => {
                self.error_msg = Some(format!("Failed to load NIfTI: {e}"));
            }
        }
    }

    fn load_gifti_surface(&mut self, path: &PathBuf, rs: &egui_wgpu::RenderState) {
        match GiftiSurfaceData::load(path) {
            Ok(surface) => {
                let mut renderer = rs.renderer.write();
                if renderer.callback_resources.get::<MeshResources>().is_none() {
                    renderer
                        .callback_resources
                        .insert(MeshResources::new(&rs.device, rs.target_format));
                }
                let mesh_resources = renderer
                    .callback_resources
                    .get_mut::<MeshResources>()
                    .expect("MeshResources inserted");
                let gpu_index = mesh_resources.add_surface(&rs.device, &surface);

                let palette: &[[f32; 3]] = &[
                    [0.94, 0.35, 0.35],
                    [0.35, 0.8, 0.95],
                    [0.4, 0.92, 0.45],
                    [0.98, 0.75, 0.35],
                    [0.85, 0.45, 0.95],
                    [0.95, 0.55, 0.2],
                ];
                let color = palette[self.gifti_surfaces.len() % palette.len()];
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "surface.gii".to_string());
                self.gifti_surfaces.push(LoadedGiftiSurface {
                    name,
                    path: path.clone(),
                    data: surface,
                    gpu_index,
                    visible: true,
                    opacity: 0.7,
                    color,
                    show_projection_map: false,
                    map_opacity: 1.0,
                    map_threshold: 0.0,
                    surface_ambient: 0.42,
                    surface_gloss: 0.45,
                    projection_mode: SurfaceProjectionMode::Density,
                    projection_dps: None,
                    projection_depth_mm: 2.0,
                    projection_colormap: SurfaceColormap::Inferno,
                    auto_range: true,
                    range_min: 0.0,
                    range_max: 1.0,
                });
                self.surface_projection_dirty = true;
                self.error_msg = None;
            }
            Err(e) => {
                self.error_msg = Some(format!("Failed to load GIFTI surface: {e}"));
            }
        }
    }

    fn reset_slice_cameras(&mut self) {
        let half_extents = self
            .nifti_volume
            .as_ref()
            .map(|v| v.slice_half_extents())
            .unwrap_or([self.volume_extent * 0.5; 3]);
        self.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, self.volume_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Coronal, self.volume_center, half_extents[1] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Sagittal, self.volume_center, half_extents[2] * 2.0),
        ];
    }

    /// Draw crosshair lines on a 2D slice view showing the other two slice positions.
    fn draw_crosshairs(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        // Get the world-space positions of the other two slices
        let (other_a, other_b) = match axis_index {
            // Axial view: show coronal (Y) and sagittal (X) positions
            0 => (self.slice_world_position(2), self.slice_world_position(1)),
            // Coronal view: show sagittal (X) and axial (Z) positions
            1 => (self.slice_world_position(2), self.slice_world_position(0)),
            // Sagittal view: show coronal (Y) and axial (Z) positions
            _ => (self.slice_world_position(1), self.slice_world_position(0)),
        };

        let slice_pos = self.slice_world_position(axis_index);

        // Create world-space points on the crosshair lines and project them
        // For each crosshair line, we create two points at the extremes of the view
        let far = 10000.0;
        let (h_p1, h_p2, v_p1, v_p2) = match axis_index {
            0 => {
                // Axial: horizontal = coronal(Y), vertical = sagittal(X)
                let y = other_b; // coronal Y position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, y, slice_pos),
                    glam::Vec3::new(far, y, slice_pos),
                    glam::Vec3::new(x, -far, slice_pos),
                    glam::Vec3::new(x, far, slice_pos),
                )
            }
            1 => {
                // Coronal: horizontal = axial(Z), vertical = sagittal(X)
                let z = other_b; // axial Z position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, slice_pos, z),
                    glam::Vec3::new(far, slice_pos, z),
                    glam::Vec3::new(x, slice_pos, -far),
                    glam::Vec3::new(x, slice_pos, far),
                )
            }
            _ => {
                // Sagittal: horizontal = axial(Z), vertical = coronal(Y)
                let z = other_b; // axial Z position
                let y = other_a; // coronal Y position
                (
                    glam::Vec3::new(slice_pos, -far, z),
                    glam::Vec3::new(slice_pos, far, z),
                    glam::Vec3::new(slice_pos, y, -far),
                    glam::Vec3::new(slice_pos, y, far),
                )
            }
        };

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // NDC [-1,1] → screen rect
            let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
            let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(); // flip Y
            egui::pos2(sx, sy)
        };

        let crosshair_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 128);
        let stroke = egui::Stroke::new(1.0, crosshair_color);
        let painter = ui.painter_at(rect);

        // Horizontal line (clipped to rect)
        painter.line_segment([project(h_p1), project(h_p2)], stroke);
        // Vertical line (clipped to rect)
        painter.line_segment([project(v_p1), project(v_p2)], stroke);
    }

    fn slice_world_position(&self, axis_index: usize) -> f32 {
        if let Some(vol) = &self.nifti_volume {
            let idx = self.slice_indices[axis_index] as f32;
            let world = match axis_index {
                0 => vol.voxel_to_world(Vec3::new(0.0, 0.0, idx)),
                1 => vol.voxel_to_world(Vec3::new(0.0, idx, 0.0)),
                2 => vol.voxel_to_world(Vec3::new(idx, 0.0, 0.0)),
                _ => Vec3::ZERO,
            };
            match axis_index {
                0 => world.z,
                1 => world.y,
                2 => world.x,
                _ => 0.0,
            }
        } else {
            self.slice_world_offsets[axis_index]
        }
    }

    fn recompute_surface_query(&mut self) {
        if !self.surface_query_active {
            self.surface_query_result = None;
            self.indices_dirty = true;
            return;
        }
        let Some(data) = &self.trx_data else {
            self.surface_query_result = None;
            self.indices_dirty = true;
            return;
        };
        if self.surface_query_surface >= self.gifti_surfaces.len() {
            self.surface_query_result = None;
            self.indices_dirty = true;
            return;
        }
        let surf = &self.gifti_surfaces[self.surface_query_surface];
        let depth = surf.projection_depth_mm.max(0.0);
        self.surface_query_result = Some(data.query_near_surface(&surf.data, depth));
        self.indices_dirty = true;
    }

    fn refresh_surface_projections(&mut self, frame: &mut eframe::Frame) {
        let Some(data) = &self.trx_data else {
            return;
        };
        if self.gifti_surfaces.is_empty() {
            return;
        }
        let selected = data.filtered_streamline_indices(
            &self.group_visible,
            self.max_streamlines,
            &self.streamline_order,
            self.sphere_query_result.as_ref(),
            self.surface_query_result.as_ref(),
        );

        let mut upload_scalars: Vec<(usize, Vec<f32>)> = Vec::new();

        for (surface_idx, surface) in self.gifti_surfaces.iter_mut().enumerate() {
            let key = SurfaceProjectionCacheKey {
                surface_idx,
                selection_revision: self.selection_revision,
                depth_bin: (surface.projection_depth_mm * 100.0).round() as i32,
                mode: surface.projection_mode,
                dps_name: surface.projection_dps.clone(),
            };

            if !self.surface_projection_cache.contains_key(&key) {
                let dps_values = surface
                    .projection_dps
                    .as_ref()
                    .and_then(|name| data.dps_data.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice()));
                let (density, mean_dps) = data.project_selected_to_surface(
                    &surface.data,
                    &selected,
                    surface.projection_depth_mm.max(0.0),
                    dps_values,
                );
                let active: Vec<f32> = match surface.projection_mode {
                    SurfaceProjectionMode::Density => density.iter().copied().filter(|v| v.is_finite()).collect(),
                    SurfaceProjectionMode::MeanDps => mean_dps.iter().copied().filter(|v| v.is_finite()).collect(),
                };
                let (data_min, data_max) = robust_range(&active);
                self.surface_projection_cache.insert(
                    key.clone(),
                    SurfaceProjectionCacheValue {
                        density,
                        mean_dps,
                        data_min,
                        data_max,
                    },
                );
            }

            if let Some(cached) = self.surface_projection_cache.get(&key) {
                let active_values = match surface.projection_mode {
                    SurfaceProjectionMode::Density => &cached.density,
                    SurfaceProjectionMode::MeanDps => &cached.mean_dps,
                };
                if surface.auto_range {
                    surface.range_min = cached.data_min;
                    surface.range_max = cached.data_max;
                }
                let fill_value = surface.range_min;
                let scalars: Vec<f32> = active_values
                    .iter()
                    .map(|v| if v.is_finite() { *v } else { fill_value })
                    .collect();
                upload_scalars.push((surface.gpu_index, scalars));
            }
        }

        if let Some(rs) = frame.wgpu_render_state() {
            let renderer = rs.renderer.read();
            if let Some(mr) = renderer.callback_resources.get::<MeshResources>() {
                for (gpu_index, scalars) in upload_scalars {
                    mr.update_surface_scalars(&rs.queue, gpu_index, &scalars);
                }
            }
        }
    }

    fn draw_mesh_intersections(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        if self.gifti_surfaces.is_empty() && (self.bundle_meshes_cpu.is_empty() || !self.show_bundle_mesh) {
            return;
        }
        let painter = ui.painter_at(rect);
        let eps = 1e-4f32;

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        for surface in &self.gifti_surfaces {
            if !surface.visible || surface.opacity <= 0.01 {
                continue;
            }

            // Surface-level early out by axis-aligned bounds.
            let (smin, smax) = match axis_index {
                0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
                1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
                _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
            };
            if slice_pos < smin - eps || slice_pos > smax + eps {
                continue;
            }

            let color = egui::Color32::from_rgba_unmultiplied(
                (surface.color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.color[2].clamp(0.0, 1.0) * 255.0) as u8,
                (surface.opacity.clamp(0.0, 1.0) * 255.0) as u8,
            );
            let stroke = egui::Stroke::new(1.25, color);

            for tri in surface.data.indices.chunks_exact(3) {
                let ia = tri[0] as usize;
                let ib = tri[1] as usize;
                let ic = tri[2] as usize;
                let a = glam::Vec3::from(surface.data.vertices[ia]);
                let b = glam::Vec3::from(surface.data.vertices[ib]);
                let c = glam::Vec3::from(surface.data.vertices[ic]);

                let tmin = tri_axis_value(a, axis_index)
                    .min(tri_axis_value(b, axis_index))
                    .min(tri_axis_value(c, axis_index));
                let tmax = tri_axis_value(a, axis_index)
                    .max(tri_axis_value(b, axis_index))
                    .max(tri_axis_value(c, axis_index));
                if slice_pos < tmin - eps || slice_pos > tmax + eps {
                    continue;
                }

                let mut pts = Vec::with_capacity(3);
                for (p0, p1) in [(a, b), (b, c), (c, a)] {
                    if let Some(p) = intersect_edge_with_slice(p0, p1, axis_index, slice_pos, eps) {
                        if !pts.iter().any(|q: &glam::Vec3| (*q - p).length_squared() <= eps * eps) {
                            pts.push(p);
                        }
                    }
                }
                if pts.len() < 2 {
                    continue;
                }
                // For rare 3-point cases (vertex on plane), keep the longest segment.
                let (p0, p1) = if pts.len() == 2 {
                    (pts[0], pts[1])
                } else {
                    let mut best = (pts[0], pts[1]);
                    let mut best_d2 = (pts[1] - pts[0]).length_squared();
                    for i in 0..pts.len() {
                        for j in (i + 1)..pts.len() {
                            let d2 = (pts[j] - pts[i]).length_squared();
                            if d2 > best_d2 {
                                best = (pts[i], pts[j]);
                                best_d2 = d2;
                            }
                        }
                    }
                    best
                };

                painter.line_segment([project(p0), project(p1)], stroke);
            }
        }

        // ── Bundle mesh contours ─────────────────────────────────────────────
        if self.show_bundle_mesh {
            for mesh in &self.bundle_meshes_cpu {
                for tri in mesh.indices.chunks_exact(3) {
                    let va = &mesh.vertices[tri[0] as usize];
                    let vb = &mesh.vertices[tri[1] as usize];
                    let vc = &mesh.vertices[tri[2] as usize];
                    let a  = glam::Vec3::from(va.position);
                    let b  = glam::Vec3::from(vb.position);
                    let c  = glam::Vec3::from(vc.position);

                    let tmin = tri_axis_value(a, axis_index)
                        .min(tri_axis_value(b, axis_index))
                        .min(tri_axis_value(c, axis_index));
                    let tmax = tri_axis_value(a, axis_index)
                        .max(tri_axis_value(b, axis_index))
                        .max(tri_axis_value(c, axis_index));
                    if slice_pos < tmin - eps || slice_pos > tmax + eps {
                        continue;
                    }

                    // Find intersections, interpolating color along each edge.
                    let mut pts: Vec<(glam::Vec3, [f32; 4])> = Vec::with_capacity(2);
                    for (p0, c0, p1, c1) in [
                        (a, va.color, b, vb.color),
                        (b, vb.color, c, vc.color),
                        (c, vc.color, a, va.color),
                    ] {
                        let d0 = tri_axis_value(p0, axis_index) - slice_pos;
                        let d1 = tri_axis_value(p1, axis_index) - slice_pos;
                        if d0.abs() <= eps && d1.abs() <= eps { continue; }
                        let t = if d0.abs() <= eps { 0.0 }
                                else if d1.abs() <= eps { 1.0 }
                                else if d0 * d1 < 0.0 { d0 / (d0 - d1) }
                                else { continue };
                        let pos = p0 + (p1 - p0) * t;
                        let col = [
                            c0[0] + (c1[0] - c0[0]) * t,
                            c0[1] + (c1[1] - c0[1]) * t,
                            c0[2] + (c1[2] - c0[2]) * t,
                            c0[3] + (c1[3] - c0[3]) * t,
                        ];
                        if !pts.iter().any(|(q, _)| (*q - pos).length_squared() <= eps * eps) {
                            pts.push((pos, col));
                        }
                    }
                    if pts.len() < 2 { continue; }
                    let (p0, col0, p1, col1) = if pts.len() == 2 {
                        (pts[0].0, pts[0].1, pts[1].0, pts[1].1)
                    } else {
                        let mut best = (pts[0].0, pts[0].1, pts[1].0, pts[1].1);
                        let mut best_d2 = (pts[1].0 - pts[0].0).length_squared();
                        for i in 0..pts.len() {
                            for j in (i + 1)..pts.len() {
                                let d2 = (pts[j].0 - pts[i].0).length_squared();
                                if d2 > best_d2 { best = (pts[i].0, pts[i].1, pts[j].0, pts[j].1); best_d2 = d2; }
                            }
                        }
                        best
                    };

                    let opacity_u8 = (self.bundle_mesh_opacity.clamp(0.0, 1.0) * 255.0) as u8;
                    let to_color = |c: [f32; 4]| egui::Color32::from_rgba_unmultiplied(
                        (c[0].clamp(0.0, 1.0) * 255.0) as u8,
                        (c[1].clamp(0.0, 1.0) * 255.0) as u8,
                        (c[2].clamp(0.0, 1.0) * 255.0) as u8,
                        opacity_u8,
                    );

                    // Draw with gradient by blending colors at the two endpoints.
                    let mid = (p0 + p1) * 0.5;
                    let mid_col = [
                        (col0[0] + col1[0]) * 0.5,
                        (col0[1] + col1[1]) * 0.5,
                        (col0[2] + col1[2]) * 0.5,
                        (col0[3] + col1[3]) * 0.5,
                    ];
                    painter.line_segment(
                        [project(p0), project(mid)],
                        egui::Stroke::new(1.5, to_color(col0)),
                    );
                    painter.line_segment(
                        [project(mid), project(p1)],
                        egui::Stroke::new(1.5, to_color(mid_col)),
                    );
                    let _ = (col1, mid_col);
                }
            }
        }
    }
}

fn tri_axis_value(p: glam::Vec3, axis_index: usize) -> f32 {
    match axis_index {
        0 => p.z,
        1 => p.y,
        _ => p.x,
    }
}

/// Returns `Some((min, max))` when the color mode is scalar and auto-range is off,
/// otherwise `None` so `recolor` will auto-detect the range from the data.
fn scalar_range_opt(
    mode: &ColorMode,
    auto: bool,
    min: f32,
    max: f32,
) -> Option<(f32, f32)> {
    if auto { return None; }
    match mode {
        ColorMode::Dpv(_) | ColorMode::Dps(_) => Some((min, max)),
        _ => None,
    }
}

fn robust_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo_idx = ((n as f32) * 0.02).floor() as usize;
    let hi_idx = ((n as f32) * 0.98).floor() as usize;
    let lo = finite[lo_idx.min(n - 1)];
    let hi = finite[hi_idx.min(n - 1)].max(lo + 1e-6);
    (lo, hi)
}

fn intersect_edge_with_slice(
    p0: glam::Vec3,
    p1: glam::Vec3,
    axis_index: usize,
    slice_pos: f32,
    eps: f32,
) -> Option<glam::Vec3> {
    let c0 = tri_axis_value(p0, axis_index);
    let c1 = tri_axis_value(p1, axis_index);
    let d0 = c0 - slice_pos;
    let d1 = c1 - slice_pos;

    // Coplanar edge: skip to avoid degenerate full-triangle artifacts.
    if d0.abs() <= eps && d1.abs() <= eps {
        return None;
    }
    if d0.abs() <= eps {
        return Some(p0);
    }
    if d1.abs() <= eps {
        return Some(p1);
    }
    if d0 * d1 > 0.0 {
        return None;
    }
    let t = d0 / (d0 - d1);
    Some(p0 + (p1 - p0) * t)
}

impl eframe::App for TrxViewerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Update slice positions if dirty
        if self.slices_dirty {
            if let (Some(vol), Some(rs)) = (&self.nifti_volume, frame.wgpu_render_state()) {
                let renderer = rs.renderer.read();
                if let Some(sr) = renderer.callback_resources.get::<SliceResources>() {
                    sr.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], vol);
                    sr.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], vol);
                    sr.update_slice(&rs.queue, SliceAxis::Sagittal, self.slice_indices[2], vol);
                }
            }
            self.slices_dirty = false;
        }

        // Update vertex colors if dirty
        if self.colors_dirty {
            if let (Some(data), Some(rs)) = (&self.trx_data, frame.wgpu_render_state()) {
                let renderer = rs.renderer.read();
                if let Some(sr) = renderer.callback_resources.get::<StreamlineResources>() {
                    sr.update_colors(&rs.queue, &data.colors);
                }
            }
            // Tube vertices embed colors — rebuild geometry so the new colors take effect.
            if self.render_style == RenderStyle::Tubes {
                self.indices_dirty = true;
            }
            // Bundle mesh vertex colors come from the color buffer; trigger rebuild.
            if self.show_bundle_mesh {
                self.bundle_mesh_dirty_at = Some(std::time::Instant::now());
            }
            self.colors_dirty = false;
        }

        // Update index buffer (and tube geometry) if any filter changed
        if self.indices_dirty {
            if let (Some(data), Some(rs)) = (&self.trx_data, frame.wgpu_render_state()) {
                if self.render_style == RenderStyle::Tubes {
                    let selected = data.filtered_streamline_indices(
                        &self.group_visible,
                        self.max_streamlines,
                        &self.streamline_order,
                        self.sphere_query_result.as_ref(),
                        self.surface_query_result.as_ref(),
                    );
                    let (tube_verts, tube_indices) = data.build_tube_vertices(&selected);
                    let mut renderer = rs.renderer.write();
                    if let Some(sr) = renderer.callback_resources.get_mut::<StreamlineResources>() {
                        sr.update_tube_geometry(&rs.device, &tube_verts, &tube_indices);
                        // Keep line indices up to date too (used when switching back)
                        let line_indices = data.build_index_buffer(
                            &self.group_visible, self.max_streamlines,
                            &self.streamline_order,
                            self.sphere_query_result.as_ref(),
                            self.surface_query_result.as_ref(),
                        );
                        sr.update_indices(&rs.device, &rs.queue, &line_indices);
                    }
                } else {
                    let indices = data.build_index_buffer(
                        &self.group_visible,
                        self.max_streamlines,
                        &self.streamline_order,
                        self.sphere_query_result.as_ref(),
                        self.surface_query_result.as_ref(),
                    );
                    let mut renderer = rs.renderer.write();
                    if let Some(sr) = renderer.callback_resources.get_mut::<StreamlineResources>() {
                        sr.update_indices(&rs.device, &rs.queue, &indices);
                    }
                }
            }
            self.indices_dirty = false;
            self.selection_revision = self.selection_revision.wrapping_add(1);
            self.surface_projection_dirty = true;
            // Bundle mesh rebuilds when selection changes (not needed for All source).
            if self.show_bundle_mesh
                && self.bundle_mesh_source != BundleMeshSource::All
            {
                self.bundle_mesh_dirty_at = Some(std::time::Instant::now());
            }
        }

        // ── Bundle mesh: check debounce + receive completed mesh ──────────────
        if let Some(t) = self.bundle_mesh_dirty_at {
            if t.elapsed() >= std::time::Duration::from_millis(150) {
                self.bundle_mesh_dirty_at = None;
                if self.show_bundle_mesh {
                    if let Some(data) = &self.trx_data {
                        let voxel_size = self.bundle_mesh_voxel_size;
                        let threshold  = self.bundle_mesh_threshold;
                        let smooth     = self.bundle_mesh_smooth;
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.bundle_mesh_pending = Some(rx);
                        let egui_ctx = ctx.clone();

                        match self.bundle_mesh_source {
                            BundleMeshSource::All => {
                                // All vertex positions + current colors.
                                let positions = data.positions.clone();
                                let colors    = data.colors.clone();
                                std::thread::spawn(move || {
                                    let mut out = Vec::new();
                                    if let Some(m) = build_bundle_mesh(&positions, &colors, voxel_size, threshold, smooth) {
                                        out.push((m, "all".to_string()));
                                    }
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                            BundleMeshSource::Selection => {
                                let selected = data.filtered_streamline_indices(
                                    &self.group_visible,
                                    self.max_streamlines,
                                    &self.streamline_order,
                                    self.sphere_query_result.as_ref(),
                                    self.surface_query_result.as_ref(),
                                );
                                let (positions, colors) = data.selected_vertex_data(&selected);
                                std::thread::spawn(move || {
                                    let mut out = Vec::new();
                                    if let Some(m) = build_bundle_mesh(&positions, &colors, voxel_size, threshold, smooth) {
                                        out.push((m, "selection".to_string()));
                                    }
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                            BundleMeshSource::PerGroup => {
                                // One mesh per group, using only visible groups.
                                let group_data: Vec<(String, Vec<[f32;3]>, Vec<[f32;4]>)> =
                                    data.groups.iter()
                                        .enumerate()
                                        .filter(|(i, _)| self.group_visible.get(*i).copied().unwrap_or(true))
                                        .map(|(_, (name, members))| {
                                            let (pos, col) = data.selected_vertex_data(members);
                                            (name.clone(), pos, col)
                                        })
                                        .collect();
                                std::thread::spawn(move || {
                                    let out: Vec<(BundleMesh, String)> = group_data
                                        .into_iter()
                                        .filter_map(|(name, pos, col)| {
                                            build_bundle_mesh(&pos, &col, voxel_size, threshold, smooth)
                                                .map(|m| (m, name))
                                        })
                                        .collect();
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                        }
                    }
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        if let Some(rx) = &self.bundle_mesh_pending {
            if let Ok(meshes) = rx.try_recv() {
                self.bundle_mesh_pending = None;
                if let Some(rs) = frame.wgpu_render_state() {
                    let mut renderer = rs.renderer.write();
                    if let Some(mr) = renderer.callback_resources.get_mut::<MeshResources>() {
                        if meshes.is_empty() {
                            mr.clear_bundle_mesh();
                            self.bundle_meshes_cpu.clear();
                        } else {
                            mr.set_bundle_meshes(&rs.device, &meshes);
                            self.bundle_meshes_cpu = meshes.into_iter().map(|(m, _)| m).collect();
                        }
                    }
                }
            }
        }

        if self.surface_projection_dirty {
            self.refresh_surface_projections(frame);
            self.surface_projection_dirty = false;
        }

        // ── Sidebar ──
        egui::SidePanel::left("sidebar")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("TRX Viewer");
                ui.separator();

                // File loading
                ui.label("Files");
                if ui.button("Open TRX...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("TRX files", &["trx"])
                        .pick_file()
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_trx(&path, rs);
                        }
                    }
                }
                if let Some(ref p) = self.trx_path {
                    ui.small(
                        p.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    );
                }

                if ui.button("Open NIfTI...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                        .pick_file()
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_nifti(&path, rs);
                        }
                    }
                }
                if let Some(ref p) = self.nifti_path {
                    ui.small(
                        p.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    );
                }

                if ui.button("Open GIFTI Surface...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("GIFTI files", &["gii", "gifti"])
                        .pick_file()
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_gifti_surface(&path, rs);
                        }
                    }
                }

                // Error display
                if let Some(ref msg) = self.error_msg {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, msg);
                }

                ui.separator();

                // TRX info
                if let Some(ref data) = self.trx_data {
                    ui.label("TRX Info");
                    ui.small(format!("Streamlines: {}", data.nb_streamlines));
                    ui.small(format!("Vertices: {}", data.nb_vertices));
                    ui.checkbox(&mut self.show_streamlines, "Show streamlines");
                    ui.separator();
                }

                // Volume info
                if let Some(ref vol) = self.nifti_volume {
                    ui.label("Volume Info");
                    ui.small(format!(
                        "Dimensions: {}x{}x{}",
                        vol.dims[0], vol.dims[1], vol.dims[2]
                    ));
                    ui.separator();
                }

                if !self.gifti_surfaces.is_empty() {
                    ui.label("GIFTI Surfaces");
                    let dps_names_all = self
                        .trx_data
                        .as_ref()
                        .map(|d| d.dps_names.clone())
                        .unwrap_or_default();
                    let mut query_changed = false;
                    let mut projection_changed = false;

                    ui.group(|ui| {
                        ui.checkbox(&mut self.surface_query_active, "Use surface depth filter");
                        ui.horizontal(|ui| {
                            ui.label("Filter surface");
                            let current = self
                                .gifti_surfaces
                                .get(self.surface_query_surface)
                                .map(|s| s.name.clone())
                                .unwrap_or_else(|| "none".to_string());
                            egui::ComboBox::from_id_salt("surface_query_surface")
                                .selected_text(current)
                                .show_ui(ui, |ui| {
                                    for (i, s) in self.gifti_surfaces.iter().enumerate() {
                                        if ui
                                            .selectable_value(&mut self.surface_query_surface, i, &s.name)
                                            .changed()
                                        {
                                            query_changed = true;
                                        }
                                    }
                                });
                        });
                        if self.surface_query_surface < self.gifti_surfaces.len() {
                            let depth = &mut self.gifti_surfaces[self.surface_query_surface].projection_depth_mm;
                            if ui
                                .add(egui::Slider::new(depth, 0.1..=20.0).text("Depth mm"))
                                .changed()
                            {
                                query_changed = true;
                                projection_changed = true;
                            }
                        }
                    });

                    for (surface_idx, surface) in self.gifti_surfaces.iter_mut().enumerate() {
                        ui.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut surface.visible, "");
                                ui.label(&surface.name);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Opacity");
                                ui.add(egui::Slider::new(&mut surface.opacity, 0.0..=1.0));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Color");
                                ui.color_edit_button_rgb(&mut surface.color);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Projection");
                                if ui
                                    .checkbox(&mut surface.show_projection_map, "Show map")
                                    .changed()
                                {
                                    projection_changed = true;
                                }
                                egui::ComboBox::from_id_salt(format!("proj_mode_{surface_idx}"))
                                    .selected_text(match surface.projection_mode {
                                        SurfaceProjectionMode::Density => "Density",
                                        SurfaceProjectionMode::MeanDps => "Mean DPS",
                                    })
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_value(
                                                &mut surface.projection_mode,
                                                SurfaceProjectionMode::Density,
                                                "Density",
                                            )
                                            .changed()
                                        {
                                            projection_changed = true;
                                        }
                                        if ui
                                            .selectable_value(
                                                &mut surface.projection_mode,
                                                SurfaceProjectionMode::MeanDps,
                                                "Mean DPS",
                                            )
                                            .changed()
                                        {
                                            projection_changed = true;
                                        }
                                    });
                            });
                            if matches!(surface.projection_mode, SurfaceProjectionMode::MeanDps) {
                                let current = surface
                                    .projection_dps
                                    .clone()
                                    .unwrap_or_else(|| "Select DPS".to_string());
                                ui.horizontal(|ui| {
                                    ui.label("DPS");
                                    egui::ComboBox::from_id_salt(format!("proj_dps_{surface_idx}"))
                                        .selected_text(current)
                                        .show_ui(ui, |ui| {
                                            for name in &dps_names_all {
                                                if ui
                                                    .selectable_label(
                                                        surface.projection_dps.as_ref() == Some(name),
                                                        name,
                                                    )
                                                    .clicked()
                                                {
                                                    surface.projection_dps = Some(name.clone());
                                                    projection_changed = true;
                                                }
                                            }
                                        });
                                });
                            }
                            ui.horizontal(|ui| {
                                ui.label("Colormap");
                                egui::ComboBox::from_id_salt(format!("proj_cmap_{surface_idx}"))
                                    .selected_text(match surface.projection_colormap {
                                        SurfaceColormap::BlueWhiteRed => "Blue-White-Red",
                                        SurfaceColormap::Viridis => "Viridis",
                                        SurfaceColormap::Inferno => "Inferno",
                                    })
                                    .show_ui(ui, |ui| {
                                        if ui
                                            .selectable_value(
                                                &mut surface.projection_colormap,
                                                SurfaceColormap::BlueWhiteRed,
                                                "Blue-White-Red",
                                            )
                                            .changed()
                                        {
                                            projection_changed = true;
                                        }
                                        if ui
                                            .selectable_value(
                                                &mut surface.projection_colormap,
                                                SurfaceColormap::Viridis,
                                                "Viridis",
                                            )
                                            .changed()
                                        {
                                            projection_changed = true;
                                        }
                                        if ui
                                            .selectable_value(
                                                &mut surface.projection_colormap,
                                                SurfaceColormap::Inferno,
                                                "Inferno",
                                            )
                                            .changed()
                                        {
                                            projection_changed = true;
                                        }
                                    });
                            });
                            if surface.show_projection_map {
                                ui.horizontal(|ui| {
                                    ui.label("Map opacity");
                                    if ui
                                        .add(egui::Slider::new(&mut surface.map_opacity, 0.0..=1.0))
                                        .changed()
                                    {
                                        projection_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Map threshold");
                                    if ui
                                        .add(egui::Slider::new(&mut surface.map_threshold, 0.0..=1.0))
                                        .changed()
                                    {
                                        projection_changed = true;
                                    }
                                });
                            }
                            ui.horizontal(|ui| {
                                ui.label("Ambient");
                                if ui
                                    .add(egui::Slider::new(&mut surface.surface_ambient, 0.0..=1.0))
                                    .changed()
                                {
                                    projection_changed = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                ui.label("Gloss");
                                if ui
                                    .add(egui::Slider::new(&mut surface.surface_gloss, 0.0..=1.0))
                                    .changed()
                                {
                                    projection_changed = true;
                                }
                            });
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut surface.auto_range, "Auto range").changed() {
                                    projection_changed = true;
                                }
                                if ui.button("Recompute").clicked() {
                                    projection_changed = true;
                                }
                            });
                            if !surface.auto_range {
                                ui.horizontal(|ui| {
                                    ui.label("Min");
                                    if ui
                                        .add(egui::DragValue::new(&mut surface.range_min).speed(0.01))
                                        .changed()
                                    {
                                        projection_changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Max");
                                    if ui
                                        .add(egui::DragValue::new(&mut surface.range_max).speed(0.01))
                                        .changed()
                                    {
                                        projection_changed = true;
                                    }
                                });
                            }
                            ui.small(
                                surface
                                    .path
                                    .file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_default(),
                            );
                        });
                    }
                    if query_changed {
                        self.recompute_surface_query();
                    }
                    if projection_changed {
                        self.surface_projection_dirty = true;
                    }
                    ui.separator();
                }

                // ── Coloring controls ──
                if self.trx_data.is_some() {
                    ui.label("Coloring");

                    // Collect available mode labels
                    let mut mode_labels = vec!["Direction RGB".to_string()];
                    let dpv_names: Vec<String> = self
                        .trx_data
                        .as_ref()
                        .map(|d| d.dpv_names.clone())
                        .unwrap_or_default();
                    let dps_names: Vec<String> = self
                        .trx_data
                        .as_ref()
                        .map(|d| d.dps_names.clone())
                        .unwrap_or_default();
                    let has_groups = self
                        .trx_data
                        .as_ref()
                        .map(|d| !d.groups.is_empty())
                        .unwrap_or(false);

                    for name in &dpv_names {
                        mode_labels.push(format!("DPV: {name}"));
                    }
                    for name in &dps_names {
                        mode_labels.push(format!("DPS: {name}"));
                    }
                    if has_groups {
                        mode_labels.push("Group".to_string());
                    }
                    mode_labels.push("Uniform".to_string());

                    // Find current selection index
                    let current_idx = match &self.color_mode {
                        ColorMode::DirectionRgb => 0,
                        ColorMode::Dpv(name) => {
                            1 + dpv_names.iter().position(|n| n == name).unwrap_or(0)
                        }
                        ColorMode::Dps(name) => {
                            1 + dpv_names.len()
                                + dps_names.iter().position(|n| n == name).unwrap_or(0)
                        }
                        ColorMode::Group => 1 + dpv_names.len() + dps_names.len(),
                        ColorMode::Uniform(_) => mode_labels.len() - 1,
                    };

                    let mut selected = current_idx;
                    egui::ComboBox::from_label("")
                        .selected_text(&mode_labels[current_idx])
                        .show_ui(ui, |ui| {
                            for (i, label) in mode_labels.iter().enumerate() {
                                ui.selectable_value(&mut selected, i, label);
                            }
                        });

                    if selected != current_idx {
                        let new_mode = if selected == 0 {
                            ColorMode::DirectionRgb
                        } else if selected <= dpv_names.len() {
                            ColorMode::Dpv(dpv_names[selected - 1].clone())
                        } else if selected <= dpv_names.len() + dps_names.len() {
                            ColorMode::Dps(
                                dps_names[selected - 1 - dpv_names.len()].clone(),
                            )
                        } else if has_groups
                            && selected == 1 + dpv_names.len() + dps_names.len()
                        {
                            ColorMode::Group
                        } else {
                            ColorMode::Uniform(self.uniform_color)
                        };

                        self.color_mode = new_mode.clone();
                        if let Some(data) = &mut self.trx_data {
                            // For scalar modes, set up auto range from data
                            if self.scalar_auto_range {
                                if let Some((lo, hi)) = data.scalar_range_for_mode(&new_mode) {
                                    self.scalar_range_min = lo;
                                    self.scalar_range_max = hi;
                                }
                            }
                            let range = scalar_range_opt(
                                &new_mode, self.scalar_auto_range,
                                self.scalar_range_min, self.scalar_range_max,
                            );
                            data.recolor(&new_mode, range);
                        }
                        self.colors_dirty = true;
                    }

                    // Uniform color picker
                    if matches!(self.color_mode, ColorMode::Uniform(_)) {
                        let mut c = self.uniform_color;
                        if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                            self.uniform_color = c;
                            self.color_mode = ColorMode::Uniform(c);
                            if let Some(data) = &mut self.trx_data {
                                data.recolor(&ColorMode::Uniform(c), None);
                            }
                            self.colors_dirty = true;
                        }
                    }

                    // ── Colorbar (DPV / DPS only) ──────────────────────────
                    let is_scalar = matches!(self.color_mode, ColorMode::Dpv(_) | ColorMode::Dps(_));
                    if is_scalar {
                        ui.add_space(4.0);

                        // Draw gradient strip
                        let bar_w = ui.available_width();
                        let bar_h = 14.0;
                        let (bar_rect, _) = ui.allocate_exact_size(
                            egui::vec2(bar_w, bar_h), egui::Sense::hover(),
                        );
                        let painter = ui.painter_at(bar_rect);
                        let n = 64usize;
                        let sw = bar_w / n as f32;
                        for i in 0..n {
                            let t = i as f32 / (n - 1) as f32;
                            let [r, g, b, _] = colormap_bwr(t);
                            let col = egui::Color32::from_rgb(
                                (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                            );
                            painter.rect_filled(
                                egui::Rect::from_min_size(
                                    egui::pos2(bar_rect.left() + i as f32 * sw, bar_rect.top()),
                                    egui::vec2(sw + 1.0, bar_h),
                                ),
                                0.0, col,
                            );
                        }

                        // Min / max labels and editable range
                        let mut range_changed = false;
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut self.scalar_auto_range, "Auto").changed()
                                && self.scalar_auto_range
                            {
                                // Re-compute range from data when re-enabling auto
                                if let Some(data) = &self.trx_data {
                                    if let Some((lo, hi)) = data.scalar_range_for_mode(&self.color_mode) {
                                        self.scalar_range_min = lo;
                                        self.scalar_range_max = hi;
                                        range_changed = true;
                                    }
                                }
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Min");
                            let resp = ui.add_enabled(
                                !self.scalar_auto_range,
                                egui::DragValue::new(&mut self.scalar_range_min).speed(0.01),
                            );
                            if resp.changed() { range_changed = true; }
                            ui.label("Max");
                            let resp = ui.add_enabled(
                                !self.scalar_auto_range,
                                egui::DragValue::new(&mut self.scalar_range_max).speed(0.01),
                            );
                            if resp.changed() { range_changed = true; }
                        });

                        if range_changed {
                            let range = Some((self.scalar_range_min, self.scalar_range_max));
                            let mode = self.color_mode.clone();
                            if let Some(data) = &mut self.trx_data {
                                data.recolor(&mode, range);
                            }
                            self.colors_dirty = true;
                        }
                    }

                    ui.separator();
                }

                // ── Render style ──
                if self.has_streamlines {
                    ui.label("Render Style");
                    let styles = [
                        (RenderStyle::Flat,        "Flat lines"),
                        (RenderStyle::Illuminated, "Illuminated"),
                        (RenderStyle::Tubes,       "Tube impostors"),
                        (RenderStyle::DepthCue,    "Depth cue"),
                    ];
                    let current_label = styles.iter()
                        .find(|(s, _)| *s == self.render_style)
                        .map(|(_, l)| *l)
                        .unwrap_or("Flat lines");

                    egui::ComboBox::from_id_salt("render_style")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for (style, label) in &styles {
                                if ui.selectable_value(&mut self.render_style, *style, *label).changed() {
                                    // Switching to/from Tubes requires geometry rebuild
                                    self.indices_dirty = true;
                                }
                            }
                        });

                    if self.render_style == RenderStyle::Tubes {
                        ui.horizontal(|ui| {
                            ui.label("Radius (mm):");
                            if ui.add(egui::Slider::new(&mut self.tube_radius, 0.1..=3.0).step_by(0.05)).changed() {
                                self.indices_dirty = true;
                            }
                        });
                    }

                    ui.separator();
                }

                // ── Bundle surface mesh ───────────────────────────────────
                if self.has_streamlines {
                    ui.label("Bundle Surface Mesh");
                    let mut rebuild = false;
                    let mesh_toggled = ui.checkbox(&mut self.show_bundle_mesh, "Show surface").changed();
                    if mesh_toggled {
                        if self.show_bundle_mesh {
                            rebuild = true;
                        } else {
                            if let Some(rs) = frame.wgpu_render_state() {
                                if let Some(mr) = rs.renderer.write()
                                    .callback_resources.get_mut::<MeshResources>()
                                {
                                    mr.clear_bundle_mesh();
                                    self.bundle_meshes_cpu.clear();
                                }
                            }
                        }
                    }

                    if self.show_bundle_mesh {
                        // Source selector
                        let src_label = match self.bundle_mesh_source {
                            BundleMeshSource::All       => "All streamlines",
                            BundleMeshSource::Selection => "Current selection",
                            BundleMeshSource::PerGroup  => "Per group",
                        };
                        egui::ComboBox::from_id_salt("bundle_src")
                            .selected_text(src_label)
                            .show_ui(ui, |ui| {
                                for (variant, label) in [
                                    (BundleMeshSource::All,       "All streamlines"),
                                    (BundleMeshSource::Selection, "Current selection"),
                                    (BundleMeshSource::PerGroup,  "Per group"),
                                ] {
                                    if ui.selectable_value(&mut self.bundle_mesh_source, variant, label).changed() {
                                        rebuild = true;
                                    }
                                }
                            });

                        ui.horizontal(|ui| {
                            ui.label("Voxel (mm):");
                            rebuild |= ui.add(
                                egui::Slider::new(&mut self.bundle_mesh_voxel_size, 0.5..=10.0).step_by(0.5)
                            ).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Threshold:");
                            rebuild |= ui.add(
                                egui::Slider::new(&mut self.bundle_mesh_threshold, 1.0..=50.0).step_by(1.0)
                            ).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Smooth (σ):");
                            rebuild |= ui.add(
                                egui::Slider::new(&mut self.bundle_mesh_smooth, 0.0..=4.0).step_by(0.25)
                            ).changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Opacity:");
                            ui.add(egui::Slider::new(&mut self.bundle_mesh_opacity, 0.0..=1.0));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Ambient:");
                            ui.add(egui::Slider::new(&mut self.bundle_mesh_ambient, 0.0..=1.0));
                        });

                        if rebuild {
                            self.bundle_mesh_dirty_at = Some(std::time::Instant::now());
                        }

                        if ui.button("Rebuild now").clicked() {
                            self.bundle_mesh_dirty_at = Some(
                                std::time::Instant::now()
                                    - std::time::Duration::from_millis(200),
                            );
                        }

                        let building = self.bundle_mesh_dirty_at.is_some() || self.bundle_mesh_pending.is_some();
                        ui.add_enabled(false, egui::Label::new(
                            egui::RichText::new(if building { "Building…" } else { " " }).small()
                        ));
                    }
                    ui.separator();
                }

                // ── Group visibility ──
                if self.trx_data.as_ref().map_or(false, |d| !d.groups.is_empty()) {
                    ui.label("Groups");

                    let group_info: Vec<(String, usize)> = self
                        .trx_data
                        .as_ref()
                        .map(|d| {
                            d.groups
                                .iter()
                                .map(|(name, members)| (name.clone(), members.len()))
                                .collect()
                        })
                        .unwrap_or_default();

                    for (i, (name, count)) in group_info.iter().enumerate() {
                        if i < self.group_visible.len() {
                            let label = format!("{name} ({count})");
                            if ui.checkbox(&mut self.group_visible[i], label).changed() {
                                self.indices_dirty = true;
                            }
                        }
                    }

                    ui.separator();
                }

                // ── Streamline filtering ──
                if let Some(ref data) = self.trx_data {
                    ui.label("Streamline Filter");
                    let nb = data.nb_streamlines;
                    ui.horizontal(|ui| {
                        ui.label("Max:");
                        if ui
                            .add(egui::Slider::new(&mut self.max_streamlines, 1..=nb))
                            .changed()
                        {
                            self.indices_dirty = true;
                        }
                    });
                    if ui
                        .checkbox(&mut self.use_random_subset, "Randomize")
                        .changed()
                    {
                        if self.use_random_subset {
                            // Simple xorshift shuffle with fixed seed
                            let mut order: Vec<u32> = (0..nb as u32).collect();
                            let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
                            for i in (1..order.len()).rev() {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                let j = (rng as usize) % (i + 1);
                                order.swap(i, j);
                            }
                            self.streamline_order = order;
                        } else {
                            self.streamline_order = (0..nb as u32).collect();
                        }
                        self.indices_dirty = true;
                    }

                    ui.horizontal(|ui| {
                        ui.label("Slab (mm):");
                        ui.add(egui::Slider::new(&mut self.slab_half_width, 0.5..=50.0));
                    });

                    ui.separator();

                    // ── Sphere query ──
                    ui.label("Sphere Query");
                    ui.small("Ctrl+click in slice view to place");
                    if self.sphere_query_active {
                        ui.horizontal(|ui| {
                            ui.label("Radius:");
                            if ui
                                .add(egui::Slider::new(&mut self.sphere_radius, 1.0..=50.0).suffix(" mm"))
                                .changed()
                            {
                                if let Some(ref data) = self.trx_data {
                                    self.sphere_query_result =
                                        Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                                }
                                self.indices_dirty = true;
                            }
                        });
                        let half = self.volume_extent * 0.6;
                        let mut sphere_moved = false;
                        ui.horizontal(|ui| {
                            ui.label("X:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.x,
                                    (self.volume_center.x - half)..=(self.volume_center.x + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Y:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.y,
                                    (self.volume_center.y - half)..=(self.volume_center.y + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Z:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.z,
                                    (self.volume_center.z - half)..=(self.volume_center.z + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        if sphere_moved {
                            if let Some(ref data) = self.trx_data {
                                self.sphere_query_result =
                                    Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                            }
                            self.indices_dirty = true;
                        }
                        if let Some(ref result) = self.sphere_query_result {
                            ui.small(format!("Matched: {} streamlines", result.len()));
                        }
                        if ui.button("Clear query").clicked() {
                            self.sphere_query_active = false;
                            self.sphere_query_result = None;
                            self.indices_dirty = true;
                        }
                    }


                    ui.separator();
                }

                // Slice controls
                if self.nifti_volume.is_some() {
                    ui.label("Slice Position");

                    let max_k;
                    let max_j;
                    let max_i;
                    if let Some(ref vol) = self.nifti_volume {
                        max_k = vol.dims[2].saturating_sub(1);
                        max_j = vol.dims[1].saturating_sub(1);
                        max_i = vol.dims[0].saturating_sub(1);
                    } else {
                        return;
                    }

                    ui.horizontal(|ui| {
                        ui.label("Axial:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[0], 0..=max_k))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Coronal:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[1], 0..=max_j))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Sagittal:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[2], 0..=max_i))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });

                    // Intensity windowing
                    ui.separator();
                    ui.label("Intensity Window");
                    ui.horizontal(|ui| {
                        ui.label("Center:");
                        ui.add(egui::Slider::new(&mut self.window_center, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Width:");
                        ui.add(egui::Slider::new(&mut self.window_width, 0.01..=2.0));
                    });
                }
            });

        // ── Main content: 4 viewports ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.trx_data.is_none() && self.nifti_volume.is_none() && self.gifti_surfaces.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a TRX, NIfTI, or GIFTI file from the sidebar to begin.");
                });
                return;
            }

            let available = ui.available_size();
            let top_height = (available.y * 0.6).max(100.0);
            let bottom_height =
                (available.y - top_height - ui.spacing().item_spacing.y).max(50.0);

            // ── 3D Viewport ──
            let (rect_3d, response_3d) = ui.allocate_exact_size(
                egui::vec2(available.x, top_height),
                egui::Sense::click_and_drag(),
            );

            if response_3d.dragged_by(egui::PointerButton::Primary) {
                let delta = response_3d.drag_delta();
                self.camera_3d.handle_drag(delta.x, -delta.y);
            }
            if response_3d.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera_3d.handle_scroll(scroll * 0.01);
                }
            }

            let aspect_3d = rect_3d.width() / rect_3d.height().max(1.0);
            let vp_3d = self.camera_3d.view_projection(aspect_3d);
            let surface_draws: Vec<(usize, MeshDrawStyle)> = self
                .gifti_surfaces
                .iter()
                .filter(|s| s.visible)
                .map(|s| {
                    (
                        s.gpu_index,
                        MeshDrawStyle {
                            color: [s.color[0], s.color[1], s.color[2], s.opacity],
                            scalar_min: s.range_min,
                            scalar_max: s.range_max,
                            scalar_enabled: s.show_projection_map,
                            colormap: s.projection_colormap,
                            ambient_strength: s.surface_ambient,
                            gloss: s.surface_gloss,
                            map_opacity: s.map_opacity,
                            map_threshold: s.map_threshold,
                        },
                    )
                })
                .collect();

            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect_3d,
                Scene3DCallback {
                    view_proj: vp_3d,
                    camera_pos: self.camera_3d.eye(),
                    has_streamlines: self.has_streamlines,
                    show_streamlines: self.show_streamlines,
                    has_slices: self.has_slices,
                    slice_visible: self.slice_visible,
                    window_center: self.window_center,
                    window_width: self.window_width,
                    surface_draws,
                    render_style: self.render_style,
                    tube_radius: self.tube_radius,
                    show_bundle_mesh: self.show_bundle_mesh,
                    bundle_mesh_opacity: self.bundle_mesh_opacity,
                    bundle_mesh_ambient: self.bundle_mesh_ambient,
                },
            ));

            // Draw 3D orientation axes
            self.draw_3d_axes(ui, rect_3d, vp_3d);

            // Draw 3D sphere indicator
            if self.sphere_query_active {
                self.draw_sphere_3d(ui, rect_3d, vp_3d);
            }

            // ── Bottom row: 3 slice views ──
            let slice_width = ((available.x - 2.0 * ui.spacing().item_spacing.x) / 3.0).max(10.0);
            let slice_height = (bottom_height - ui.spacing().item_spacing.y - 18.0).max(10.0);

            ui.horizontal(|ui| {
                let axis_names  = ["Axial",   "Coronal",  "Sagittal"];
                let axis_labels = ["Z",       "Y",        "X"];
                for i in 0..3 {
                    ui.vertical(|ui| {
                        let pos_mm = self.slice_world_position(i);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.slice_visible[i], "");
                            ui.label(format!("{} ({} = {:.1} mm)", axis_names[i], axis_labels[i], pos_mm));
                        });
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(slice_width, slice_height),
                            egui::Sense::click_and_drag(),
                        );

                        if response.hovered() {
                            let scroll = ui.input(|inp| inp.smooth_scroll_delta.y);
                            if scroll.abs() > 0.5 {
                                if let Some(ref vol) = self.nifti_volume {
                                    let max_idx = match i {
                                        0 => vol.dims[2].saturating_sub(1),
                                        1 => vol.dims[1].saturating_sub(1),
                                        2 => vol.dims[0].saturating_sub(1),
                                        _ => 0,
                                    };
                                    let delta = if scroll > 0.0 { 1isize } else { -1 };
                                    let new_idx = (self.slice_indices[i] as isize + delta)
                                        .clamp(0, max_idx as isize)
                                        as usize;
                                    if new_idx != self.slice_indices[i] {
                                        self.slice_indices[i] = new_idx;
                                        self.slices_dirty = true;
                                    }
                                } else {
                                    // No NIfTI: scroll moves the world-space slab position.
                                    let step = self.volume_extent * 0.005;
                                    let delta = if scroll > 0.0 { step } else { -step };
                                    let half = self.volume_extent * 0.6;
                                    let center = match i {
                                        0 => self.volume_center.z,
                                        1 => self.volume_center.y,
                                        _ => self.volume_center.x,
                                    };
                                    self.slice_world_offsets[i] = (self.slice_world_offsets[i] + delta)
                                        .clamp(center - half, center + half);
                                }
                            }
                        }

                        // Ctrl+click to place sphere query center
                        if response.clicked() && self.has_streamlines {
                            let ctrl = ui.input(|inp| inp.modifiers.ctrl || inp.modifiers.command);
                            if ctrl {
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let aspect_q = rect.width() / rect.height().max(1.0);
                                    let sp = self.slice_world_position(i);
                                    let world = self.slice_cameras[i].screen_to_world(
                                        pos, rect, aspect_q, sp,
                                    );
                                    self.sphere_center = world;
                                    self.sphere_query_active = true;
                                    if let Some(ref data) = self.trx_data {
                                        self.sphere_query_result =
                                            Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                                    }
                                    self.indices_dirty = true;
                                }
                            }
                        }

                        let aspect = rect.width() / rect.height().max(1.0);
                        let slice_pos = self.slice_world_position(i);
                        let vp_slice =
                            self.slice_cameras[i].view_projection(aspect, slice_pos);

                        // Slab axis: axial=Z(2), coronal=Y(1), sagittal=X(0)
                        let slab_axis = match i {
                            0 => 2u32, // axial → Z
                            1 => 1u32, // coronal → Y
                            _ => 0u32, // sagittal → X
                        };

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            SliceViewCallback {
                                view_proj: vp_slice,
                                quad_index: i,
                                bind_group_index: i + 1,
                                has_slices: self.has_slices,
                                has_streamlines: self.has_streamlines,
                                show_streamlines: self.show_streamlines,
                                window_center: self.window_center,
                                window_width: self.window_width,
                                slab_axis,
                                slab_min: slice_pos - self.slab_half_width,
                                slab_max: slice_pos + self.slab_half_width,
                            },
                        ));

                        // Draw crosshairs showing the other two slice positions.
                        // Drawn whenever any data is loaded so the user can see the
                        // slice plane even without a NIfTI background.
                        self.draw_crosshairs(ui, rect, i, vp_slice);

                        // Draw anatomical orientation labels
                        self.draw_orientation_labels(ui, rect, i, vp_slice);

                        // Draw sphere query circle
                        if self.sphere_query_active {
                            self.draw_sphere_circle(ui, rect, i, vp_slice, slice_pos);
                        }
                        self.draw_mesh_intersections(ui, rect, i, vp_slice, slice_pos);
                    });
                }
            });
        });
    }
}

impl TrxViewerApp {
    /// Draw the sphere query as a circle on a slice view.
    fn draw_sphere_circle(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        // Get the sphere center coordinate along this slice's normal axis
        let center_on_axis = match axis_index {
            0 => self.sphere_center.z, // axial
            1 => self.sphere_center.y, // coronal
            _ => self.sphere_center.x, // sagittal
        };

        let d = (slice_pos - center_on_axis).abs();
        if d >= self.sphere_radius {
            return;
        }

        // Circle radius on this slice plane
        let circle_r = (self.sphere_radius * self.sphere_radius - d * d).sqrt();

        // Project sphere center to screen
        let clip = view_proj * self.sphere_center.extend(1.0);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
        let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height();

        // Convert world-space radius to screen pixels
        // Use a point offset by circle_r in the first in-plane axis
        let offset_world = match axis_index {
            0 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            1 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            _ => self.sphere_center + Vec3::new(0.0, circle_r, 0.0),
        };
        let clip2 = view_proj * offset_world.extend(1.0);
        let ndc_x2 = clip2.x / clip2.w;
        let ndc_y2 = clip2.y / clip2.w;
        let sx2 = rect.left() + (ndc_x2 + 1.0) * 0.5 * rect.width();
        let sy2 = rect.top() + (1.0 - ndc_y2) * 0.5 * rect.height();
        let screen_r = ((sx2 - sx).powi(2) + (sy2 - sy).powi(2)).sqrt();

        let painter = ui.painter_at(rect);
        let circle_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        painter.circle_stroke(
            egui::pos2(sx, sy),
            screen_r,
            egui::Stroke::new(2.0, circle_color),
        );
    }

    /// Draw three axis-aligned circles in the 3D view to indicate the sphere query position.
    fn draw_sphere_3d(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        view_proj: glam::Mat4,
    ) {
        let painter = ui.painter_at(rect);
        let color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        let stroke = egui::Stroke::new(1.5, color);
        let n = 48usize;
        let c = self.sphere_center;
        let r = self.sphere_radius;

        let project = |p: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * p.extend(1.0);
            let nx = clip.x / clip.w;
            let ny = clip.y / clip.w;
            egui::pos2(
                rect.left() + (nx + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ny) * 0.5 * rect.height(),
            )
        };

        // Three rings: XY (axial plane), XZ (coronal), YZ (sagittal)
        let ring_points = |axis_a: glam::Vec3, axis_b: glam::Vec3| -> Vec<egui::Pos2> {
            (0..=n)
                .map(|k| {
                    let t = k as f32 / n as f32 * std::f32::consts::TAU;
                    project(c + axis_a * (r * t.cos()) + axis_b * (r * t.sin()))
                })
                .collect()
        };

        for pts in [
            ring_points(glam::Vec3::X, glam::Vec3::Y), // XY plane
            ring_points(glam::Vec3::X, glam::Vec3::Z), // XZ plane
            ring_points(glam::Vec3::Y, glam::Vec3::Z), // YZ plane
        ] {
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], stroke);
            }
        }
        // Small crosshair at center
        let cp = project(c);
        let arm = 6.0;
        painter.line_segment([egui::pos2(cp.x - arm, cp.y), egui::pos2(cp.x + arm, cp.y)], stroke);
        painter.line_segment([egui::pos2(cp.x, cp.y - arm), egui::pos2(cp.x, cp.y + arm)], stroke);
    }

    /// Draw anatomical orientation labels (R/L/A/P/S/I) on a slice view.
    fn draw_orientation_labels(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        _axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        let center = self.volume_center;
        let axis_len = (self.volume_extent * 0.2).max(10.0);

        // Define the 6 anatomical directions as offsets from center.
        let directions: &[(Vec3, &str)] = &[
            (Vec3::X * axis_len, "R"),  // +X = Right
            (-Vec3::X * axis_len, "L"), // -X = Left
            (Vec3::Y * axis_len, "A"),  // +Y = Anterior
            (-Vec3::Y * axis_len, "P"), // -Y = Posterior
            (Vec3::Z * axis_len, "S"),  // +Z = Superior
            (-Vec3::Z * axis_len, "I"), // -Z = Inferior
        ];

        let project = |world: Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            if clip.w.abs() < 1e-6 {
                return rect.center();
            }
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        let painter = ui.painter_at(rect);
        let label_color = egui::Color32::from_rgb(220, 220, 220);
        let font = egui::FontId::proportional(14.0);
        let margin = 16.0;
        let center_screen = project(center);

        // Place labels by projecting a small offset and extending from center to the viewport edge.
        for &(offset, label) in directions {
            let p = project(center + offset);
            let delta = egui::vec2(p.x - center_screen.x, p.y - center_screen.y);
            let len2 = delta.length_sq();
            // Skip look-axis directions that collapse to the center in this view.
            if len2 < 1e-6 {
                continue;
            }
            let dir = delta / len2.sqrt();
            let tx = if dir.x.abs() > 1e-6 {
                ((rect.width() * 0.5 - margin) / dir.x.abs()).abs()
            } else {
                f32::INFINITY
            };
            let ty = if dir.y.abs() > 1e-6 {
                ((rect.height() * 0.5 - margin) / dir.y.abs()).abs()
            } else {
                f32::INFINITY
            };
            let t = tx.min(ty);
            let label_pos = egui::pos2(
                center_screen.x + dir.x * t,
                center_screen.y + dir.y * t,
            );

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                label,
                font.clone(),
                label_color,
            );
        }
    }

    /// Draw 3D orientation axes in the corner of the 3D viewport.
    fn draw_3d_axes(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);

        // Place axes in bottom-left corner
        let origin_screen = egui::pos2(rect.left() + 50.0, rect.bottom() - 50.0);
        let axis_length = 30.0;

        let axes = [
            (Vec3::X, "R", egui::Color32::RED),
            (Vec3::Y, "A", egui::Color32::GREEN),
            (Vec3::Z, "S", egui::Color32::from_rgb(80, 120, 255)),
        ];

        for (dir, label, color) in axes {
            // Project the direction vector (just the rotation, no translation)
            let clip0 = view_proj * Vec3::ZERO.extend(1.0);
            let clip1 = view_proj * dir.extend(1.0);
            // Direction in NDC
            let ndc0 = egui::vec2(clip0.x / clip0.w, clip0.y / clip0.w);
            let ndc1 = egui::vec2(clip1.x / clip1.w, clip1.y / clip1.w);
            let dir_ndc = ndc1 - ndc0;
            let dir_screen = egui::vec2(dir_ndc.x, -dir_ndc.y); // flip Y
            let dir_norm = if dir_screen.length() > 0.001 {
                dir_screen / dir_screen.length()
            } else {
                egui::vec2(0.0, 0.0)
            };

            let end = origin_screen + dir_norm * axis_length;
            painter.line_segment(
                [origin_screen, end],
                egui::Stroke::new(2.0, color),
            );
            painter.text(
                end + dir_norm * 10.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }
}

// ── Paint Callbacks ──

struct Scene3DCallback {
    view_proj: glam::Mat4,
    camera_pos: glam::Vec3,
    has_streamlines: bool,
    show_streamlines: bool,
    has_slices: bool,
    slice_visible: [bool; 3],
    window_center: f32,
    window_width: f32,
    surface_draws: Vec<(usize, MeshDrawStyle)>,
    render_style: RenderStyle,
    tube_radius: f32,
    show_bundle_mesh: bool,
    bundle_mesh_opacity: f32,
    bundle_mesh_ambient: f32,
}

impl egui_wgpu::CallbackTrait for Scene3DCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get_mut::<StreamlineResources>() {
            // depth_cue mode reuses tube_radius field as depth_far
            let aux = if self.render_style == RenderStyle::DepthCue { 300.0 } else { self.tube_radius };
            res.update_uniforms(queue, 0, self.view_proj, self.camera_pos,
                self.render_style as u32, 3, 0.0, 0.0, aux);
        }
        if let Some(res) = callback_resources.get_mut::<SliceResources>() {
            res.update_uniforms(queue, 0, self.view_proj, self.window_center, self.window_width);
        }
        if let Some(res) = callback_resources.get_mut::<MeshResources>() {
            for (surface_index, style) in &self.surface_draws {
                res.update_surface_uniforms(
                    queue,
                    *surface_index,
                    0,
                    self.view_proj,
                    style,
                    self.camera_pos,
                );
            }
            if self.show_bundle_mesh {
                res.update_bundle_uniforms(
                    queue,
                    self.view_proj,
                    self.camera_pos,
                    self.bundle_mesh_opacity,
                    self.bundle_mesh_ambient,
                );
            }
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if self.has_slices {
            if let Some(sr) = callback_resources.get::<SliceResources>() {
                render_pass.set_pipeline(&sr.pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                render_pass.set_index_buffer(
                    sr.quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                for i in 0..3 {
                    if !self.slice_visible[i] { continue; }
                    render_pass.set_vertex_buffer(0, sr.quad_buffers[i].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }

        if self.has_streamlines && self.show_streamlines {
            if let Some(sr) = callback_resources.get::<StreamlineResources>() {
                render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                if self.render_style == RenderStyle::Tubes {
                    if let (Some(tvb), Some(tib)) = (&sr.tube_vertex_buffer, &sr.tube_index_buffer) {
                        render_pass.set_pipeline(&sr.tube_pipeline);
                        render_pass.set_vertex_buffer(0, tvb.slice(..));
                        render_pass.set_index_buffer(tib.slice(..), wgpu::IndexFormat::Uint32);
                        render_pass.draw_indexed(0..sr.num_tube_indices, 0, 0..1);
                    }
                } else {
                    render_pass.set_pipeline(&sr.pipeline);
                    render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                    render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                    render_pass.set_vertex_buffer(2, sr.tangent_buffer.slice(..));
                    render_pass.set_index_buffer(sr.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
                }
            }
        }

        if let Some(mr) = callback_resources.get::<MeshResources>() {
            if !self.surface_draws.is_empty() {
                mr.paint(render_pass, 0, &self.surface_draws);
            }
            if self.show_bundle_mesh {
                mr.paint_bundle(render_pass);
            }
        }
    }
}

struct SliceViewCallback {
    view_proj: glam::Mat4,
    quad_index: usize,
    bind_group_index: usize,
    has_slices: bool,
    has_streamlines: bool,
    show_streamlines: bool,
    window_center: f32,
    window_width: f32,
    /// Slab clipping axis for streamlines: 0=X, 1=Y, 2=Z.
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
}

impl egui_wgpu::CallbackTrait for SliceViewCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get_mut::<SliceResources>() {
            res.update_uniforms(queue, self.bind_group_index, self.view_proj, self.window_center, self.window_width);
        }
        if let Some(res) = callback_resources.get_mut::<StreamlineResources>() {
            // Slice views always render flat lines regardless of 3D render style.
            res.update_uniforms(
                queue,
                self.bind_group_index,
                self.view_proj,
                glam::Vec3::ZERO,
                0, // flat
                self.slab_axis,
                self.slab_min,
                self.slab_max,
                0.5,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if self.has_slices {
            if let Some(sr) = callback_resources.get::<SliceResources>() {
                render_pass.set_pipeline(&sr.pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                render_pass.set_index_buffer(
                    sr.quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.set_vertex_buffer(0, sr.quad_buffers[self.quad_index].slice(..));
                render_pass.draw_indexed(0..6, 0, 0..1);
            }
        }

        if self.has_streamlines && self.show_streamlines {
            if let Some(sr) = callback_resources.get::<StreamlineResources>() {
                render_pass.set_pipeline(&sr.slice_pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                render_pass.set_vertex_buffer(2, sr.tangent_buffer.slice(..));
                render_pass.set_index_buffer(sr.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
            }
        }
    }
}
