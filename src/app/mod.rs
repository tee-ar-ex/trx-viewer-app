mod callbacks;
mod dirty_updates;
mod file_loading;
mod helpers;
mod state;
mod ui;

use callbacks::{Scene3DCallback, SliceViewCallback, StreamlineDrawInfo, VolumeDrawInfo};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use glam::Vec3;

use crate::data::bundle_mesh::{build_bundle_mesh, BundleMesh};
use crate::data::loaded_files::{BundleMeshSource, FileId, LoadedNifti, LoadedTrx};
use crate::data::trx_data::RenderStyle;
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis};
use crate::renderer::streamline_renderer::AllStreamlineResources;

use state::{
    LoadedGiftiSurface, SurfaceProjectionCacheKey,
    SurfaceProjectionCacheValue,
};

/// Main application state.
pub struct TrxViewerApp {
    pub(crate) trx_files: Vec<LoadedTrx>,
    pub(crate) nifti_files: Vec<LoadedNifti>,
    pub(crate) next_file_id: FileId,
    pub(crate) camera_3d: OrbitCamera,
    pub(crate) slice_cameras: [OrthoSliceCamera; 3],
    pub(crate) slice_indices: [usize; 3],
    pub(crate) slices_dirty: bool,
    pub(crate) volume_center: Vec3,
    pub(crate) volume_extent: f32,
    pub(crate) error_msg: Option<String>,
    pub(crate) gifti_surfaces: Vec<LoadedGiftiSurface>,
    pub(crate) surface_query_active: bool,
    pub(crate) surface_query_surface: usize,
    pub(crate) surface_query_result: Option<HashSet<u32>>,
    pub(crate) selection_revision: u64,
    pub(crate) surface_projection_cache: HashMap<SurfaceProjectionCacheKey, SurfaceProjectionCacheValue>,
    pub(crate) surface_projection_dirty: bool,
    pub(crate) show_streamlines: bool,
    pub(crate) slice_visible: [bool; 3],
    pub(crate) slice_world_offsets: [f32; 3],
    pub(crate) sphere_query_active: bool,
    pub(crate) sphere_center: Vec3,
    pub(crate) sphere_radius: f32,
}

impl TrxViewerApp {
    /// Returns true if any TRX file is loaded.
    pub(crate) fn has_streamlines(&self) -> bool {
        !self.trx_files.is_empty()
    }

    pub fn new(
        cc: &eframe::CreationContext<'_>,
        trx_path: Option<String>,
        nifti_path: Option<String>,
    ) -> Self {
        let mut app = Self {
            trx_files: Vec::new(),
            nifti_files: Vec::new(),
            next_file_id: 0,
            camera_3d: OrbitCamera::new(Vec3::ZERO, 200.0),
            slice_cameras: [
                OrthoSliceCamera::new(SliceAxis::Axial, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Coronal, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Sagittal, Vec3::ZERO, 200.0),
            ],
            slice_indices: [0; 3],
            slices_dirty: false,
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
            show_streamlines: true,
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            sphere_query_active: false,
            sphere_center: Vec3::ZERO,
            sphere_radius: 10.0,
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
}

impl eframe::App for TrxViewerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Update slice positions if dirty
        if self.slices_dirty {
            if let Some(rs) = frame.wgpu_render_state() {
                let renderer = rs.renderer.read();
                if let Some(all) = renderer.callback_resources.get::<AllSliceResources>() {
                    for (file_id, sr) in &all.entries {
                        if let Some(nf) = self.nifti_files.iter().find(|n| n.id == *file_id) {
                            sr.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], &nf.volume);
                            sr.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], &nf.volume);
                            sr.update_slice(&rs.queue, SliceAxis::Sagittal, self.slice_indices[2], &nf.volume);
                        }
                    }
                }
            }
            self.slices_dirty = false;
        }

        // ── Per-TRX dirty flag processing ──────────────────────────────────────
        for i in 0..self.trx_files.len() {
            // Update vertex colors if dirty
            if self.trx_files[i].colors_dirty {
                if let Some(rs) = frame.wgpu_render_state() {
                    let renderer = rs.renderer.read();
                    if let Some(all) = renderer.callback_resources.get::<AllStreamlineResources>() {
                        let id = self.trx_files[i].id;
                        if let Some((_, sr)) = all.entries.iter().find(|(fid, _)| *fid == id) {
                            sr.update_colors(&rs.queue, &self.trx_files[i].data.colors);
                        }
                    }
                }
                // Tube vertices embed colors — rebuild geometry so the new colors take effect.
                if self.trx_files[i].render_style == RenderStyle::Tubes {
                    self.trx_files[i].indices_dirty = true;
                }
                // Bundle mesh vertex colors come from the color buffer; trigger rebuild.
                if self.trx_files[i].show_bundle_mesh {
                    self.trx_files[i].bundle_mesh_dirty_at = Some(std::time::Instant::now());
                }
                self.trx_files[i].colors_dirty = false;
            }

            // Update index buffer (and tube geometry) if any filter changed
            if self.trx_files[i].indices_dirty {
                if let Some(rs) = frame.wgpu_render_state() {
                    let trx = &self.trx_files[i];
                    if trx.render_style == RenderStyle::Tubes {
                        let selected = trx.data.filtered_streamline_indices(
                            &trx.group_visible,
                            trx.max_streamlines,
                            &trx.streamline_order,
                            trx.sphere_query_result.as_ref(),
                            self.surface_query_result.as_ref(),
                        );
                        let (tube_verts, tube_indices) = trx.data.build_tube_vertices(&selected);
                        let mut renderer = rs.renderer.write();
                        if let Some(all) = renderer.callback_resources.get_mut::<AllStreamlineResources>() {
                            let id = trx.id;
                            if let Some((_, sr)) = all.entries.iter_mut().find(|(fid, _)| *fid == id) {
                                sr.update_tube_geometry(&rs.device, &tube_verts, &tube_indices);
                                let line_indices = trx.data.build_index_buffer(
                                    &trx.group_visible, trx.max_streamlines,
                                    &trx.streamline_order,
                                    trx.sphere_query_result.as_ref(),
                                    self.surface_query_result.as_ref(),
                                );
                                sr.update_indices(&rs.device, &rs.queue, &line_indices);
                            }
                        }
                    } else {
                        let trx = &self.trx_files[i];
                        let indices = trx.data.build_index_buffer(
                            &trx.group_visible,
                            trx.max_streamlines,
                            &trx.streamline_order,
                            trx.sphere_query_result.as_ref(),
                            self.surface_query_result.as_ref(),
                        );
                        let mut renderer = rs.renderer.write();
                        if let Some(all) = renderer.callback_resources.get_mut::<AllStreamlineResources>() {
                            let id = trx.id;
                            if let Some((_, sr)) = all.entries.iter_mut().find(|(fid, _)| *fid == id) {
                                sr.update_indices(&rs.device, &rs.queue, &indices);
                            }
                        }
                    }
                }
                self.trx_files[i].indices_dirty = false;
                self.selection_revision = self.selection_revision.wrapping_add(1);
                self.surface_projection_dirty = true;
                // Bundle mesh rebuilds when selection changes (not needed for All source).
                if self.trx_files[i].show_bundle_mesh
                    && self.trx_files[i].bundle_mesh_source != BundleMeshSource::All
                {
                    self.trx_files[i].bundle_mesh_dirty_at = Some(std::time::Instant::now());
                }
            }
        }

        // ── Bundle mesh: check debounce + receive completed mesh ──────────────
        for i in 0..self.trx_files.len() {
            if let Some(t) = self.trx_files[i].bundle_mesh_dirty_at {
                if t.elapsed() >= std::time::Duration::from_millis(150) {
                    self.trx_files[i].bundle_mesh_dirty_at = None;
                    if self.trx_files[i].show_bundle_mesh {
                        // Extract all data we need before the mutable borrow.
                        let voxel_size = self.trx_files[i].bundle_mesh_voxel_size;
                        let threshold  = self.trx_files[i].bundle_mesh_threshold;
                        let smooth     = self.trx_files[i].bundle_mesh_smooth;
                        let source     = self.trx_files[i].bundle_mesh_source;
                        let (tx, rx) = std::sync::mpsc::channel();
                        let egui_ctx = ctx.clone();

                        // Prepare thread payload from immutable borrows before assigning rx.
                        enum BundlePayload {
                            All(Vec<[f32;3]>, Vec<[f32;4]>),
                            Selection(Vec<[f32;3]>, Vec<[f32;4]>),
                            PerGroup(Vec<(String, Vec<[f32;3]>, Vec<[f32;4]>)>),
                        }
                        let payload = {
                            let trx = &self.trx_files[i];
                            match source {
                                BundleMeshSource::All => {
                                    BundlePayload::All(trx.data.positions.clone(), trx.data.colors.clone())
                                }
                                BundleMeshSource::Selection => {
                                    let selected = trx.data.filtered_streamline_indices(
                                        &trx.group_visible,
                                        trx.max_streamlines,
                                        &trx.streamline_order,
                                        trx.sphere_query_result.as_ref(),
                                        self.surface_query_result.as_ref(),
                                    );
                                    let (positions, colors) = trx.data.selected_vertex_data(&selected);
                                    BundlePayload::Selection(positions, colors)
                                }
                                BundleMeshSource::PerGroup => {
                                    let group_data: Vec<(String, Vec<[f32;3]>, Vec<[f32;4]>)> =
                                        trx.data.groups.iter()
                                            .enumerate()
                                            .filter(|(gi, _)| trx.group_visible.get(*gi).copied().unwrap_or(true))
                                            .map(|(_, (name, members))| {
                                                let (pos, col) = trx.data.selected_vertex_data(members);
                                                (name.clone(), pos, col)
                                            })
                                            .collect();
                                    BundlePayload::PerGroup(group_data)
                                }
                            }
                        };

                        // Now safe to mutably borrow for pending assignment.
                        self.trx_files[i].bundle_mesh_pending = Some(rx);

                        match payload {
                            BundlePayload::All(positions, colors) => {
                                std::thread::spawn(move || {
                                    let mut out = Vec::new();
                                    if let Some(m) = build_bundle_mesh(&positions, &colors, voxel_size, threshold, smooth) {
                                        out.push((m, "all".to_string()));
                                    }
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                            BundlePayload::Selection(positions, colors) => {
                                std::thread::spawn(move || {
                                    let mut out = Vec::new();
                                    if let Some(m) = build_bundle_mesh(&positions, &colors, voxel_size, threshold, smooth) {
                                        out.push((m, "selection".to_string()));
                                    }
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                            BundlePayload::PerGroup(group_data) => {
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
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }

            if let Some(rx) = &self.trx_files[i].bundle_mesh_pending {
                if let Ok(meshes) = rx.try_recv() {
                    self.trx_files[i].bundle_mesh_pending = None;
                    let file_id = self.trx_files[i].id;
                    if let Some(rs) = frame.wgpu_render_state() {
                        let mut renderer = rs.renderer.write();
                        if let Some(mr) = renderer.callback_resources.get_mut::<MeshResources>() {
                            if meshes.is_empty() {
                                mr.clear_bundle_mesh(file_id);
                                self.trx_files[i].bundle_meshes_cpu.clear();
                            } else {
                                mr.set_bundle_meshes(file_id, &rs.device, &meshes);
                                self.trx_files[i].bundle_meshes_cpu = meshes.into_iter().map(|(m, _)| m).collect();
                            }
                        }
                    }
                }
            }
        }

        if self.surface_projection_dirty {
            self.refresh_surface_projections(frame);
            self.surface_projection_dirty = false;
        }

        // ── Handle dropped files ──
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in &dropped {
            if let Some(path) = &file.path {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                match ext.as_str() {
                    "trx" => {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_trx(path, rs);
                        }
                    }
                    "gz" if stem.ends_with(".nii") => {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_nifti(path, rs);
                        }
                    }
                    "nii" => {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_nifti(path, rs);
                        }
                    }
                    "gii" | "gifti" => {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_gifti_surface(path, rs);
                        }
                    }
                    _ => {
                        self.error_msg = Some(format!("Unknown file type: .{}", ext));
                    }
                }
            }
        }

        // ── Menu bar ──
        let menu_action = ui::menu_bar::show_menu_bar(ctx);
        if menu_action.open_trx {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("TRX files", &["trx"])
                .pick_file()
            {
                if let Some(rs) = frame.wgpu_render_state() {
                    self.load_trx(&path, rs);
                }
            }
        }
        if menu_action.open_nifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                .pick_file()
            {
                if let Some(rs) = frame.wgpu_render_state() {
                    self.load_nifti(&path, rs);
                }
            }
        }
        if menu_action.open_gifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GIFTI files", &["gii", "gifti"])
                .pick_file()
            {
                if let Some(rs) = frame.wgpu_render_state() {
                    self.load_gifti_surface(&path, rs);
                }
            }
        }

        // ── Sidebar ──
        self.show_sidebar(ctx, frame);

        // ── Main content: 4 viewports ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.trx_files.is_empty() && self.nifti_files.is_empty() && self.gifti_surfaces.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("Drop files here or use File > Open to begin.");
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

            let volume_draws: Vec<VolumeDrawInfo> = self.nifti_files.iter()
                .filter(|n| n.visible)
                .map(|n| VolumeDrawInfo {
                    file_id: n.id,
                    window_center: n.window_center,
                    window_width: n.window_width,
                    colormap: n.colormap.as_u32(),
                    opacity: n.opacity,
                })
                .collect();

            let streamline_draws: Vec<StreamlineDrawInfo> = self.trx_files.iter()
                .filter(|t| t.visible)
                .map(|t| StreamlineDrawInfo {
                    file_id: t.id,
                    render_style: t.render_style,
                    tube_radius: t.tube_radius,
                    slab_half_width: t.slab_half_width,
                })
                .collect();

            let bundle_draws: Vec<callbacks::BundleDrawInfo> = self.trx_files.iter()
                .filter(|t| t.show_bundle_mesh)
                .map(|t| callbacks::BundleDrawInfo {
                    file_id: t.id,
                    opacity: t.bundle_mesh_opacity,
                    ambient: t.bundle_mesh_ambient,
                })
                .collect();

            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect_3d,
                Scene3DCallback {
                    view_proj: vp_3d,
                    camera_pos: self.camera_3d.eye(),
                    streamline_draws: streamline_draws.clone(),
                    show_streamlines: self.show_streamlines,
                    volume_draws,
                    slice_visible: self.slice_visible,
                    surface_draws,
                    bundle_draws,
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
                                if let Some(nf) = self.nifti_files.first() {
                                    let vol = &nf.volume;
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
                        if response.clicked() && self.has_streamlines() {
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
                                    for trx in &mut self.trx_files {
                                        trx.sphere_query_result =
                                            Some(trx.data.query_sphere(self.sphere_center, self.sphere_radius));
                                        trx.indices_dirty = true;
                                    }
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

                        let slice_volume_draws: Vec<VolumeDrawInfo> = self.nifti_files.iter()
                            .filter(|n| n.visible)
                            .map(|n| VolumeDrawInfo {
                                file_id: n.id,
                                window_center: n.window_center,
                                window_width: n.window_width,
                                colormap: n.colormap.as_u32(),
                                opacity: n.opacity,
                            })
                            .collect();

                        let slice_streamline_draws: Vec<StreamlineDrawInfo> = self.trx_files.iter()
                            .filter(|t| t.visible)
                            .map(|t| StreamlineDrawInfo {
                                file_id: t.id,
                                render_style: t.render_style,
                                tube_radius: t.tube_radius,
                                slab_half_width: t.slab_half_width,
                            })
                            .collect();

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            SliceViewCallback {
                                view_proj: vp_slice,
                                quad_index: i,
                                bind_group_index: i + 1,
                                volume_draws: slice_volume_draws,
                                streamline_draws: slice_streamline_draws,
                                show_streamlines: self.show_streamlines,
                                slab_axis,
                                slab_min: slice_pos,
                                slab_max: slice_pos,
                            },
                        ));

                        // Draw crosshairs showing the other two slice positions.
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
