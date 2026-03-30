mod callbacks;
mod dirty_updates;
mod file_loading;
mod helpers;
mod state;
mod ui;

use callbacks::{Scene3DCallback, SliceViewCallback, StreamlineDrawInfo, VolumeDrawInfo};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;

use crate::data::bundle_mesh::{BundleMesh, BundleMeshColorStrategy, build_bundle_mesh};
use crate::data::loaded_files::{
    BundleMeshColorMode, BundleMeshSource, FileId, LoadedNifti, LoadedTrx,
};
use crate::data::orientation_field::{
    BoundaryContactField, BoundaryGlyphParams, StreamlineSet,
};
use crate::data::trx_data::{RenderStyle, build_tube_vertices_from_data};
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::{MeshDrawStyle, MeshResources};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis};
use crate::renderer::streamline_renderer::AllStreamlineResources;

use state::{
    ImportDialogState, LoadedGiftiSurface, PendingFileLoad, SceneLightingParams, WorkerMessage,
    WorkerReceiver, WorkerSender,
};
pub(crate) use state::SceneLightingParams as AppSceneLightingParams;

const TUBE_REBUILD_DEBOUNCE_MS: u64 = 350;

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
    pub(crate) surface_projection_dirty: bool,
    pub(crate) show_streamlines: bool,
    pub(crate) slice_visible: [bool; 3],
    pub(crate) slice_world_offsets: [f32; 3],
    pub(crate) sphere_query_active: bool,
    pub(crate) sphere_center: Vec3,
    pub(crate) sphere_radius: f32,
    pub(crate) worker_tx: WorkerSender,
    pub(crate) worker_rx: WorkerReceiver,
    pub(crate) next_job_id: u64,
    pub(crate) pending_file_loads: Vec<PendingFileLoad>,
    pub(crate) import_dialog: ImportDialogState,
    pub(crate) surface_query_dirty: bool,
    pub(crate) surface_query_pending: bool,
    pub(crate) surface_query_revision: u64,
    pub(crate) surface_projection_pending: bool,
    pub(crate) surface_projection_revision: u64,
    pub(crate) show_boundary_glyphs: bool,
    pub(crate) boundary_glyph_params: BoundaryGlyphParams,
    pub(crate) scene_lighting: SceneLightingParams,
    pub(crate) boundary_field: Option<Arc<BoundaryContactField>>,
    pub(crate) boundary_field_pending: Option<std::sync::mpsc::Receiver<Option<BoundaryContactField>>>,
    pub(crate) boundary_field_dirty_at: Option<std::time::Instant>,
    pub(crate) boundary_field_revision: u64,
}

impl TrxViewerApp {
    fn bundle_mesh_color_strategy(
        trx: &LoadedTrx,
        boundary_field: Option<&BoundaryContactField>,
    ) -> (BundleMeshColorStrategy, bool) {
        match trx.bundle_mesh_color_mode {
            BundleMeshColorMode::StreamlineColor => (BundleMeshColorStrategy::SampledRgb, false),
            BundleMeshColorMode::DirectionOrientation => {
                (BundleMeshColorStrategy::DominantOrientation, false)
            }
            BundleMeshColorMode::BoundaryField => (
                if boundary_field.is_some() {
                    BundleMeshColorStrategy::BoundaryField
                } else {
                    BundleMeshColorStrategy::SampledRgb
                },
                true,
            ),
            BundleMeshColorMode::Constant => {
                (BundleMeshColorStrategy::Constant(trx.uniform_color), false)
            }
        }
    }

    fn schedule_tube_rebuild(&mut self, trx_idx: usize) {
        let trx = &mut self.trx_files[trx_idx];
        trx.tube_build_revision = trx.tube_build_revision.wrapping_add(1);
        trx.tube_mesh_dirty_at = Some(std::time::Instant::now());
    }

    pub(crate) fn mark_boundary_field_dirty(&mut self) {
        if self.trx_files.iter().any(|t| t.include_in_boundary_glyphs) {
            self.boundary_field_dirty_at = Some(std::time::Instant::now());
        } else {
            self.boundary_field_dirty_at = None;
            self.boundary_field = None;
        }
    }

    fn global_boundary_streamline_sets(&self) -> Vec<StreamlineSet> {
        self.trx_files
            .iter()
            .filter(|trx| trx.include_in_boundary_glyphs)
            .filter_map(|trx| {
                let selected = trx.data.filtered_streamline_indices(
                    &trx.group_visible,
                    trx.max_streamlines,
                    &trx.streamline_order,
                    trx.sphere_query_result.as_ref(),
                    self.surface_query_result.as_ref(),
                );
                if selected.is_empty() {
                    return None;
                }
                let (positions, _colors, offsets) = trx.data.selected_tube_data(&selected);
                if positions.len() < 2 {
                    return None;
                }
                Some(StreamlineSet { positions, offsets })
            })
            .collect()
    }

    fn poll_worker_messages(&mut self, frame: &mut eframe::Frame) {
        while let Ok(message) = self.worker_rx.try_recv() {
            match message {
                WorkerMessage::TrxLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_trx(path, data, rs);
                            }
                        }
                        Err(err) => self.error_msg = Some(format!("Failed to load TRX: {err}")),
                    }
                }
                WorkerMessage::NiftiLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(vol) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_nifti(path, vol, rs);
                            }
                        }
                        Err(err) => self.error_msg = Some(format!("Failed to load NIfTI: {err}")),
                    }
                }
                WorkerMessage::GiftiLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(surface) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_gifti_surface(path, surface, rs);
                            }
                        }
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to load GIFTI surface: {err}"))
                        }
                    }
                }
                WorkerMessage::ImportedTractogramLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(data) => {
                            if let Some(rs) = frame.wgpu_render_state() {
                                self.apply_loaded_trx(path, data, rs);
                            }
                        }
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to import tractogram: {err}"))
                        }
                    }
                }
                WorkerMessage::StreamlineIndicesBuilt { file_id, indices } => {
                    if let Some(trx_idx) = self.trx_files.iter().position(|trx| trx.id == file_id) {
                        self.trx_files[trx_idx].index_build_pending = false;
                        if self.trx_files[trx_idx].indices_dirty {
                            continue;
                        }
                        if let Some(rs) = frame.wgpu_render_state() {
                            let mut renderer = rs.renderer.write();
                            if let Some(all) =
                                renderer.callback_resources.get_mut::<AllStreamlineResources>()
                            {
                                if let Some((_, sr)) =
                                    all.entries.iter_mut().find(|(fid, _)| *fid == file_id)
                                {
                                    sr.update_indices(&rs.device, &rs.queue, &indices);
                                }
                            }
                        }
                        if self.trx_files[trx_idx].render_style == RenderStyle::Tubes {
                            self.schedule_tube_rebuild(trx_idx);
                        }
                        self.selection_revision = self.selection_revision.wrapping_add(1);
                        self.surface_projection_dirty = true;
                        if self.trx_files[trx_idx].show_bundle_mesh
                            && self.trx_files[trx_idx].bundle_mesh_source != BundleMeshSource::All
                        {
                            self.trx_files[trx_idx].bundle_mesh_dirty_at =
                                Some(std::time::Instant::now());
                        }
                    }
                }
                WorkerMessage::SurfaceQueryDone { revision, result } => {
                    self.apply_surface_query_result(revision, result);
                }
                WorkerMessage::SurfaceProjectionDone { revision, outputs } => {
                    self.apply_surface_projection_result(frame, revision, outputs);
                }
            }
        }
    }

    fn kick_streamline_index_jobs(&mut self) {
        for trx in &mut self.trx_files {
            if !trx.indices_dirty || trx.index_build_pending {
                continue;
            }

            let file_id = trx.id;
            let data = trx.data.clone();
            let group_visible = trx.group_visible.clone();
            let max_streamlines = trx.max_streamlines;
            let streamline_order = trx.streamline_order.clone();
            let sphere_query_result = trx.sphere_query_result.clone();
            let surface_query_result = self.surface_query_result.clone();
            let tx = self.worker_tx.clone();

            trx.indices_dirty = false;
            trx.index_build_pending = true;

            std::thread::spawn(move || {
                let indices = data.build_index_buffer(
                    &group_visible,
                    max_streamlines,
                    &streamline_order,
                    sphere_query_result.as_ref(),
                    surface_query_result.as_ref(),
                );
                let _ = tx.send(WorkerMessage::StreamlineIndicesBuilt { file_id, indices });
            });
        }
    }

    fn active_task_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .pending_file_loads
            .iter()
            .map(|job| job.label.clone())
            .collect();

        for trx in &self.trx_files {
            if trx.index_build_pending || trx.indices_dirty {
                labels.push(format!("Updating visible streamlines for {}", trx.name));
            }
            if trx.tube_mesh_pending.is_some() || trx.tube_mesh_dirty_at.is_some() {
                labels.push(format!("Building streamtubes for {}", trx.name));
            }
            if trx.bundle_mesh_pending.is_some() || trx.bundle_mesh_dirty_at.is_some() {
                labels.push(format!("Building bundle mesh for {}", trx.name));
            }
        }
        if self.boundary_field_pending.is_some() || self.boundary_field_dirty_at.is_some() {
            labels.push("Building boundary glyphs".to_string());
        }

        if self.surface_query_pending || self.surface_query_dirty {
            labels.push("Recomputing surface depth filter".to_string());
        }
        if self.surface_projection_pending || self.surface_projection_dirty {
            labels.push("Updating surface projection map".to_string());
        }

        labels
    }

    fn draw_activity_overlay(&self, ctx: &egui::Context) {
        let tasks = self.active_task_labels();
        if tasks.is_empty() {
            return;
        }

        egui::Area::new("activity_overlay".into())
            .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_min_width(280.0);
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new());
                        ui.label("Working");
                    });
                    ui.separator();
                    for task in tasks {
                        ui.small(task);
                    }
                });
            });
    }

    /// Returns true if any TRX file is loaded.
    pub(crate) fn has_streamlines(&self) -> bool {
        !self.trx_files.is_empty()
    }

    fn open_import_dialog(&mut self, path: Option<PathBuf>) {
        let detected = path
            .as_ref()
            .and_then(|selected| trx_rs::detect_format(selected).ok());
        self.import_dialog.open_with_path(path, detected);
    }

    pub fn new(
        cc: &eframe::CreationContext<'_>,
        trx_path: Option<String>,
        nifti_path: Option<String>,
    ) -> Self {
        let (worker_tx, worker_rx) = std::sync::mpsc::channel();
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
            surface_projection_dirty: false,
            show_streamlines: true,
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            sphere_query_active: false,
            sphere_center: Vec3::ZERO,
            sphere_radius: 10.0,
            worker_tx,
            worker_rx,
            next_job_id: 1,
            pending_file_loads: Vec::new(),
            import_dialog: ImportDialogState::default(),
            surface_query_dirty: false,
            surface_query_pending: false,
            surface_query_revision: 0,
            surface_projection_pending: false,
            surface_projection_revision: 0,
            show_boundary_glyphs: false,
            boundary_glyph_params: BoundaryGlyphParams::default(),
            scene_lighting: SceneLightingParams::default(),
            boundary_field: None,
            boundary_field_pending: None,
            boundary_field_dirty_at: None,
            boundary_field_revision: 0,
        };

        if cc.wgpu_render_state.is_some() {
            if let Some(path) = trx_path {
                app.begin_load_trx(PathBuf::from(path));
            }
            if let Some(path) = nifti_path {
                app.begin_load_nifti(PathBuf::from(path));
            }
        }

        app
    }
}

impl eframe::App for TrxViewerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.poll_worker_messages(frame);

        // Update slice positions if dirty
        if self.slices_dirty {
            if let Some(rs) = frame.wgpu_render_state() {
                let renderer = rs.renderer.read();
                if let Some(all) = renderer.callback_resources.get::<AllSliceResources>() {
                    for (file_id, sr) in &all.entries {
                        if let Some(nf) = self.nifti_files.iter().find(|n| n.id == *file_id) {
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Axial,
                                self.slice_indices[0],
                                &nf.volume,
                            );
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Coronal,
                                self.slice_indices[1],
                                &nf.volume,
                            );
                            sr.update_slice(
                                &rs.queue,
                                SliceAxis::Sagittal,
                                self.slice_indices[2],
                                &nf.volume,
                            );
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
                // Tube meshes embed colors — schedule a rebuild so the new colors take effect.
                if self.trx_files[i].render_style == RenderStyle::Tubes {
                    self.schedule_tube_rebuild(i);
                }
                // Bundle mesh vertex colors come from the color buffer; trigger rebuild.
                if self.trx_files[i].show_bundle_mesh {
                    self.trx_files[i].bundle_mesh_dirty_at = Some(std::time::Instant::now());
                }
                self.trx_files[i].colors_dirty = false;
            }

        }

        self.kick_streamline_index_jobs();
        self.kick_surface_query_job();
        self.kick_surface_projection_job();

        if let Some(t) = self.boundary_field_dirty_at {
            if t.elapsed() >= std::time::Duration::from_millis(150)
                && self.boundary_field_pending.is_none()
            {
                self.boundary_field_dirty_at = None;
                let params = self.boundary_glyph_params.clone();
                let sets = self.global_boundary_streamline_sets();
                if sets.is_empty() {
                    self.boundary_field = None;
                    if let Some(rs) = frame.wgpu_render_state() {
                        let mut renderer = rs.renderer.write();
                        if let Some(gr) = renderer.callback_resources.get_mut::<GlyphResources>() {
                            gr.clear();
                        }
                    }
                } else {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let egui_ctx = ctx.clone();
                    self.boundary_field_revision = self.boundary_field_revision.wrapping_add(1);
                    self.boundary_field_pending = Some(rx);
                    std::thread::spawn(move || {
                        let field = BoundaryContactField::build_from_streamlines(&sets, &params);
                        let _ = tx.send(field);
                        egui_ctx.request_repaint();
                    });
                }
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        if let Some(rx) = &self.boundary_field_pending {
            if let Ok(field) = rx.try_recv() {
                self.boundary_field_pending = None;
                self.boundary_field = field.map(Arc::new);
                if let Some(rs) = frame.wgpu_render_state() {
                    let mut renderer = rs.renderer.write();
                    if let Some(gr) = renderer.callback_resources.get_mut::<GlyphResources>() {
                        if let Some(field) = &self.boundary_field {
                            gr.set_field(
                                &rs.device,
                                field.clone(),
                                self.boundary_glyph_params.scale,
                                self.boundary_glyph_params.min_contacts,
                            );
                        } else {
                            gr.clear();
                        }
                    }
                }
                for trx in &mut self.trx_files {
                    if trx.show_bundle_mesh
                        && matches!(trx.bundle_mesh_color_mode, BundleMeshColorMode::BoundaryField)
                    {
                        trx.bundle_mesh_dirty_at = Some(std::time::Instant::now());
                    }
                }
            }
        }

        // ── Streamtube mesh: check debounce + receive completed geometry ─────
        for i in 0..self.trx_files.len() {
            if let Some(t) = self.trx_files[i].tube_mesh_dirty_at {
                if t.elapsed() >= std::time::Duration::from_millis(TUBE_REBUILD_DEBOUNCE_MS)
                    && self.trx_files[i].tube_mesh_pending.is_none()
                    && self.trx_files[i].render_style == RenderStyle::Tubes
                {
                    self.trx_files[i].tube_mesh_dirty_at = None;

                    let (revision, radius, sides, payload) = {
                        let trx = &self.trx_files[i];
                        let selected = trx.data.filtered_streamline_indices(
                            &trx.group_visible,
                            trx.max_streamlines,
                            &trx.streamline_order,
                            trx.sphere_query_result.as_ref(),
                            self.surface_query_result.as_ref(),
                        );
                        let payload = trx.data.selected_tube_data(&selected);
                        (
                            trx.tube_build_revision,
                            trx.tube_radius,
                            trx.tube_sides,
                            payload,
                        )
                    };

                    let (tx, rx) = std::sync::mpsc::channel();
                    let egui_ctx = ctx.clone();
                    self.trx_files[i].tube_mesh_pending = Some(rx);

                    std::thread::spawn(move || {
                        let (positions, colors, offsets) = payload;
                        let (vertices, indices) =
                            build_tube_vertices_from_data(&positions, &colors, &offsets, radius, sides);
                        let _ = tx.send((revision, vertices, indices));
                        egui_ctx.request_repaint();
                    });
                }
                ctx.request_repaint_after(std::time::Duration::from_millis(50));
            }

            if let Some(rx) = &self.trx_files[i].tube_mesh_pending {
                if let Ok((revision, vertices, indices)) = rx.try_recv() {
                    self.trx_files[i].tube_mesh_pending = None;
                    if revision == self.trx_files[i].tube_build_revision
                        && self.trx_files[i].render_style == RenderStyle::Tubes
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            let mut renderer = rs.renderer.write();
                            if let Some(all) =
                                renderer.callback_resources.get_mut::<AllStreamlineResources>()
                            {
                                let id = self.trx_files[i].id;
                                if let Some((_, sr)) =
                                    all.entries.iter_mut().find(|(fid, _)| *fid == id)
                                {
                                    sr.update_tube_geometry(&rs.device, &vertices, &indices);
                                }
                            }
                        }
                    } else if self.trx_files[i].render_style == RenderStyle::Tubes {
                        self.trx_files[i].tube_mesh_dirty_at = Some(std::time::Instant::now());
                    }
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
                        let threshold = self.trx_files[i].bundle_mesh_threshold;
                        let smooth = self.trx_files[i].bundle_mesh_smooth;
                        let source = self.trx_files[i].bundle_mesh_source;
                        let boundary_field = self.boundary_field.clone();
                        let (color_strategy, uses_boundary_field) = Self::bundle_mesh_color_strategy(
                            &self.trx_files[i],
                            boundary_field.as_deref(),
                        );
                        let (tx, rx) = std::sync::mpsc::channel();
                        let egui_ctx = ctx.clone();

                        // Prepare thread payload from immutable borrows before assigning rx.
                        enum BundlePayload {
                            All(Vec<[f32; 3]>, Vec<[f32; 4]>),
                            Selection(Vec<[f32; 3]>, Vec<[f32; 4]>),
                            PerGroup(Vec<(String, Vec<[f32; 3]>, Vec<[f32; 4]>)>),
                        }
                        let payload = {
                            let trx = &self.trx_files[i];
                            match source {
                                BundleMeshSource::All => BundlePayload::All(
                                    trx.data.positions.clone(),
                                    trx.data.colors.clone(),
                                ),
                                BundleMeshSource::Selection => {
                                    let selected = trx.data.filtered_streamline_indices(
                                        &trx.group_visible,
                                        trx.max_streamlines,
                                        &trx.streamline_order,
                                        trx.sphere_query_result.as_ref(),
                                        self.surface_query_result.as_ref(),
                                    );
                                    let (positions, colors) =
                                        trx.data.selected_vertex_data(&selected);
                                    BundlePayload::Selection(positions, colors)
                                }
                                BundleMeshSource::PerGroup => {
                                    let group_data: Vec<(String, Vec<[f32; 3]>, Vec<[f32; 4]>)> =
                                        trx.data
                                            .groups
                                            .iter()
                                            .enumerate()
                                            .filter(|(gi, _)| {
                                                trx.group_visible.get(*gi).copied().unwrap_or(true)
                                            })
                                            .map(|(_, (name, members))| {
                                                let (pos, col) =
                                                    trx.data.selected_vertex_data(members);
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
                                    if let Some(m) = build_bundle_mesh(
                                        &positions,
                                        &colors,
                                        voxel_size,
                                        threshold,
                                        smooth,
                                        color_strategy,
                                        boundary_field.as_deref(),
                                    ) {
                                        out.push((m, "all".to_string()));
                                    }
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                            BundlePayload::Selection(positions, colors) => {
                                std::thread::spawn(move || {
                                    let mut out = Vec::new();
                                    if let Some(m) = build_bundle_mesh(
                                        &positions,
                                        &colors,
                                        voxel_size,
                                        threshold,
                                        smooth,
                                        color_strategy,
                                        boundary_field.as_deref(),
                                    ) {
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
                                            build_bundle_mesh(
                                                &pos,
                                                &col,
                                                voxel_size,
                                                threshold,
                                                smooth,
                                                color_strategy,
                                                boundary_field.as_deref(),
                                            )
                                            .map(|m| (m, name))
                                        })
                                        .collect();
                                    let _ = tx.send(out);
                                    egui_ctx.request_repaint();
                                });
                            }
                        }
                        let _ = uses_boundary_field;
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
                                self.trx_files[i].bundle_meshes_cpu =
                                    meshes.into_iter().map(|(m, _)| m).collect();
                            }
                        }
                    }
                }
            }
        }

        // ── Handle dropped files ──
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in &dropped {
            if let Some(path) = &file.path {
                match helpers::classify_dropped_path(path) {
                    helpers::DroppedPathKind::OpenTrx => {
                        self.begin_load_trx(path.clone());
                    }
                    helpers::DroppedPathKind::ImportTractogram(_) => {
                        self.open_import_dialog(Some(path.clone()));
                    }
                    helpers::DroppedPathKind::OpenNifti => {
                        self.begin_load_nifti(path.clone());
                    }
                    helpers::DroppedPathKind::OpenGifti => {
                        self.begin_load_gifti_surface(path.clone());
                    }
                    helpers::DroppedPathKind::Unsupported => {
                        self.error_msg =
                            Some(format!("Unknown or unsupported file type: {}", path.display()));
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
                self.begin_load_trx(path);
            }
        }
        if menu_action.import_tractogram {
            self.open_import_dialog(None);
        }
        if menu_action.open_nifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                .pick_file()
            {
                self.begin_load_nifti(path);
            }
        }
        if menu_action.open_gifti {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GIFTI files", &["gii", "gifti"])
                .pick_file()
            {
                self.begin_load_gifti_surface(path);
            }
        }
        let import_action = ui::import_dialog::show_import_dialog(ctx, &mut self.import_dialog);
        if import_action.import_requested {
            if self
                .import_dialog
                .detected_format
                .is_some_and(|format| matches!(format, trx_rs::Format::Tck | trx_rs::Format::Vtk | trx_rs::Format::TinyTrack))
                && self.import_dialog.source_path.is_some()
            {
                let import_state = self.import_dialog.clone();
                self.begin_import_tractogram(&import_state);
                self.import_dialog.close();
            } else {
                self.import_dialog.error_msg =
                    Some("Choose a supported foreign tractogram file to import.".to_string());
            }
        }

        // ── Sidebar ──
        self.show_sidebar(ctx, frame);

        // ── Main content: 4 viewports ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.trx_files.is_empty()
                && self.nifti_files.is_empty()
                && self.gifti_surfaces.is_empty()
            {
                ui.centered_and_justified(|ui| {
                    ui.label("Drop files here or use File > Open to begin.");
                });
                return;
            }

            let available = ui.available_size();
            let any_slice_visible = self.slice_visible.iter().any(|&v| v);
            let top_height = if any_slice_visible {
                (available.y * 0.6).max(100.0)
            } else {
                available.y
            };
            let bottom_height = if any_slice_visible {
                (available.y - top_height - ui.spacing().item_spacing.y).max(50.0)
            } else {
                0.0
            };

            // ── 3D Viewport ──
            let (rect_3d, response_3d) = ui.allocate_exact_size(
                egui::vec2(available.x, top_height),
                egui::Sense::click_and_drag(),
            );

            if response_3d.dragged_by(egui::PointerButton::Primary) {
                let delta = ui.input(|i| i.pointer.delta());
                self.camera_3d.handle_drag(delta.x, delta.y);
            }
            if response_3d.dragged_by(egui::PointerButton::Middle) {
                let delta = ui.input(|i| i.pointer.delta());
                self.camera_3d.pan_screen(delta.x, delta.y);
            }
            if response_3d.dragged_by(egui::PointerButton::Secondary) {
                let delta = ui.input(|i| i.pointer.delta());
                self.camera_3d.handle_zoom_drag(delta.x, delta.y);
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
                            gloss: s.surface_gloss,
                            map_opacity: s.map_opacity,
                            map_threshold: s.map_threshold,
                        },
                    )
                })
                .collect();

            let volume_draws: Vec<VolumeDrawInfo> = self
                .nifti_files
                .iter()
                .filter(|n| n.visible)
                .map(|n| VolumeDrawInfo {
                    file_id: n.id,
                    window_center: n.window_center,
                    window_width: n.window_width,
                    colormap: n.colormap.as_u32(),
                    opacity: n.opacity,
                })
                .collect();

            let streamline_draws: Vec<StreamlineDrawInfo> = self
                .trx_files
                .iter()
                .filter(|t| t.visible)
                .map(|t| StreamlineDrawInfo {
                    file_id: t.id,
                    render_style: t.render_style,
                    tube_radius: t.tube_radius,
                    slab_half_width: t.slab_half_width,
                })
                .collect();

            let bundle_draws: Vec<callbacks::BundleDrawInfo> = self
                .trx_files
                .iter()
                .filter(|t| t.show_bundle_mesh)
                .map(|t| callbacks::BundleDrawInfo {
                    file_id: t.id,
                    opacity: t.bundle_mesh_opacity,
                })
                .collect();
            let glyph_visible = self.show_boundary_glyphs && self.boundary_field.is_some();

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
                    show_boundary_glyphs: glyph_visible,
                    boundary_glyph_color_mode: self.boundary_glyph_params.color_mode,
                    boundary_glyph_draw_step: self.boundary_glyph_params.density_3d_step as u32,
                    scene_lighting: self.scene_lighting,
                },
            ));

            // Draw 3D orientation axes
            self.draw_3d_axes(ui, rect_3d, vp_3d);

            // Draw 3D sphere indicator
            if self.sphere_query_active {
                self.draw_sphere_3d(ui, rect_3d, vp_3d);
            }

            // ── Bottom row: visible slice views ──
            let visible_slice_indices: Vec<usize> = self
                .slice_visible
                .iter()
                .enumerate()
                .filter_map(|(i, visible)| visible.then_some(i))
                .collect();
            if !visible_slice_indices.is_empty() {
                let count = visible_slice_indices.len() as f32;
                let spacing = ui.spacing().item_spacing.x * (count - 1.0).max(0.0);
                let slice_width = ((available.x - spacing) / count).max(10.0);
                let slice_height = (bottom_height - ui.spacing().item_spacing.y - 18.0).max(10.0);

                ui.horizontal(|ui| {
                    let axis_names = ["Axial", "Coronal", "Sagittal"];
                    let axis_labels = ["Z", "Y", "X"];
                    for &i in &visible_slice_indices {
                    ui.vertical(|ui| {
                        let pos_mm = self.slice_world_position(i);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.slice_visible[i], "");
                            ui.label(format!(
                                "{} ({} = {:.1} mm)",
                                axis_names[i], axis_labels[i], pos_mm
                            ));
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
                                    self.slice_world_offsets[i] = (self.slice_world_offsets[i]
                                        + delta)
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
                                    let world = self.slice_cameras[i]
                                        .screen_to_world(pos, rect, aspect_q, sp);
                                    self.sphere_center = world;
                                    self.sphere_query_active = true;
                                    for trx in &mut self.trx_files {
                                        trx.sphere_query_result =
                                            Some(trx.data.query_sphere(
                                                self.sphere_center,
                                                self.sphere_radius,
                                            ));
                                        trx.indices_dirty = true;
                                    }
                                    self.mark_boundary_field_dirty();
                                }
                            }
                        }

                        let aspect = rect.width() / rect.height().max(1.0);
                        let slice_pos = self.slice_world_position(i);
                        let vp_slice = self.slice_cameras[i].view_projection(aspect, slice_pos);
                        let glyph_slab_half_width = self
                            .boundary_field
                            .as_ref()
                            .map(|field| 0.5 * field.grid.voxel_size_mm)
                            .unwrap_or(0.0);

                        // Slab axis: axial=Z(2), coronal=Y(1), sagittal=X(0)
                        let slab_axis = match i {
                            0 => 2u32, // axial → Z
                            1 => 1u32, // coronal → Y
                            _ => 0u32, // sagittal → X
                        };

                        let slice_volume_draws: Vec<VolumeDrawInfo> = self
                            .nifti_files
                            .iter()
                            .filter(|n| n.visible)
                            .map(|n| VolumeDrawInfo {
                                file_id: n.id,
                                window_center: n.window_center,
                                window_width: n.window_width,
                                colormap: n.colormap.as_u32(),
                                opacity: n.opacity,
                            })
                            .collect();

                        let slice_streamline_draws: Vec<StreamlineDrawInfo> = self
                            .trx_files
                            .iter()
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
                                slab_min: slice_pos - glyph_slab_half_width,
                                slab_max: slice_pos + glyph_slab_half_width,
                                show_boundary_glyphs: glyph_visible,
                                boundary_glyph_color_mode: self.boundary_glyph_params.color_mode,
                                boundary_glyph_draw_step: self.boundary_glyph_params.slice_density_step as u32,
                                scene_lighting: self.scene_lighting,
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
            }
        });

        self.draw_activity_overlay(ctx);
        if !self.active_task_labels().is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}
