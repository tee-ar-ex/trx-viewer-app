mod callbacks;
mod file_loading;
mod helpers;
mod state;
mod ui;
mod workflow;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;

use crate::data::loaded_files::{FileId, LoadedNifti, LoadedTrx};
use crate::data::orientation_field::BoundaryContactField;
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis};

pub(crate) use state::SceneLightingParams as AppSceneLightingParams;
use state::{
    ImportDialogState, LoadedGiftiSurface, PendingFileLoad, SceneLightingParams, WorkerMessage,
    WorkerReceiver, WorkerSender,
};
use workflow::{
    LoadedParcellation, StreamlineDisplayRuntime, WorkflowDocument, WorkflowExecutionCache,
    WorkflowJobKind, WorkflowJobMessage, WorkflowRuntime, WorkflowSelection, default_document,
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
    pub(crate) status_msg: Option<String>,
    pub(crate) gifti_surfaces: Vec<LoadedGiftiSurface>,
    pub(crate) parcellations: Vec<LoadedParcellation>,
    pub(crate) slice_visible: [bool; 3],
    pub(crate) slice_world_offsets: [f32; 3],
    pub(crate) worker_tx: WorkerSender,
    pub(crate) worker_rx: WorkerReceiver,
    pub(crate) next_job_id: u64,
    pub(crate) pending_file_loads: Vec<PendingFileLoad>,
    pub(crate) import_dialog: ImportDialogState,
    pub(crate) scene_lighting: SceneLightingParams,
    pub(crate) boundary_field: Option<Arc<BoundaryContactField>>,
    pub(crate) boundary_field_revision: u64,
    pub(crate) workflow_document: WorkflowDocument,
    pub(crate) workflow_runtime: WorkflowRuntime,
    pub(crate) workflow_selection: Option<WorkflowSelection>,
    pub(crate) workflow_graph_focus_request: Option<egui::Rect>,
    pub(crate) workflow_display_runtimes:
        HashMap<workflow::WorkflowNodeUuid, StreamlineDisplayRuntime>,
    pub(crate) next_workflow_draw_id: FileId,
    pub(crate) workflow_project_path: Option<PathBuf>,
    pub(crate) workflow_node_feedback: HashMap<workflow::WorkflowNodeUuid, String>,
    pub(crate) workflow_execution_cache: WorkflowExecutionCache,
    pub(crate) workflow_run_expensive_requested: bool,
    pub(crate) workflow_run_session_active: bool,
    pub(crate) workflow_job_tx: std::sync::mpsc::Sender<WorkflowJobMessage>,
    pub(crate) workflow_job_rx: std::sync::mpsc::Receiver<WorkflowJobMessage>,
    pub(crate) workflow_jobs_in_flight: HashMap<workflow::WorkflowNodeUuid, (WorkflowJobKind, u64)>,
}

impl TrxViewerApp {
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
                WorkerMessage::ParcellationLoaded {
                    job_id,
                    path,
                    result,
                } => {
                    self.pending_file_loads.retain(|job| job.job_id != job_id);
                    match result {
                        Ok(source) => self.apply_loaded_parcellation(path, source),
                        Err(err) => {
                            self.error_msg = Some(format!("Failed to load parcellation: {err}"))
                        }
                    }
                }
                WorkerMessage::ImportedStreamlinesLoaded {
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
                            self.error_msg = Some(format!("Failed to import streamlines: {err}"))
                        }
                    }
                }
            }
        }
    }

    fn active_task_labels(&self) -> Vec<String> {
        let mut labels: Vec<String> = self
            .pending_file_loads
            .iter()
            .map(|job| job.label.clone())
            .collect();

        for (node_uuid, (kind, _)) in &self.workflow_jobs_in_flight {
            let label = self
                .workflow_document
                .graph
                .node_ids()
                .find(|(_, node)| node.uuid == *node_uuid)
                .map(|(_, node)| node.label.clone())
                .filter(|label| !label.is_empty())
                .unwrap_or_else(|| workflow::workflow_job_kind_title(*kind).to_string());
            labels.push(format!(
                "Building {} for {}",
                workflow::workflow_job_kind_title(*kind),
                label
            ));
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
        let (workflow_job_tx, workflow_job_rx) = std::sync::mpsc::channel();
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
            status_msg: None,
            gifti_surfaces: Vec::new(),
            parcellations: Vec::new(),
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            worker_tx,
            worker_rx,
            next_job_id: 1,
            pending_file_loads: Vec::new(),
            import_dialog: ImportDialogState::default(),
            scene_lighting: SceneLightingParams::default(),
            boundary_field: None,
            boundary_field_revision: 0,
            workflow_document: default_document(),
            workflow_runtime: WorkflowRuntime::default(),
            workflow_selection: None,
            workflow_graph_focus_request: None,
            workflow_display_runtimes: HashMap::new(),
            next_workflow_draw_id: 1_000_000,
            workflow_project_path: None,
            workflow_node_feedback: HashMap::new(),
            workflow_execution_cache: WorkflowExecutionCache::default(),
            workflow_run_expensive_requested: false,
            workflow_run_session_active: false,
            workflow_job_tx,
            workflow_job_rx,
            workflow_jobs_in_flight: HashMap::new(),
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
        self.poll_workflow_job_messages();

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
                        self.error_msg = Some(format!(
                            "Unknown or unsupported file type: {}",
                            path.display()
                        ));
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
        if menu_action.new_workflow_project {
            self.new_workflow_project(frame);
        }
        if menu_action.open_workflow_project {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Workflow Project", &["json"])
                .pick_file()
            {
                self.open_workflow_project(path, frame);
            }
        }
        if menu_action.save_workflow_project {
            self.save_workflow_project(false);
        }
        if menu_action.save_workflow_project_as {
            self.save_workflow_project(true);
        }
        if menu_action.import_streamlines {
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
        if menu_action.open_parcellation {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                .pick_file()
            {
                self.begin_load_parcellation(path);
            }
        }
        let import_action = ui::import_dialog::show_import_dialog(ctx, &mut self.import_dialog);
        if import_action.import_requested {
            if self.import_dialog.detected_format.is_some_and(|format| {
                matches!(
                    format,
                    trx_rs::Format::Tck | trx_rs::Format::Vtk | trx_rs::Format::TinyTrack
                )
            }) && self.import_dialog.source_path.is_some()
            {
                let import_state = self.import_dialog.clone();
                self.begin_import_streamlines(&import_state);
                self.import_dialog.close();
            } else {
                self.import_dialog.error_msg =
                    Some("Choose a supported foreign streamline file to import.".to_string());
            }
        }

        self.refresh_workflow_runtime();
        self.queue_workflow_jobs();
        self.sync_workflow_resources(frame);
        self.show_workspace(ctx, frame);

        self.draw_activity_overlay(ctx);
        if !self.active_task_labels().is_empty() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
    }
}
