use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use egui::Rect;
use glam::Vec3;

use crate::app::workflow::{
    LoadedParcellation, StreamlineDisplayRuntime, WorkflowDocument, WorkflowExecutionCache,
    WorkflowJobKind, WorkflowJobMessage, WorkflowNodeUuid, WorkflowRuntime, WorkflowSelection,
    default_document,
};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{FileId, LoadedNifti, LoadedTrx, StreamlineBacking};
use crate::data::nifti_data::NiftiVolume;
use crate::data::orientation_field::BoundaryContactField;
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::TrxGpuData;
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::renderer::slice_renderer::SliceAxis;
use trx_rs::Format;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SliceViewKind {
    Axial,
    Coronal,
    Sagittal,
}

impl SliceViewKind {
    pub const ALL: [Self; 3] = [Self::Axial, Self::Coronal, Self::Sagittal];

    pub fn label(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
        }
    }

    pub fn slice_axis_index(self) -> Option<usize> {
        match self {
            Self::Axial => Some(0),
            Self::Coronal => Some(1),
            Self::Sagittal => Some(2),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum View2DMode {
    Slice,
    Ortho,
    Lightbox,
}

impl View2DMode {
    pub const ALL: [Self; 3] = [Self::Slice, Self::Ortho, Self::Lightbox];

    pub fn label(self) -> &'static str {
        match self {
            Self::Slice => "Slice",
            Self::Ortho => "Ortho",
            Self::Lightbox => "Lightbox",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportTarget {
    View3D,
    View2D,
}

impl ExportTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::View3D => "3D View",
            Self::View2D => "2D View",
        }
    }
}

pub struct PendingExportRequest {
    pub target: ExportTarget,
    pub path: PathBuf,
    pub scale: u32,
    pub requested_screenshot: bool,
}

#[derive(Clone)]
pub struct ExportDialogState {
    pub open: bool,
    pub target: ExportTarget,
    pub scale: u32,
}

impl Default for ExportDialogState {
    fn default() -> Self {
        Self {
            open: false,
            target: ExportTarget::View3D,
            scale: 2,
        }
    }
}

pub struct View2DState {
    pub window_open: bool,
    pub mode: View2DMode,
    pub single_view: SliceViewKind,
    pub lightbox_axis: SliceViewKind,
    pub lightbox_rows: usize,
    pub lightbox_cols: usize,
    pub active_axis: usize,
    pub ortho_show_row: bool,
}

impl Default for View2DState {
    fn default() -> Self {
        Self {
            window_open: true,
            mode: View2DMode::Ortho,
            single_view: SliceViewKind::Axial,
            lightbox_axis: SliceViewKind::Axial,
            lightbox_rows: 3,
            lightbox_cols: 4,
            active_axis: 0,
            ortho_show_row: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SceneLightingPreset {
    Flat,
    Soft,
    Studio,
}

impl SceneLightingPreset {
    pub fn label(self) -> &'static str {
        match self {
            Self::Flat => "Flat",
            Self::Soft => "Soft",
            Self::Studio => "Studio",
        }
    }

    pub const ALL: [Self; 3] = [Self::Flat, Self::Soft, Self::Studio];
}

#[derive(Clone, Copy)]
pub struct SceneLightingParams {
    pub preset: SceneLightingPreset,
}

impl SceneLightingParams {
    pub fn ambient_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 1.0,
            SceneLightingPreset::Soft => 0.62,
            SceneLightingPreset::Studio => 0.50,
        }
    }

    pub fn key_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.34,
            SceneLightingPreset::Studio => 0.52,
        }
    }

    pub fn fill_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.24,
            SceneLightingPreset::Studio => 0.30,
        }
    }

    pub fn headlight_mix(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.28,
            SceneLightingPreset::Studio => 0.18,
        }
    }

    pub fn specular_strength(self) -> f32 {
        match self.preset {
            SceneLightingPreset::Flat => 0.0,
            SceneLightingPreset::Soft => 0.14,
            SceneLightingPreset::Studio => 0.26,
        }
    }
}

impl Default for SceneLightingParams {
    fn default() -> Self {
        Self {
            preset: SceneLightingPreset::Soft,
        }
    }
}

pub struct LoadedGiftiSurface {
    pub id: usize,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<GiftiSurfaceData>,
    pub gpu_index: usize,
    pub visible: bool,
    pub opacity: f32,
    pub color: [f32; 3],
    pub outline_color: [f32; 3],
    pub outline_thickness: f32,
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub surface_gloss: f32,
    pub projection_colormap: SurfaceColormap,
    pub auto_range: bool,
    pub range_min: f32,
    pub range_max: f32,
}

pub struct PendingFileLoad {
    pub job_id: u64,
    pub label: String,
}

#[derive(Clone, Default)]
pub struct ImportDialogState {
    pub open: bool,
    pub source_path: Option<PathBuf>,
    pub detected_format: Option<Format>,
    pub reference_path: Option<PathBuf>,
    pub error_msg: Option<String>,
}

impl ImportDialogState {
    pub fn open_with_path(&mut self, path: Option<PathBuf>, format: Option<Format>) {
        self.open = true;
        self.source_path = path;
        self.detected_format = format;
        self.reference_path = None;
        self.error_msg = None;
    }

    pub fn close(&mut self) {
        self.open = false;
        self.error_msg = None;
    }
}

#[derive(Clone, Default)]
pub struct MergeStreamlineRowState {
    pub source_path: Option<PathBuf>,
    pub detected_format: Option<Format>,
    pub reference_path: Option<PathBuf>,
    pub group_name: String,
}

#[derive(Clone, Default)]
pub struct MergeStreamlinesDialogState {
    pub open: bool,
    pub rows: Vec<MergeStreamlineRowState>,
    pub output_path: Option<PathBuf>,
    pub delete_dps: bool,
    pub delete_dpv: bool,
    pub delete_groups: bool,
    pub positions_dtype: Option<FormatlessDType>,
    pub error_msg: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatlessDType {
    Float16,
    Float32,
    Float64,
}

impl FormatlessDType {
    pub const ALL: [Self; 3] = [Self::Float16, Self::Float32, Self::Float64];

    pub fn label(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::Float32 => "float32",
            Self::Float64 => "float64",
        }
    }
}

impl From<FormatlessDType> for trx_rs::DType {
    fn from(value: FormatlessDType) -> Self {
        match value {
            FormatlessDType::Float16 => trx_rs::DType::Float16,
            FormatlessDType::Float32 => trx_rs::DType::Float32,
            FormatlessDType::Float64 => trx_rs::DType::Float64,
        }
    }
}

impl MergeStreamlinesDialogState {
    pub fn open(&mut self) {
        self.open = true;
        self.error_msg = None;
        if self.rows.len() < 2 {
            self.rows.resize_with(2, MergeStreamlineRowState::default);
        }
    }

    pub fn close(&mut self) {
        self.open = false;
        self.error_msg = None;
    }
}

pub struct LoadedStreamlineSource {
    pub data: TrxGpuData,
    pub backing: StreamlineBacking,
}

pub struct LoadedParcellationSource {
    pub data: ParcellationVolume,
    pub label_table_path: Option<PathBuf>,
}

pub struct SceneState {
    pub trx_files: Vec<LoadedTrx>,
    pub nifti_files: Vec<LoadedNifti>,
    pub gifti_surfaces: Vec<LoadedGiftiSurface>,
    pub parcellations: Vec<LoadedParcellation>,
    pub next_file_id: FileId,
}

impl Default for SceneState {
    fn default() -> Self {
        Self {
            trx_files: Vec::new(),
            nifti_files: Vec::new(),
            gifti_surfaces: Vec::new(),
            parcellations: Vec::new(),
            next_file_id: 0,
        }
    }
}

pub struct ViewportState {
    pub camera_3d: OrbitCamera,
    pub slice_cameras: [OrthoSliceCamera; 3],
    pub slice_indices: [usize; 3],
    pub slices_dirty: bool,
    pub volume_center: Vec3,
    pub volume_extent: f32,
    pub slice_visible: [bool; 3],
    pub slice_world_offsets: [f32; 3],
    pub scene_lighting: SceneLightingParams,
    pub boundary_field: Option<Arc<BoundaryContactField>>,
    pub boundary_field_revision: u64,
    pub window_3d_open: bool,
    pub window_3d_size: [f32; 2],
    pub view_2d: View2DState,
    pub window_2d_size: [f32; 2],
    pub export_dialog: ExportDialogState,
    pub pending_export: Option<PendingExportRequest>,
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
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
            slice_visible: [true; 3],
            slice_world_offsets: [0.0; 3],
            scene_lighting: SceneLightingParams::default(),
            boundary_field: None,
            boundary_field_revision: 0,
            window_3d_open: true,
            window_3d_size: [1200.0, 900.0],
            view_2d: View2DState::default(),
            window_2d_size: [1400.0, 900.0],
            export_dialog: ExportDialogState::default(),
            pending_export: None,
        }
    }
}

impl ViewportState {
    fn gifti_axis_bounds(
        &self,
        gifti_surfaces: &[LoadedGiftiSurface],
        axis_index: usize,
    ) -> Option<(f32, f32)> {
        let mut min_pos = f32::INFINITY;
        let mut max_pos = f32::NEG_INFINITY;

        for surface in gifti_surfaces {
            let (surface_min, surface_max) = match axis_index {
                0 => (surface.data.bbox_min.z, surface.data.bbox_max.z),
                1 => (surface.data.bbox_min.y, surface.data.bbox_max.y),
                _ => (surface.data.bbox_min.x, surface.data.bbox_max.x),
            };
            min_pos = min_pos.min(surface_min);
            max_pos = max_pos.max(surface_max);
        }

        if min_pos.is_finite() && max_pos.is_finite() {
            Some((min_pos, max_pos))
        } else {
            None
        }
    }

    pub fn slice_world_position_for_index(
        &self,
        nifti_files: &[LoadedNifti],
        axis_index: usize,
        index: usize,
    ) -> f32 {
        if let Some(nf) = nifti_files.first() {
            let vol = &nf.volume;
            let idx = index as f32;
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

    pub fn slice_world_position(&self, nifti_files: &[LoadedNifti], axis_index: usize) -> f32 {
        self.slice_world_position_for_index(nifti_files, axis_index, self.slice_indices[axis_index])
    }

    pub fn step_slice(
        &mut self,
        nifti_files: &[LoadedNifti],
        gifti_surfaces: &[LoadedGiftiSurface],
        axis_index: usize,
        delta: isize,
    ) -> bool {
        if let Some(nf) = nifti_files.first() {
            let vol = &nf.volume;
            let max_idx = match axis_index {
                0 => vol.dims[2].saturating_sub(1),
                1 => vol.dims[1].saturating_sub(1),
                _ => vol.dims[0].saturating_sub(1),
            };
            let new_idx = (self.slice_indices[axis_index] as isize + delta)
                .clamp(0, max_idx as isize) as usize;
            if new_idx != self.slice_indices[axis_index] {
                self.slice_indices[axis_index] = new_idx;
                self.slices_dirty = true;
                return true;
            }
            return false;
        }

        let Some(field) = self.boundary_field.as_ref() else {
            let Some((min_pos, max_pos)) = self.gifti_axis_bounds(gifti_surfaces, axis_index)
            else {
                return false;
            };
            let span = (max_pos - min_pos).abs();
            let step = (span / 256.0).max(0.5);
            let new_pos = (self.slice_world_offsets[axis_index] + delta as f32 * step)
                .clamp(min_pos, max_pos);
            if (new_pos - self.slice_world_offsets[axis_index]).abs() > f32::EPSILON {
                self.slice_world_offsets[axis_index] = new_pos;
                return true;
            }
            return false;
        };

        let voxel = field.grid.voxel_size_mm.max(0.5);
        let dims = field.grid.dims;
        let min_pos = match axis_index {
            0 => field.grid.origin_ras.z + 0.5 * voxel,
            1 => field.grid.origin_ras.y + 0.5 * voxel,
            _ => field.grid.origin_ras.x + 0.5 * voxel,
        };
        let max_pos = match axis_index {
            0 => field.grid.origin_ras.z + (dims[2] as f32 - 0.5) * voxel,
            1 => field.grid.origin_ras.y + (dims[1] as f32 - 0.5) * voxel,
            _ => field.grid.origin_ras.x + (dims[0] as f32 - 0.5) * voxel,
        };
        let new_pos =
            (self.slice_world_offsets[axis_index] + delta as f32 * voxel).clamp(min_pos, max_pos);
        if (new_pos - self.slice_world_offsets[axis_index]).abs() > f32::EPSILON {
            self.slice_world_offsets[axis_index] = new_pos;
            return true;
        }
        false
    }
}

pub struct WorkflowState {
    pub document: WorkflowDocument,
    pub runtime: WorkflowRuntime,
    pub selection: Option<WorkflowSelection>,
    pub graph_focus_request: Option<Rect>,
    pub display_runtimes: HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    pub next_draw_id: FileId,
    pub project_path: Option<PathBuf>,
    pub node_feedback: HashMap<WorkflowNodeUuid, String>,
    pub execution_cache: WorkflowExecutionCache,
    pub run_expensive_requested: bool,
    pub run_session_active: bool,
    pub job_tx: mpsc::Sender<WorkflowJobMessage>,
    pub job_rx: mpsc::Receiver<WorkflowJobMessage>,
    pub jobs_in_flight: HashMap<WorkflowNodeUuid, (WorkflowJobKind, u64)>,
}

impl WorkflowState {
    pub fn new(
        job_tx: mpsc::Sender<WorkflowJobMessage>,
        job_rx: mpsc::Receiver<WorkflowJobMessage>,
    ) -> Self {
        Self {
            document: default_document(),
            runtime: WorkflowRuntime::default(),
            selection: None,
            graph_focus_request: None,
            display_runtimes: HashMap::new(),
            next_draw_id: 1_000_000,
            project_path: None,
            node_feedback: HashMap::new(),
            execution_cache: WorkflowExecutionCache::default(),
            run_expensive_requested: false,
            run_session_active: false,
            job_tx,
            job_rx,
            jobs_in_flight: HashMap::new(),
        }
    }
}

pub enum WorkerMessage {
    TrxLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    ImportedStreamlinesLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    MergedStreamlinesCreated {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedStreamlineSource, String>,
    },
    NiftiLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<NiftiVolume, String>,
    },
    GiftiLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<GiftiSurfaceData, String>,
    },
    ParcellationLoaded {
        job_id: u64,
        path: PathBuf,
        result: Result<LoadedParcellationSource, String>,
    },
}

pub type WorkerSender = mpsc::Sender<WorkerMessage>;
pub type WorkerReceiver = mpsc::Receiver<WorkerMessage>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_dialog_open_with_path_sets_state() {
        let mut state = ImportDialogState::default();
        let path = PathBuf::from("sample.tck.gz");
        state.open_with_path(Some(path.clone()), Some(Format::Tck));
        assert!(state.open);
        assert_eq!(state.source_path.as_deref(), Some(path.as_path()));
        assert_eq!(state.detected_format, Some(Format::Tck));
        assert!(state.reference_path.is_none());
        assert!(state.error_msg.is_none());
    }

    #[test]
    fn merge_dialog_open_initializes_two_rows() {
        let mut state = MergeStreamlinesDialogState::default();
        state.open();
        assert!(state.open);
        assert_eq!(state.rows.len(), 2);
        assert!(state.output_path.is_none());
        assert!(state.error_msg.is_none());
    }
}
