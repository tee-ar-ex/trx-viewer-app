use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use egui::emath::TSTransform;
use egui::{Pos2, Rect};
use egui_snarl::{
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlViewer},
};
use egui_tiles::{Container, Linear, LinearDir, Tile, Tiles, Tree};
use glam::Vec3;
use petgraph::Directed;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use trx_rs::{AnyTrxFile, ConversionOptions, DType, DataArray, Tractogram, write_tractogram};

use crate::app::state::LoadedGiftiSurface;
use crate::data::bundle_mesh::{BundleMesh, BundleMeshColorStrategy, build_bundle_mesh};
use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{
    FileId, LoadedNifti, LoadedTrx, StreamlineBacking, VolumeColormap,
};
use crate::data::orientation_field::{
    BoundaryContactField, BoundaryGlyphColorMode, BoundaryGlyphNormalization, StreamlineSet,
};
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::{
    ColorMode, RenderStyle, TrxGpuData, build_tube_vertices_from_data, group_name_color,
};
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::MeshResources;
use crate::renderer::mesh_renderer::SurfaceColormap;
use crate::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkflowNodeUuid(pub u64);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WorkflowNode {
    pub uuid: WorkflowNodeUuid,
    pub kind: WorkflowNodeKind,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WorkflowNodeKind {
    StreamlineSource {
        source_id: FileId,
    },
    VolumeSource {
        source_id: FileId,
    },
    SurfaceSource {
        source_id: FileId,
    },
    ParcellationSource {
        source_id: FileId,
    },
    LimitStreamlines {
        limit: usize,
        randomize: bool,
        seed: u64,
    },
    GroupSelect {
        groups_csv: String,
    },
    RandomSubset {
        limit: usize,
        seed: u64,
    },
    SphereQuery {
        center: [f32; 3],
        radius_mm: f32,
    },
    SurfaceDepthQuery {
        depth_mm: f32,
    },
    RemoveDuplicates,
    Merge,
    AddGroupsFromParcellation,
    ParcelSelect {
        labels_csv: String,
    },
    ParcelROI,
    ParcelROA,
    ParcelEnd {
        endpoint_count: usize,
    },
    ParcelLimiting,
    ParcelTerminative,
    ParcelSurfaceBuild,
    ColorByDirection,
    ColorByGroup,
    ColorByDPV {
        field: String,
    },
    ColorByDPS {
        field: String,
    },
    UniformColor {
        color: [f32; 4],
    },
    SurfaceProjectionDensity {
        depth_mm: f32,
    },
    SurfaceProjectionMeanDps {
        depth_mm: f32,
        field: String,
    },
    BundleSurfaceBuild {
        #[serde(default)]
        per_group: bool,
        voxel_size_mm: f32,
        threshold: f32,
        smooth_sigma: f32,
        opacity: f32,
    },
    BoundaryFieldBuild {
        #[serde(default = "default_boundary_field_voxel_size_mm")]
        voxel_size_mm: f32,
        #[serde(default = "default_boundary_field_sphere_lod")]
        sphere_lod: u32,
        #[serde(default = "default_boundary_field_normalization")]
        normalization: BoundaryGlyphNormalization,
    },
    StreamlineDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        render_style: RenderStyle,
        tube_radius_mm: f32,
        tube_sides: u32,
        slab_half_width_mm: f32,
    },
    VolumeDisplay {
        colormap: VolumeColormap,
        opacity: f32,
        window_center: f32,
        window_width: f32,
    },
    SurfaceDisplay {
        color: [f32; 3],
        opacity: f32,
        show_projection_map: bool,
        map_opacity: f32,
        map_threshold: f32,
        gloss: f32,
        projection_colormap: SurfaceColormap,
        range_min: f32,
        range_max: f32,
    },
    BundleSurfaceDisplay {
        #[serde(default)]
        color_mode: BundleSurfaceColorMode,
    },
    BoundaryGlyphDisplay {
        #[serde(default = "default_enabled")]
        enabled: bool,
        #[serde(default = "default_boundary_glyph_scale")]
        scale: f32,
        #[serde(default = "default_boundary_glyph_density_3d_step")]
        density_3d_step: usize,
        #[serde(default = "default_boundary_glyph_slice_density_step")]
        slice_density_step: usize,
        #[serde(default = "default_boundary_glyph_color_mode")]
        color_mode: BoundaryGlyphColorMode,
        #[serde(default = "default_boundary_glyph_min_contacts")]
        min_contacts: u32,
    },
    ParcellationDisplay {
        labels_csv: String,
        opacity: f32,
    },
    SaveStreamlines {
        output_path: String,
    },
}

fn default_enabled() -> bool {
    true
}

fn default_boundary_field_voxel_size_mm() -> f32 {
    3.0
}

fn default_boundary_field_sphere_lod() -> u32 {
    12
}

fn default_boundary_field_normalization() -> BoundaryGlyphNormalization {
    BoundaryGlyphNormalization::GlobalPeak
}

fn default_boundary_glyph_scale() -> f32 {
    2.0
}

fn default_boundary_glyph_density_3d_step() -> usize {
    2
}

fn default_boundary_glyph_slice_density_step() -> usize {
    1
}

fn default_boundary_glyph_color_mode() -> BoundaryGlyphColorMode {
    BoundaryGlyphColorMode::DirectionRgb
}

fn default_boundary_glyph_min_contacts() -> u32 {
    1
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkflowDocument {
    pub next_node_uuid: u64,
    pub graph: Snarl<WorkflowNode>,
    pub workspace: Tree<WorkspacePane>,
    pub assets: Vec<WorkflowAssetDocument>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WorkflowProject {
    pub version: u32,
    pub document: WorkflowDocument,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum WorkflowAssetDocument {
    Streamlines {
        id: FileId,
        path: PathBuf,
        imported: bool,
    },
    Volume {
        id: FileId,
        path: PathBuf,
    },
    Surface {
        id: FileId,
        path: PathBuf,
    },
    Parcellation {
        id: FileId,
        path: PathBuf,
        label_table_path: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkspacePane {
    Assets,
    Preview,
    Graph,
    Inspector,
}

#[derive(Clone, Debug, Default)]
pub struct NodeEvalState {
    pub summary: String,
    pub error: Option<String>,
    pub execution: Option<WorkflowExecutionStatus>,
    pub fingerprint: Option<u64>,
    pub last_result_summary: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowExecutionStatus {
    NeverRun,
    Stale,
    Queued,
    Running,
    Ready,
    Failed(String),
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum BundleSurfaceColorMode {
    #[default]
    Solid,
    BoundaryField,
}

impl BundleSurfaceColorMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Solid => "Solid",
            Self::BoundaryField => "Boundary field",
        }
    }
}

impl WorkflowExecutionStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::NeverRun => "Run required",
            Self::Stale => "Stale",
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Ready => "Ready",
            Self::Failed(_) => "Failed",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExpensiveNodeRunRecord {
    pub current_fingerprint: Option<u64>,
    pub last_success_fingerprint: Option<u64>,
    pub status: WorkflowExecutionStatus,
    pub last_result_summary: Option<String>,
}

impl Default for ExpensiveNodeRunRecord {
    fn default() -> Self {
        Self {
            current_fingerprint: None,
            last_success_fingerprint: None,
            status: WorkflowExecutionStatus::NeverRun,
            last_result_summary: None,
        }
    }
}

#[derive(Clone)]
pub struct CachedSurfaceQuery {
    pub fingerprint: u64,
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub struct CachedDerivedStreamline {
    pub fingerprint: u64,
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
pub struct CachedSurfaceStreamlineMap {
    pub fingerprint: u64,
    pub map: SurfaceStreamlineMap,
}

#[derive(Clone)]
pub struct CachedBoundaryField {
    pub fingerprint: u64,
    pub field: Arc<BoundaryContactField>,
}

#[derive(Clone)]
pub struct CachedTubeGeometry {
    pub fingerprint: u64,
    pub vertices: Vec<crate::data::trx_data::TubeMeshVertex>,
    pub indices: Vec<u32>,
}

#[derive(Clone)]
pub struct CachedBundleSurfaceMeshes {
    pub fingerprint: u64,
    pub meshes: Vec<(BundleMesh, String)>,
}

#[derive(Clone, Default)]
pub struct WorkflowExecutionCache {
    pub node_runs: HashMap<WorkflowNodeUuid, ExpensiveNodeRunRecord>,
    pub derived_streamline_cache: HashMap<WorkflowNodeUuid, CachedDerivedStreamline>,
    pub surface_query_cache: HashMap<WorkflowNodeUuid, CachedSurfaceQuery>,
    pub surface_streamline_map_cache: HashMap<WorkflowNodeUuid, CachedSurfaceStreamlineMap>,
    pub tube_geometry_cache: HashMap<WorkflowNodeUuid, CachedTubeGeometry>,
    pub bundle_surface_mesh_cache: HashMap<WorkflowNodeUuid, CachedBundleSurfaceMeshes>,
    pub boundary_field_cache: HashMap<WorkflowNodeUuid, CachedBoundaryField>,
}

#[derive(Clone)]
pub struct SceneFramePlan {
    pub reactive_streamline_plans: Vec<ReactiveStreamlinePlan>,
    pub surface_query_plans: Vec<SurfaceQueryPlan>,
    pub surface_map_plans: Vec<SurfaceMapPlan>,
    pub streamline_draws: Vec<StreamlineDrawPlan>,
    pub volume_draws: Vec<VolumeDrawPlan>,
    pub surface_draws: Vec<SurfaceDrawPlan>,
    pub bundle_surface_plans: Vec<BundleSurfacePlan>,
    pub bundle_draws: Vec<BundleDrawPlan>,
    pub parcellation_draws: Vec<ParcellationDrawPlan>,
    pub boundary_field_plans: Vec<BoundaryFieldPlan>,
    pub boundary_glyph_draws: Vec<BoundaryGlyphDrawPlan>,
}

impl Default for SceneFramePlan {
    fn default() -> Self {
        Self {
            reactive_streamline_plans: Vec::new(),
            surface_query_plans: Vec::new(),
            surface_map_plans: Vec::new(),
            streamline_draws: Vec::new(),
            volume_draws: Vec::new(),
            surface_draws: Vec::new(),
            bundle_surface_plans: Vec::new(),
            bundle_draws: Vec::new(),
            parcellation_draws: Vec::new(),
            boundary_field_plans: Vec::new(),
            boundary_glyph_draws: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub struct StreamlineFlow {
    pub dataset: Arc<StreamlineDataset>,
    pub selected_streamlines: Arc<Vec<u32>>,
    pub color_mode: ColorMode,
    pub scalar_auto_range: bool,
    pub scalar_range_min: f32,
    pub scalar_range_max: f32,
}

#[derive(Clone)]
pub struct StreamlineDataset {
    pub name: String,
    pub gpu_data: Arc<TrxGpuData>,
    pub backing: StreamlineBacking,
}

#[derive(Clone)]
pub struct ParcellationAsset {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub data: Arc<ParcellationVolume>,
    pub label_table_path: Option<PathBuf>,
    pub visible: bool,
}

#[derive(Clone)]
pub struct LoadedParcellation {
    pub asset: ParcellationAsset,
}

#[derive(Clone)]
pub struct StreamlineDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub draw_id: FileId,
    pub label: String,
    pub visible: bool,
    pub flow: StreamlineFlow,
    pub render_style: RenderStyle,
    pub tube_radius_mm: f32,
    pub tube_sides: u32,
    pub slab_half_width_mm: f32,
}

#[derive(Clone)]
pub struct BundleDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub build_node_uuid: WorkflowNodeUuid,
    pub boundary_field_node_uuid: Option<WorkflowNodeUuid>,
    pub draw_id: FileId,
    pub label: String,
    pub flow: StreamlineFlow,
    pub per_group: bool,
    pub color_mode: BundleSurfaceColorMode,
    pub voxel_size_mm: f32,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct BoundaryGlyphDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub visible: bool,
    pub scale: f32,
    pub density_3d_step: usize,
    pub slice_density_step: usize,
    pub color_mode: BoundaryGlyphColorMode,
    pub min_contacts: u32,
}

#[derive(Clone, Copy)]
pub struct VolumeDrawPlan {
    pub source_id: FileId,
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub window_center: f32,
    pub window_width: f32,
}

#[derive(Clone)]
pub struct SurfaceDrawPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub source_id: FileId,
    pub gpu_index: usize,
    pub color: [f32; 3],
    pub opacity: f32,
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub gloss: f32,
    pub projection_colormap: SurfaceColormap,
    pub range_min: f32,
    pub range_max: f32,
    pub projection_scalars: Option<Vec<f32>>,
}

const DEFAULT_SURFACE_COLOR: [f32; 3] = [0.72, 0.72, 0.72];
const DEFAULT_SURFACE_OPACITY: f32 = 1.0;

#[derive(Clone)]
pub struct ParcellationDrawPlan {
    pub source_id: FileId,
    pub labels: BTreeSet<u32>,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct SurfaceStreamlineMap {
    pub surface_id: FileId,
    pub scalars: Vec<f32>,
    pub range_min: f32,
    pub range_max: f32,
}

#[derive(Clone)]
pub struct SurfaceQueryPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: f32,
}

#[derive(Clone)]
pub struct SurfaceMapPlan {
    pub node_uuid: WorkflowNodeUuid,
    pub flow: StreamlineFlow,
    pub surface_id: FileId,
    pub surface: Arc<GiftiSurfaceData>,
    pub depth_mm: f32,
    pub dps_field: Option<String>,
}

#[derive(Clone, Copy)]
pub enum ReactiveStreamlineOp {
    Merge,
}

#[derive(Clone)]
pub struct ReactiveStreamlinePlan {
    pub node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub op: ReactiveStreamlineOp,
    pub left: StreamlineFlow,
    pub right: StreamlineFlow,
}

#[derive(Clone)]
pub struct BundleSurfacePlan {
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub flow: StreamlineFlow,
    pub per_group: bool,
    pub voxel_size_mm: f32,
    pub threshold: f32,
    pub smooth_sigma: f32,
    pub opacity: f32,
}

#[derive(Clone)]
pub struct BoundaryFieldPlan {
    pub build_node_uuid: WorkflowNodeUuid,
    pub label: String,
    pub flow: StreamlineFlow,
    pub voxel_size_mm: f32,
    pub sphere_lod: u32,
    pub normalization: BoundaryGlyphNormalization,
}

#[derive(Clone)]
pub struct WorkflowRuntime {
    pub scene_plan: SceneFramePlan,
    pub node_state: HashMap<WorkflowNodeUuid, NodeEvalState>,
    pub save_streamline_targets: HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    pub graph_error: Option<String>,
}

impl Default for WorkflowRuntime {
    fn default() -> Self {
        Self {
            scene_plan: SceneFramePlan::default(),
            node_state: HashMap::new(),
            save_streamline_targets: HashMap::new(),
            graph_error: None,
        }
    }
}

#[derive(Clone)]
pub struct SaveStreamlinePlan {
    pub node_uuid: WorkflowNodeUuid,
    pub output_path: PathBuf,
    pub flow: StreamlineFlow,
}

#[derive(Clone)]
struct ParcelSelection {
    source_id: FileId,
    labels: BTreeSet<u32>,
}

#[derive(Clone)]
enum WorkflowValue {
    Streamline(StreamlineFlow),
    Volume(FileId),
    Surface(FileId),
    Parcellation(FileId),
    ParcelSelection(ParcelSelection),
    SurfaceStreamlineMap(SurfaceStreamlineMap),
    BundleSurface(BundleSurfacePlan),
    BoundaryField(BoundaryFieldPlan),
}

#[derive(Clone)]
struct EvaluatedValue {
    value: WorkflowValue,
    stale: bool,
}

impl From<WorkflowValue> for EvaluatedValue {
    fn from(value: WorkflowValue) -> Self {
        Self {
            value,
            stale: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkflowSelection {
    Node(WorkflowNodeUuid),
    Asset(FileId),
}

#[derive(Clone)]
pub struct StreamlineDisplayRuntime {
    pub draw_id: FileId,
    pub fingerprint: u64,
    pub bundle_fingerprint: Option<u64>,
    pub bundle_meshes_cpu: Vec<BundleMesh>,
}

impl Default for StreamlineDisplayRuntime {
    fn default() -> Self {
        Self {
            draw_id: 0,
            fingerprint: 0,
            bundle_fingerprint: None,
            bundle_meshes_cpu: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkflowJobKind {
    ReactiveStreamline,
    SurfaceQuery,
    SurfaceMap,
    TubeGeometry,
    BundleSurface,
    BoundaryField,
}

#[derive(Clone)]
pub enum WorkflowJobPayload {
    ReactiveStreamline(ReactiveStreamlinePlan),
    SurfaceQuery(SurfaceQueryPlan),
    SurfaceMap(SurfaceMapPlan),
    TubeGeometry(StreamlineDrawPlan),
    BundleSurface {
        plan: BundleSurfacePlan,
        color_mode: BundleSurfaceColorMode,
        boundary_field: Option<Arc<BoundaryContactField>>,
    },
    BoundaryField {
        plan: BoundaryFieldPlan,
    },
}

#[derive(Clone)]
pub enum WorkflowJobOutput {
    ReactiveStreamline(StreamlineFlow),
    SurfaceQuery(StreamlineFlow),
    SurfaceMap(SurfaceStreamlineMap),
    TubeGeometry {
        vertices: Vec<crate::data::trx_data::TubeMeshVertex>,
        indices: Vec<u32>,
    },
    BundleSurface {
        meshes: Vec<(BundleMesh, String)>,
    },
    BoundaryField {
        field: Option<Arc<BoundaryContactField>>,
    },
}

#[derive(Clone)]
pub enum WorkflowJobMessage {
    Started {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        kind: WorkflowJobKind,
    },
    Finished {
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        kind: WorkflowJobKind,
        result: Result<WorkflowJobOutput, String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortKind {
    Streamline,
    Volume,
    Surface,
    Parcellation,
    ParcelSelection,
    SurfaceMap,
    BundleSurface,
    BoundaryField,
}

pub struct SeededWorkflowBranch {
    pub bounds: Rect,
    pub primary_selection: WorkflowSelection,
}

pub fn default_workspace_tree() -> Tree<WorkspacePane> {
    let mut tiles = Tiles::default();
    let assets = tiles.insert_pane(WorkspacePane::Assets);
    let preview = tiles.insert_pane(WorkspacePane::Preview);
    let graph = tiles.insert_pane(WorkspacePane::Graph);
    let inspector = tiles.insert_pane(WorkspacePane::Inspector);
    let mut sidebar = Linear::new(LinearDir::Vertical, vec![inspector, assets]);
    sidebar.shares[inspector] = 1.2;
    sidebar.shares[assets] = 0.9;
    let sidebar = tiles.insert_new(Tile::Container(Container::Linear(sidebar)));

    let mut root = Linear::new(LinearDir::Horizontal, vec![sidebar, preview, graph]);
    root.shares[sidebar] = 0.9;
    root.shares[preview] = 2.4;
    root.shares[graph] = 1.25;
    let root = tiles.insert_new(Tile::Container(Container::Linear(root)));
    Tree::new("workflow_workspace", root, tiles)
}

pub fn default_document() -> WorkflowDocument {
    WorkflowDocument {
        next_node_uuid: 1,
        graph: Snarl::new(),
        workspace: default_workspace_tree(),
        assets: Vec::new(),
    }
}

impl Default for WorkflowProject {
    fn default() -> Self {
        Self {
            version: 1,
            document: default_document(),
        }
    }
}

impl WorkflowNodeKind {
    pub fn is_expensive(&self) -> bool {
        match self {
            Self::SurfaceDepthQuery { .. }
            | Self::SurfaceProjectionDensity { .. }
            | Self::SurfaceProjectionMeanDps { .. }
            | Self::BundleSurfaceBuild { .. }
            | Self::BoundaryFieldBuild { .. } => true,
            Self::StreamlineDisplay { render_style, .. } => *render_style == RenderStyle::Tubes,
            _ => false,
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::StreamlineSource { .. } => "Streamline Source",
            Self::VolumeSource { .. } => "Volume Source",
            Self::SurfaceSource { .. } => "Surface Source",
            Self::ParcellationSource { .. } => "Parcellation Source",
            Self::LimitStreamlines { .. } => "Limit Streamlines",
            Self::GroupSelect { .. } => "Group Select",
            Self::RandomSubset { .. } => "Random Subset",
            Self::SphereQuery { .. } => "Sphere Query",
            Self::SurfaceDepthQuery { .. } => "Surface Depth Query",
            Self::RemoveDuplicates => "Remove Duplicates",
            Self::Merge => "Merge",
            Self::AddGroupsFromParcellation => "Add Groups From Parcellation",
            Self::ParcelSelect { .. } => "Parcel Select",
            Self::ParcelROI => "Parcel ROI",
            Self::ParcelROA => "Parcel ROA",
            Self::ParcelEnd { .. } => "Parcel End",
            Self::ParcelLimiting => "Parcel Limiting",
            Self::ParcelTerminative => "Parcel Terminative",
            Self::ParcelSurfaceBuild => "Parcel Surface Build",
            Self::ColorByDirection => "Color By Direction",
            Self::ColorByGroup => "Color By Group",
            Self::ColorByDPV { .. } => "Color By DPV",
            Self::ColorByDPS { .. } => "Color By DPS",
            Self::UniformColor { .. } => "Uniform Color",
            Self::SurfaceProjectionDensity { .. } => "Map Streamlines to Surface",
            Self::SurfaceProjectionMeanDps { .. } => "Map Streamlines to Surface (Mean DPS)",
            Self::BundleSurfaceBuild { .. } => "Bundle Surface Build",
            Self::BoundaryFieldBuild { .. } => "Boundary Field Build",
            Self::StreamlineDisplay { .. } => "Streamline Display",
            Self::VolumeDisplay { .. } => "Volume Display",
            Self::SurfaceDisplay { .. } => "Surface Display",
            Self::BundleSurfaceDisplay { .. } => "Bundle Surface Display",
            Self::BoundaryGlyphDisplay { .. } => "Boundary Glyph Display",
            Self::ParcellationDisplay { .. } => "Parcellation Display",
            Self::SaveStreamlines { .. } => "Save Streamlines",
        }
    }

    fn inputs(&self) -> Vec<PortKind> {
        match self {
            Self::StreamlineSource { .. }
            | Self::VolumeSource { .. }
            | Self::SurfaceSource { .. }
            | Self::ParcellationSource { .. } => Vec::new(),
            Self::LimitStreamlines { .. }
            | Self::GroupSelect { .. }
            | Self::RandomSubset { .. }
            | Self::SphereQuery { .. }
            | Self::RemoveDuplicates
            | Self::ColorByDirection
            | Self::ColorByGroup
            | Self::ColorByDPV { .. }
            | Self::ColorByDPS { .. }
            | Self::UniformColor { .. }
            | Self::StreamlineDisplay { .. }
            | Self::SaveStreamlines { .. } => vec![PortKind::Streamline],
            Self::BundleSurfaceBuild { .. } => vec![PortKind::Streamline],
            Self::BoundaryFieldBuild { .. } => vec![PortKind::Streamline],
            Self::BundleSurfaceDisplay { .. } => {
                vec![PortKind::BundleSurface, PortKind::BoundaryField]
            }
            Self::BoundaryGlyphDisplay { .. } => vec![PortKind::BoundaryField],
            Self::SurfaceDepthQuery { .. } => vec![PortKind::Streamline, PortKind::Surface],
            Self::Merge => {
                vec![PortKind::Streamline, PortKind::Streamline]
            }
            Self::AddGroupsFromParcellation => vec![PortKind::Streamline, PortKind::Parcellation],
            Self::ParcelSelect { .. } | Self::ParcellationDisplay { .. } => {
                vec![PortKind::Parcellation]
            }
            Self::ParcelROI
            | Self::ParcelROA
            | Self::ParcelEnd { .. }
            | Self::ParcelLimiting
            | Self::ParcelTerminative => {
                vec![PortKind::Streamline, PortKind::ParcelSelection]
            }
            Self::ParcelSurfaceBuild => vec![PortKind::ParcelSelection],
            Self::SurfaceProjectionDensity { .. } | Self::SurfaceProjectionMeanDps { .. } => {
                vec![PortKind::Streamline, PortKind::Surface]
            }
            Self::VolumeDisplay { .. } => vec![PortKind::Volume],
            Self::SurfaceDisplay { .. } => vec![PortKind::Surface, PortKind::SurfaceMap],
        }
    }

    fn outputs(&self) -> Vec<PortKind> {
        match self {
            Self::StreamlineSource { .. }
            | Self::LimitStreamlines { .. }
            | Self::GroupSelect { .. }
            | Self::RandomSubset { .. }
            | Self::SphereQuery { .. }
            | Self::SurfaceDepthQuery { .. }
            | Self::RemoveDuplicates
            | Self::Merge
            | Self::AddGroupsFromParcellation
            | Self::ParcelROI
            | Self::ParcelROA
            | Self::ParcelEnd { .. }
            | Self::ParcelLimiting
            | Self::ParcelTerminative
            | Self::ColorByDirection
            | Self::ColorByGroup
            | Self::ColorByDPV { .. }
            | Self::ColorByDPS { .. }
            | Self::UniformColor { .. } => vec![PortKind::Streamline],
            Self::VolumeSource { .. } => vec![PortKind::Volume],
            Self::SurfaceSource { .. } => vec![PortKind::Surface],
            Self::ParcellationSource { .. } => vec![PortKind::Parcellation],
            Self::ParcelSelect { .. } => vec![PortKind::ParcelSelection],
            Self::SurfaceProjectionDensity { .. } | Self::SurfaceProjectionMeanDps { .. } => {
                vec![PortKind::SurfaceMap]
            }
            Self::BundleSurfaceBuild { .. } => vec![PortKind::BundleSurface],
            Self::BoundaryFieldBuild { .. } => vec![PortKind::BoundaryField],
            Self::ParcelSurfaceBuild
            | Self::StreamlineDisplay { .. }
            | Self::VolumeDisplay { .. }
            | Self::SurfaceDisplay { .. }
            | Self::BoundaryGlyphDisplay { .. }
            | Self::ParcellationDisplay { .. }
            | Self::BundleSurfaceDisplay { .. }
            | Self::SaveStreamlines { .. } => Vec::new(),
        }
    }
}

pub fn make_node(document: &mut WorkflowDocument, kind: WorkflowNodeKind, pos: Pos2) -> NodeId {
    let uuid = WorkflowNodeUuid(document.next_node_uuid);
    document.next_node_uuid += 1;
    document.graph.insert_node(
        pos,
        WorkflowNode {
            uuid,
            label: kind.title().to_string(),
            kind,
        },
    )
}

pub fn suggest_asset_branch_origin(document: &WorkflowDocument) -> Pos2 {
    let mut min_x = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for (pos, _) in document.graph.nodes_pos() {
        min_x = min_x.min(pos.x);
        max_y = max_y.max(pos.y);
    }

    if min_x.is_finite() && max_y.is_finite() {
        Pos2::new(min_x, max_y + 170.0)
    } else {
        Pos2::new(40.0, 80.0)
    }
}

fn branch_bounds(document: &WorkflowDocument, nodes: &[NodeId]) -> Rect {
    let mut bounds = Rect::NOTHING;
    for node_id in nodes {
        if let Some(node) = document.graph.get_node_info(*node_id) {
            bounds.extend_with(node.pos);
        }
    }
    if bounds.is_finite() {
        bounds.expand2(egui::vec2(220.0, 120.0))
    } else {
        Rect::from_min_size(Pos2::ZERO, egui::vec2(640.0, 240.0))
    }
}

pub fn add_default_nodes_for_asset(
    document: &mut WorkflowDocument,
    asset: &WorkflowAssetDocument,
    pos: Pos2,
    streamline_limit: Option<usize>,
) -> SeededWorkflowBranch {
    match asset {
        WorkflowAssetDocument::Streamlines { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::StreamlineSource { source_id: *id },
                pos,
            );
            let limit = make_node(
                document,
                WorkflowNodeKind::LimitStreamlines {
                    limit: streamline_limit.unwrap_or(30_000).max(1),
                    randomize: false,
                    seed: 1,
                },
                pos + egui::vec2(240.0, 0.0),
            );
            let display = make_node(
                document,
                WorkflowNodeKind::StreamlineDisplay {
                    enabled: true,
                    render_style: RenderStyle::Flat,
                    tube_radius_mm: 0.4,
                    tube_sides: 8,
                    slab_half_width_mm: 5.0,
                },
                pos + egui::vec2(480.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: limit,
                    input: 0,
                },
            );
            document.graph.connect(
                OutPinId {
                    node: limit,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, limit, display]),
                primary_selection: WorkflowSelection::Node(document.graph[limit].uuid),
            }
        }
        WorkflowAssetDocument::Volume { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::VolumeSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
                pos + egui::vec2(220.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
        WorkflowAssetDocument::Surface { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::SurfaceSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::SurfaceDisplay {
                    color: DEFAULT_SURFACE_COLOR,
                    opacity: DEFAULT_SURFACE_OPACITY,
                    show_projection_map: false,
                    map_opacity: 1.0,
                    map_threshold: 0.0,
                    gloss: 0.45,
                    projection_colormap: SurfaceColormap::Inferno,
                    range_min: 0.0,
                    range_max: 1.0,
                },
                pos + egui::vec2(220.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
        WorkflowAssetDocument::Parcellation { id, .. } => {
            let source = make_node(
                document,
                WorkflowNodeKind::ParcellationSource { source_id: *id },
                pos,
            );
            let display = make_node(
                document,
                WorkflowNodeKind::ParcellationDisplay {
                    labels_csv: String::new(),
                    opacity: 0.9,
                },
                pos + egui::vec2(240.0, 0.0),
            );
            document.graph.connect(
                OutPinId {
                    node: source,
                    output: 0,
                },
                InPinId {
                    node: display,
                    input: 0,
                },
            );
            SeededWorkflowBranch {
                bounds: branch_bounds(document, &[source, display]),
                primary_selection: WorkflowSelection::Node(document.graph[source].uuid),
            }
        }
    }
}

pub struct WorkflowGraphViewer<'a> {
    pub selected: &'a mut Option<WorkflowSelection>,
    pub focus_bounds: &'a mut Option<Rect>,
    pub viewport_rect: Rect,
    pub node_state: &'a HashMap<WorkflowNodeUuid, NodeEvalState>,
}

impl SnarlViewer<WorkflowNode> for WorkflowGraphViewer<'_> {
    fn title(&mut self, node: &WorkflowNode) -> String {
        if node.label.is_empty() {
            node.kind.title().to_string()
        } else {
            node.label.clone()
        }
    }

    fn inputs(&mut self, node: &WorkflowNode) -> usize {
        node.kind.inputs().len()
    }

    fn outputs(&mut self, node: &WorkflowNode) -> usize {
        node.kind.outputs().len()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        *self.selected = Some(WorkflowSelection::Node(snarl[pin.id.node].uuid));
        ui.label(port_name(snarl[pin.id.node].kind.inputs()[pin.id.input]));
        pin_info_for_port(snarl[pin.id.node].kind.inputs()[pin.id.input])
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        *self.selected = Some(WorkflowSelection::Node(snarl[pin.id.node].uuid));
        ui.label(port_name(snarl[pin.id.node].kind.outputs()[pin.id.output]));
        pin_info_for_port(snarl[pin.id.node].kind.outputs()[pin.id.output])
    }

    fn has_body(&mut self, _node: &WorkflowNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) {
        ui.small(match &snarl[node].kind {
            WorkflowNodeKind::LimitStreamlines {
                limit,
                randomize,
                seed,
            } => {
                if *randomize {
                    format!("Keep {limit} streamlines, random seed {seed}")
                } else {
                    format!("Keep first {limit} streamlines")
                }
            }
            WorkflowNodeKind::GroupSelect { groups_csv } => {
                if groups_csv.trim().is_empty() {
                    "All groups".to_string()
                } else {
                    format!("Groups: {groups_csv}")
                }
            }
            WorkflowNodeKind::RandomSubset { limit, seed } => {
                format!("Keep {limit} streamlines, seed {seed}")
            }
            WorkflowNodeKind::StreamlineDisplay { enabled, .. } => {
                if *enabled {
                    "Visible".to_string()
                } else {
                    "Hidden".to_string()
                }
            }
            WorkflowNodeKind::SphereQuery { center, radius_mm } => {
                format!(
                    "center=({:.1}, {:.1}, {:.1}) r={radius_mm:.1} mm",
                    center[0], center[1], center[2]
                )
            }
            WorkflowNodeKind::ParcelSelect { labels_csv } => {
                if labels_csv.trim().is_empty() {
                    "Labels: all nonzero".to_string()
                } else {
                    format!("Labels: {labels_csv}")
                }
            }
            WorkflowNodeKind::SaveStreamlines { output_path } => {
                if output_path.is_empty() {
                    "No output path".to_string()
                } else {
                    output_path.clone()
                }
            }
            other => other.title().to_string(),
        });
        if let Some(state) = self.node_state.get(&snarl[node].uuid)
            && let Some(execution) = &state.execution
        {
            let color = match execution {
                WorkflowExecutionStatus::Ready => egui::Color32::from_rgb(96, 210, 128),
                WorkflowExecutionStatus::NeverRun | WorkflowExecutionStatus::Stale => {
                    egui::Color32::from_rgb(255, 196, 96)
                }
                WorkflowExecutionStatus::Queued => egui::Color32::from_rgb(156, 168, 255),
                WorkflowExecutionStatus::Running => egui::Color32::from_rgb(110, 180, 255),
                WorkflowExecutionStatus::Failed(_) => egui::Color32::from_rgb(255, 112, 112),
            };
            ui.colored_label(color, execution.label());
        }
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<WorkflowNode>) {
        let Some(out_kind) = snarl[from.id.node]
            .kind
            .outputs()
            .get(from.id.output)
            .copied()
        else {
            return;
        };
        let Some(in_kind) = snarl[to.id.node].kind.inputs().get(to.id.input).copied() else {
            return;
        };
        if out_kind != in_kind {
            return;
        }
        for &remote in &to.remotes {
            snarl.disconnect(remote, to.id);
        }
        snarl.connect(from.id, to.id);
    }

    fn has_graph_menu(&mut self, _pos: Pos2, _snarl: &mut Snarl<WorkflowNode>) -> bool {
        true
    }

    fn show_graph_menu(&mut self, pos: Pos2, ui: &mut egui::Ui, snarl: &mut Snarl<WorkflowNode>) {
        ui.menu_button("Streamline Filters", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::LimitStreamlines {
                    limit: 30_000,
                    randomize: false,
                    seed: 1,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::GroupSelect {
                    groups_csv: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::RandomSubset {
                    limit: 10_000,
                    seed: 1,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SphereQuery {
                    center: [0.0, 0.0, 0.0],
                    radius_mm: 10.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceDepthQuery { depth_mm: 2.0 },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::RemoveDuplicates);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::Merge);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::AddGroupsFromParcellation);
        });

        ui.menu_button("Parcellation", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcelSelect {
                    labels_csv: String::new(),
                },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelROI);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelROA);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcelEnd { endpoint_count: 1 },
            );
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelLimiting);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelTerminative);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ParcelSurfaceBuild);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ParcellationDisplay {
                    labels_csv: String::new(),
                    opacity: 0.9,
                },
            );
        });

        ui.menu_button("Styling", |ui| {
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ColorByDirection);
            add_node_button(ui, snarl, pos, WorkflowNodeKind::ColorByGroup);
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ColorByDPV {
                    field: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::ColorByDPS {
                    field: String::new(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::UniformColor {
                    color: [0.95, 0.8, 0.2, 1.0],
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceProjectionDensity { depth_mm: 2.0 },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceProjectionMeanDps {
                    depth_mm: 2.0,
                    field: String::new(),
                },
            );
        });

        ui.menu_button("Rendering", |ui| {
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::StreamlineDisplay {
                    enabled: true,
                    render_style: RenderStyle::Flat,
                    tube_radius_mm: 0.4,
                    tube_sides: 8,
                    slab_half_width_mm: 5.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::VolumeDisplay {
                    colormap: VolumeColormap::Grayscale,
                    opacity: 1.0,
                    window_center: 0.5,
                    window_width: 1.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SurfaceDisplay {
                    color: DEFAULT_SURFACE_COLOR,
                    opacity: DEFAULT_SURFACE_OPACITY,
                    show_projection_map: false,
                    map_opacity: 1.0,
                    map_threshold: 0.0,
                    gloss: 0.45,
                    projection_colormap: SurfaceColormap::Inferno,
                    range_min: 0.0,
                    range_max: 1.0,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BundleSurfaceBuild {
                    per_group: false,
                    voxel_size_mm: 2.0,
                    threshold: 3.0,
                    smooth_sigma: 0.5,
                    opacity: 0.5,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BundleSurfaceDisplay {
                    color_mode: BundleSurfaceColorMode::Solid,
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BoundaryFieldBuild {
                    voxel_size_mm: default_boundary_field_voxel_size_mm(),
                    sphere_lod: default_boundary_field_sphere_lod(),
                    normalization: default_boundary_field_normalization(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::BoundaryGlyphDisplay {
                    enabled: default_enabled(),
                    scale: default_boundary_glyph_scale(),
                    density_3d_step: default_boundary_glyph_density_3d_step(),
                    slice_density_step: default_boundary_glyph_slice_density_step(),
                    color_mode: default_boundary_glyph_color_mode(),
                    min_contacts: default_boundary_glyph_min_contacts(),
                },
            );
            add_node_button(
                ui,
                snarl,
                pos,
                WorkflowNodeKind::SaveStreamlines {
                    output_path: String::new(),
                },
            );
        });
    }

    fn current_transform(&mut self, to_global: &mut TSTransform, _snarl: &mut Snarl<WorkflowNode>) {
        let Some(bounds) = self.focus_bounds.take() else {
            return;
        };

        let padded = bounds.expand2(egui::vec2(180.0, 120.0));
        let size = padded.size();
        let fit_scale_x = if size.x > 1.0 {
            self.viewport_rect.width() / size.x
        } else {
            2.0
        };
        let fit_scale_y = if size.y > 1.0 {
            self.viewport_rect.height() / size.y
        } else {
            2.0
        };
        let scaling = fit_scale_x.min(fit_scale_y).clamp(0.2, 2.0);
        to_global.scaling = scaling;
        to_global.translation =
            self.viewport_rect.center().to_vec2() - padded.center().to_vec2() * scaling;
    }

    fn has_node_menu(&mut self, _node: &WorkflowNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<WorkflowNode>,
    ) {
        if ui.button("Delete").clicked() {
            snarl.remove_node(node);
            ui.close();
        }
    }
}

fn port_name(port: PortKind) -> &'static str {
    match port {
        PortKind::Streamline => "Streamline",
        PortKind::Volume => "Volume",
        PortKind::Surface => "Surface",
        PortKind::Parcellation => "Parcellation",
        PortKind::ParcelSelection => "Parcel Set",
        PortKind::SurfaceMap => "Surface Map",
        PortKind::BundleSurface => "Bundle Surface",
        PortKind::BoundaryField => "Boundary Field",
    }
}

fn add_node_button(
    ui: &mut egui::Ui,
    snarl: &mut Snarl<WorkflowNode>,
    pos: Pos2,
    kind: WorkflowNodeKind,
) {
    if ui.button(kind.title()).clicked() {
        snarl.insert_node(
            pos,
            WorkflowNode {
                uuid: WorkflowNodeUuid(0),
                label: kind.title().to_string(),
                kind,
            },
        );
        ui.close();
    }
}

fn pin_info_for_port(port: PortKind) -> PinInfo {
    let color = match port {
        PortKind::Streamline => egui::Color32::from_rgb(82, 181, 255),
        PortKind::Volume => egui::Color32::from_rgb(255, 177, 79),
        PortKind::Surface => egui::Color32::from_rgb(145, 255, 161),
        PortKind::Parcellation => egui::Color32::from_rgb(255, 108, 145),
        PortKind::ParcelSelection => egui::Color32::from_rgb(255, 217, 79),
        PortKind::SurfaceMap => egui::Color32::from_rgb(214, 139, 255),
        PortKind::BundleSurface => egui::Color32::from_rgb(143, 224, 201),
        PortKind::BoundaryField => egui::Color32::from_rgb(255, 160, 96),
    };
    PinInfo::circle().with_fill(color)
}

pub fn ensure_node_uuids(document: &mut WorkflowDocument) {
    let mut next = document.next_node_uuid.max(1);
    for node_id in document
        .graph
        .node_ids()
        .map(|(node_id, _)| node_id)
        .collect::<Vec<_>>()
    {
        if document.graph[node_id].uuid.0 == 0 {
            document.graph[node_id].uuid = WorkflowNodeUuid(next);
            next += 1;
        } else {
            next = next.max(document.graph[node_id].uuid.0 + 1);
        }
    }
    document.next_node_uuid = next;
}

pub fn evaluate_scene_plan(
    document: &WorkflowDocument,
    streamline_assets: &[LoadedTrx],
    volume_assets: &[LoadedNifti],
    surface_assets: &[LoadedGiftiSurface],
    parcellation_assets: &[LoadedParcellation],
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    execution_cache: &mut WorkflowExecutionCache,
    _run_expensive: bool,
) -> WorkflowRuntime {
    let mut runtime = WorkflowRuntime::default();
    let compiled = compile_graph(document);
    let Ok((order, connections)) = compiled else {
        runtime.graph_error = compiled.err();
        return runtime;
    };

    let streamline_map: HashMap<FileId, &LoadedTrx> = streamline_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let volume_map: HashMap<FileId, &LoadedNifti> = volume_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let surface_map: HashMap<FileId, &LoadedGiftiSurface> = surface_assets
        .iter()
        .map(|asset| (asset.id, asset))
        .collect();
    let parcellation_map: HashMap<FileId, &LoadedParcellation> = parcellation_assets
        .iter()
        .map(|asset| (asset.asset.id, asset))
        .collect();

    let mut values = HashMap::<WorkflowNodeUuid, EvaluatedValue>::new();
    let mut projection_by_surface = HashMap::<FileId, SurfaceStreamlineMap>::new();

    for node_id in order {
        let node = &document.graph[node_id];
        let input_values: Vec<Option<EvaluatedValue>> = node
            .kind
            .inputs()
            .iter()
            .enumerate()
            .map(|(input_idx, _)| {
                connections
                    .get(&(node.uuid, input_idx))
                    .and_then(|remote| values.get(remote).cloned())
            })
            .collect();

        let mut node_state = NodeEvalState {
            summary: node.kind.title().to_string(),
            error: None,
            execution: None,
            fingerprint: None,
            last_result_summary: None,
        };
        let result = evaluate_node(
            node,
            &input_values,
            &streamline_map,
            &volume_map,
            &surface_map,
            &parcellation_map,
            display_ids,
            next_draw_id,
            &mut runtime.scene_plan,
            &mut projection_by_surface,
            &mut runtime.save_streamline_targets,
            execution_cache,
            _run_expensive,
            &mut node_state,
        );

        match result {
            Ok(Some(value)) => {
                if node_state.summary == node.kind.title() {
                    node_state.summary = summarize_value(&value.value);
                }
                values.insert(node.uuid, value);
            }
            Ok(None) => {
                if node_state.summary == node.kind.title() {
                    node_state.summary = runtime
                        .save_streamline_targets
                        .get(&node.uuid)
                        .map(|target| format!("Ready to save to {}", target.output_path.display()))
                        .unwrap_or_else(|| node.kind.title().to_string());
                }
            }
            Err(error) => {
                node_state.summary = node.kind.title().to_string();
                node_state.error = Some(error);
            }
        }

        runtime.node_state.insert(node.uuid, node_state);
    }

    runtime
        .scene_plan
        .surface_draws
        .iter_mut()
        .for_each(|draw| {
            if let Some(projection) = projection_by_surface.get(&draw.source_id) {
                draw.show_projection_map = true;
                draw.range_min = projection.range_min;
                draw.range_max = projection.range_max;
                draw.projection_scalars = Some(projection.scalars.clone());
            }
        });

    runtime
}

fn compile_graph(
    document: &WorkflowDocument,
) -> Result<
    (
        Vec<NodeId>,
        HashMap<(WorkflowNodeUuid, usize), WorkflowNodeUuid>,
    ),
    String,
> {
    let mut graph = StableGraph::<WorkflowNodeUuid, (), Directed>::default();
    let mut uuid_to_graph = HashMap::new();
    let mut uuid_to_node = HashMap::new();

    for (node_id, _) in document.graph.node_ids() {
        let uuid = document.graph[node_id].uuid;
        let graph_idx = graph.add_node(uuid);
        uuid_to_graph.insert(uuid, graph_idx);
        uuid_to_node.insert(uuid, node_id);
    }

    let mut connections = HashMap::new();
    for (out_pin, in_pin) in document.graph.wires() {
        let out_uuid = document.graph[out_pin.node].uuid;
        let in_uuid = document.graph[in_pin.node].uuid;
        graph.add_edge(uuid_to_graph[&out_uuid], uuid_to_graph[&in_uuid], ());
        connections.insert((in_uuid, in_pin.input), out_uuid);
    }

    let ordered =
        toposort(&graph, None).map_err(|_| "Workflow graph contains a cycle".to_string())?;
    let order = ordered
        .into_iter()
        .filter_map(|idx| graph.node_weight(idx).copied())
        .filter_map(|uuid| uuid_to_node.get(&uuid).copied())
        .collect();

    Ok((order, connections))
}

#[allow(clippy::too_many_arguments)]
fn evaluate_node(
    node: &WorkflowNode,
    inputs: &[Option<EvaluatedValue>],
    streamline_assets: &HashMap<FileId, &LoadedTrx>,
    volume_assets: &HashMap<FileId, &LoadedNifti>,
    surface_assets: &HashMap<FileId, &LoadedGiftiSurface>,
    parcellation_assets: &HashMap<FileId, &LoadedParcellation>,
    display_ids: &mut HashMap<WorkflowNodeUuid, StreamlineDisplayRuntime>,
    next_draw_id: &mut FileId,
    scene_plan: &mut SceneFramePlan,
    projection_by_surface: &mut HashMap<FileId, SurfaceStreamlineMap>,
    save_targets: &mut HashMap<WorkflowNodeUuid, SaveStreamlinePlan>,
    execution_cache: &mut WorkflowExecutionCache,
    _run_expensive: bool,
    node_state: &mut NodeEvalState,
) -> Result<Option<EvaluatedValue>, String> {
    match &node.kind {
        WorkflowNodeKind::StreamlineSource { source_id } => {
            let source = streamline_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing streamline source {source_id}"))?;
            let dataset = Arc::new(StreamlineDataset {
                name: source.name.clone(),
                gpu_data: source.data.clone(),
                backing: source.backing.clone().ok_or_else(|| {
                    format!(
                        "Streamline source {} is missing export backing",
                        source.name
                    )
                })?,
            });
            let selected = (0..source.data.nb_streamlines as u32).collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    dataset,
                    selected_streamlines: Arc::new(selected),
                    color_mode: ColorMode::DirectionRgb,
                    scalar_auto_range: true,
                    scalar_range_min: 0.0,
                    scalar_range_max: 1.0,
                })
                .into(),
            ))
        }
        WorkflowNodeKind::VolumeSource { source_id } => {
            volume_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing volume source {source_id}"))?;
            Ok(Some(WorkflowValue::Volume(*source_id).into()))
        }
        WorkflowNodeKind::SurfaceSource { source_id } => {
            surface_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing surface source {source_id}"))?;
            Ok(Some(WorkflowValue::Surface(*source_id).into()))
        }
        WorkflowNodeKind::ParcellationSource { source_id } => {
            parcellation_assets
                .get(source_id)
                .ok_or_else(|| format!("Missing parcellation source {source_id}"))?;
            Ok(Some(WorkflowValue::Parcellation(*source_id).into()))
        }
        WorkflowNodeKind::LimitStreamlines {
            limit,
            randomize,
            seed,
        } => {
            let flow = expect_streamline_input(inputs, "Limit Streamlines")?;
            let mut selected = flow.selected_streamlines.as_ref().clone();
            if *randomize {
                selected.sort_by_key(|index| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    index.hash(&mut hasher);
                    hasher.finish()
                });
            }
            selected.truncate(*limit);
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::GroupSelect { groups_csv } => {
            let flow = expect_streamline_input(inputs, "Group Select")?;
            let labels = parse_csv_set(groups_csv);
            if labels.is_empty() {
                return Ok(Some(WorkflowValue::Streamline(flow).into()));
            }
            let keep: HashSet<u32> = flow
                .dataset
                .gpu_data
                .groups
                .iter()
                .filter(|(name, _)| labels.contains(name))
                .flat_map(|(_, members)| members.iter().copied())
                .collect();
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| keep.contains(index))
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::RandomSubset { limit, seed } => {
            let flow = expect_streamline_input(inputs, "Random Subset")?;
            let mut selected = flow.selected_streamlines.as_ref().clone();
            selected.sort_by_key(|index| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                seed.hash(&mut hasher);
                index.hash(&mut hasher);
                hasher.finish()
            });
            selected.truncate(*limit);
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::SphereQuery { center, radius_mm } => {
            let flow = expect_streamline_input(inputs, "Sphere Query")?;
            let hits = flow
                .dataset
                .gpu_data
                .query_sphere(Vec3::new(center[0], center[1], center[2]), *radius_mm);
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| hits.contains(index))
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::SurfaceDepthQuery { depth_mm } => {
            let flow = expect_streamline_input(inputs, "Surface Depth Query")?;
            let surface_id = expect_surface_input(inputs, "Surface Depth Query")?;
            let fingerprint = workflow_surface_query_fingerprint(&flow, surface_id, *depth_mm);
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_query_plans.push(SurfaceQueryPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_query_cache.get(&node.uuid) {
                node_state.summary =
                    format!("{} streamlines", cache.flow.selected_streamlines.len());
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::Streamline(cache.flow.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::RemoveDuplicates => {
            let flow = expect_streamline_input(inputs, "Remove Duplicates")?;
            let mut seen = HashSet::new();
            let mut keep = Vec::new();
            for &streamline_index in flow.selected_streamlines.iter() {
                let key = streamline_key(flow.dataset.gpu_data.as_ref(), streamline_index as usize);
                if seen.insert(key) {
                    keep.push(streamline_index);
                }
            }
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(keep),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::Merge => {
            let left = expect_streamline_input(inputs, node.kind.title())?;
            let right = match inputs.get(1).cloned().flatten() {
                Some(value) => match value.value {
                    WorkflowValue::Streamline(flow) => flow,
                    _ => {
                        return Err(format!(
                            "{} needs a right streamline input",
                            node.kind.title()
                        ));
                    }
                },
                None => {
                    return Err(format!(
                        "{} needs a right streamline input",
                        node.kind.title()
                    ));
                }
            };
            let plan = ReactiveStreamlinePlan {
                node_uuid: node.uuid,
                label: node.label.clone(),
                op: ReactiveStreamlineOp::Merge,
                left,
                right,
            };
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.reactive_streamline_plans.push(plan);
            if let Some(cache) = execution_cache.derived_streamline_cache.get(&node.uuid) {
                node_state.summary =
                    format!("{} streamlines", cache.flow.selected_streamlines.len());
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::Streamline(cache.flow.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint),
                }));
            }
            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Waiting")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::ParcelSelect { labels_csv } => {
            let source_id = expect_parcellation_input(inputs, "Parcel Select")?;
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let labels = resolve_selected_labels(labels_csv, &parcellation.asset.data);
            Ok(Some(
                WorkflowValue::ParcelSelection(ParcelSelection { source_id, labels }).into(),
            ))
        }
        WorkflowNodeKind::ParcelROI => {
            let flow = expect_streamline_input(inputs, "Parcel ROI")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROI")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROI is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation
                        .asset
                        .data
                        .streamline_hits_labels(points, &parcel_selection.labels)
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelROA => {
            let flow = expect_streamline_input(inputs, "Parcel ROA")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel ROA")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel ROA is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation
                        .asset
                        .data
                        .streamline_avoids_labels(points, &parcel_selection.labels)
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelEnd { endpoint_count } => {
            let flow = expect_streamline_input(inputs, "Parcel End")?;
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel End")?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| "Parcel End is missing its parcellation".to_string())?;
            let selected = flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| {
                    let points = streamline_points(flow.dataset.gpu_data.as_ref(), *index as usize);
                    parcellation.asset.data.streamline_end_hits_labels(
                        points,
                        &parcel_selection.labels,
                        *endpoint_count,
                    )
                })
                .collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    selected_streamlines: Arc::new(selected),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ParcelLimiting | WorkflowNodeKind::ParcelTerminative => {
            let flow = expect_streamline_input(inputs, node.kind.title())?;
            let parcel_selection = expect_parcel_selection_input(inputs, node.kind.title())?;
            let parcellation = parcellation_assets
                .get(&parcel_selection.source_id)
                .ok_or_else(|| format!("{} is missing its parcellation", node.kind.title()))?;
            let tractogram = match node.kind {
                WorkflowNodeKind::ParcelLimiting => crop_flow_to_parcels(
                    &flow,
                    &parcellation.asset.data,
                    &parcel_selection.labels,
                    true,
                )?,
                _ => crop_flow_to_parcels(
                    &flow,
                    &parcellation.asset.data,
                    &parcel_selection.labels,
                    false,
                )?,
            };
            let gpu_data =
                Arc::new(TrxGpuData::from_tractogram(&tractogram).map_err(|err| err.to_string())?);
            let selected = (0..gpu_data.nb_streamlines as u32).collect();
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    dataset: Arc::new(StreamlineDataset {
                        name: node.label.clone(),
                        gpu_data,
                        backing: StreamlineBacking::Derived(Arc::new(tractogram)),
                    }),
                    selected_streamlines: Arc::new(selected),
                    color_mode: flow.color_mode.clone(),
                    scalar_auto_range: true,
                    scalar_range_min: 0.0,
                    scalar_range_max: 1.0,
                })
                .into(),
            ))
        }
        WorkflowNodeKind::AddGroupsFromParcellation => {
            let flow = expect_streamline_input(inputs, "Add Groups From Parcellation")?;
            let source_id = match inputs.get(1).cloned().flatten() {
                Some(value) => match value.value {
                    WorkflowValue::Parcellation(source_id) => source_id,
                    _ => {
                        return Err(
                            "Add Groups From Parcellation needs a parcellation input".to_string()
                        );
                    }
                },
                _ => {
                    return Err(
                        "Add Groups From Parcellation needs a parcellation input".to_string()
                    );
                }
            };
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let grouped = add_groups_from_parcellation(
                node,
                &flow,
                &parcellation.asset.data,
                &parcellation.asset.name,
            )?;
            Ok(Some(WorkflowValue::Streamline(grouped).into()))
        }
        WorkflowNodeKind::ColorByDirection => {
            let flow = expect_streamline_input(inputs, "Color By Direction")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::DirectionRgb,
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByGroup => {
            let flow = expect_streamline_input(inputs, "Color By Group")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Group,
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByDPV { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPV")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dpv(field.clone()),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::ColorByDPS { field } => {
            let flow = expect_streamline_input(inputs, "Color By DPS")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Dps(field.clone()),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::UniformColor { color } => {
            let flow = expect_streamline_input(inputs, "Uniform Color")?;
            Ok(Some(
                WorkflowValue::Streamline(StreamlineFlow {
                    color_mode: ColorMode::Uniform(*color),
                    ..flow
                })
                .into(),
            ))
        }
        WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
            let flow = expect_streamline_input(inputs, "Map Streamlines to Surface")?;
            let surface_id = expect_surface_input(inputs, "Map Streamlines to Surface")?;
            let fingerprint =
                workflow_surface_projection_fingerprint(&flow, surface_id, *depth_mm, None);
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_map_plans.push(SurfaceMapPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
                dps_field: None,
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_streamline_map_cache.get(&node.uuid) {
                projection_by_surface.insert(cache.map.surface_id, cache.map.clone());
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceStreamlineMap(cache.map.clone()));
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::SurfaceStreamlineMap(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
            let flow = expect_streamline_input(inputs, "Map Streamlines to Surface (Mean DPS)")?;
            let surface_id = expect_surface_input(inputs, "Map Streamlines to Surface (Mean DPS)")?;
            let fingerprint =
                workflow_surface_projection_fingerprint(&flow, surface_id, *depth_mm, Some(field));
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let surface = surface_assets
                .get(&surface_id)
                .ok_or_else(|| format!("Missing surface {surface_id}"))?;
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            scene_plan.surface_map_plans.push(SurfaceMapPlan {
                node_uuid: node.uuid,
                flow,
                surface_id,
                surface: surface.data.clone(),
                depth_mm: *depth_mm,
                dps_field: Some(field.clone()),
            });

            sync_node_state_from_run_record(node_state, record);
            if let Some(cache) = execution_cache.surface_streamline_map_cache.get(&node.uuid) {
                projection_by_surface.insert(cache.map.surface_id, cache.map.clone());
                node_state.summary =
                    summarize_value(&WorkflowValue::SurfaceStreamlineMap(cache.map.clone()));
                return Ok(Some(EvaluatedValue {
                    value: WorkflowValue::SurfaceStreamlineMap(cache.map.clone()),
                    stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
                }));
            }

            node_state.summary = node_state
                .execution
                .as_ref()
                .map(|status| status.label())
                .unwrap_or("Run required")
                .to_string();
            Ok(None)
        }
        WorkflowNodeKind::StreamlineDisplay {
            enabled,
            render_style,
            tube_radius_mm,
            tube_sides,
            slab_half_width_mm,
        } => {
            let flow = expect_streamline_input(inputs, "Streamline Display")?;
            let runtime = display_ids.entry(node.uuid).or_insert_with(|| {
                let draw_id = *next_draw_id;
                *next_draw_id += 1;
                StreamlineDisplayRuntime {
                    draw_id,
                    ..Default::default()
                }
            });
            let plan = StreamlineDrawPlan {
                node_uuid: node.uuid,
                draw_id: runtime.draw_id,
                label: node.label.clone(),
                visible: *enabled,
                flow,
                render_style: *render_style,
                tube_radius_mm: *tube_radius_mm,
                tube_sides: *tube_sides,
                slab_half_width_mm: *slab_half_width_mm,
            };
            node_state.summary = if *enabled {
                "Visible".to_string()
            } else {
                "Hidden".to_string()
            };
            if *render_style == RenderStyle::Tubes {
                let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
                let fingerprint = workflow_streamline_fingerprint(&plan);
                let record = execution_cache.node_runs.entry(node.uuid).or_default();
                prime_expensive_record(record, fingerprint);
                sync_node_state_from_run_record(node_state, record);
                if upstream_stale && matches!(record.status, WorkflowExecutionStatus::Ready) {
                    node_state.execution = Some(WorkflowExecutionStatus::Stale);
                }
            } else {
                node_state.execution = None;
            }
            scene_plan.streamline_draws.push(plan);
            Ok(None)
        }
        WorkflowNodeKind::BundleSurfaceBuild {
            per_group,
            voxel_size_mm,
            threshold,
            smooth_sigma,
            opacity,
        } => {
            let flow = expect_streamline_input(inputs, "Bundle Surface Build")?;
            let bundle = BundleSurfacePlan {
                build_node_uuid: node.uuid,
                label: node.label.clone(),
                flow,
                per_group: *per_group,
                voxel_size_mm: *voxel_size_mm,
                threshold: *threshold,
                smooth_sigma: *smooth_sigma,
                opacity: *opacity,
            };
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let fingerprint = workflow_bundle_plan_fingerprint(&bundle);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.bundle_surface_plans.push(bundle.clone());
            Ok(Some(EvaluatedValue {
                value: WorkflowValue::BundleSurface(bundle),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }))
        }
        WorkflowNodeKind::VolumeDisplay {
            colormap,
            opacity,
            window_center,
            window_width,
        } => {
            let source_id = expect_volume_input(inputs, "Volume Display")?;
            let _ = volume_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing volume {source_id}"))?;
            scene_plan.volume_draws.push(VolumeDrawPlan {
                source_id,
                colormap: *colormap,
                opacity: *opacity,
                window_center: *window_center,
                window_width: *window_width,
            });
            Ok(None)
        }
        WorkflowNodeKind::SurfaceDisplay {
            color,
            opacity,
            show_projection_map,
            map_opacity,
            map_threshold,
            gloss,
            projection_colormap,
            range_min,
            range_max,
        } => {
            let source_id = expect_surface_input(inputs, "Surface Display")?;
            let surface = surface_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing surface {source_id}"))?;
            let projection = inputs
                .get(1)
                .and_then(|value| value.as_ref())
                .and_then(|value| {
                    if let WorkflowValue::SurfaceStreamlineMap(value) = &value.value {
                        Some(value.clone())
                    } else {
                        None
                    }
                });
            let projection_enabled = *show_projection_map || projection.is_some();
            let final_range = projection
                .as_ref()
                .map(|p| (p.range_min, p.range_max))
                .unwrap_or((*range_min, *range_max));
            let projection_scalars = projection.as_ref().map(|value| value.scalars.clone());
            projection_by_surface.extend(
                projection
                    .as_ref()
                    .cloned()
                    .into_iter()
                    .map(|projection| (projection.surface_id, projection)),
            );
            scene_plan.surface_draws.push(SurfaceDrawPlan {
                node_uuid: node.uuid,
                source_id,
                gpu_index: surface.gpu_index,
                color: *color,
                opacity: *opacity,
                show_projection_map: projection_enabled,
                map_opacity: *map_opacity,
                map_threshold: *map_threshold,
                gloss: *gloss,
                projection_colormap: *projection_colormap,
                range_min: final_range.0,
                range_max: final_range.1,
                projection_scalars,
            });
            Ok(None)
        }
        WorkflowNodeKind::ParcellationDisplay {
            labels_csv,
            opacity,
        } => {
            let source_id = expect_parcellation_input(inputs, "Parcellation Display")?;
            let parcellation = parcellation_assets
                .get(&source_id)
                .ok_or_else(|| format!("Missing parcellation {source_id}"))?;
            let labels = resolve_selected_labels(labels_csv, &parcellation.asset.data);
            scene_plan.parcellation_draws.push(ParcellationDrawPlan {
                source_id,
                labels,
                opacity: *opacity,
            });
            Ok(None)
        }
        WorkflowNodeKind::BoundaryFieldBuild {
            voxel_size_mm,
            sphere_lod,
            normalization,
        } => {
            let flow = expect_streamline_input(inputs, "Boundary Field Build")?;
            let plan = BoundaryFieldPlan {
                build_node_uuid: node.uuid,
                label: node.label.clone(),
                flow,
                voxel_size_mm: *voxel_size_mm,
                sphere_lod: *sphere_lod,
                normalization: *normalization,
            };
            let upstream_stale = inputs.iter().flatten().any(|value| value.stale);
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, fingerprint);
            sync_node_state_from_run_record(node_state, record);
            scene_plan.boundary_field_plans.push(plan.clone());
            Ok(Some(EvaluatedValue {
                value: WorkflowValue::BoundaryField(plan),
                stale: record.last_success_fingerprint != Some(fingerprint) || upstream_stale,
            }))
        }
        WorkflowNodeKind::SaveStreamlines { output_path } => {
            let flow = expect_streamline_input(inputs, "Save Streamlines")?;
            if output_path.trim().is_empty() {
                return Err("Save Streamlines needs an output path".to_string());
            }
            save_targets.insert(
                node.uuid,
                SaveStreamlinePlan {
                    node_uuid: node.uuid,
                    output_path: PathBuf::from(output_path),
                    flow,
                },
            );
            Ok(None)
        }
        WorkflowNodeKind::BundleSurfaceDisplay { color_mode } => {
            let (bundle, stale) = expect_bundle_surface_input(inputs, "Bundle Surface Display")?;
            let boundary_field = inputs
                .get(1)
                .and_then(|value| value.as_ref())
                .map(|value| expect_boundary_field_input(Some(value), "Bundle Surface Display"))
                .transpose()?;
            let runtime = display_ids.entry(node.uuid).or_insert_with(|| {
                let draw_id = *next_draw_id;
                *next_draw_id += 1;
                StreamlineDisplayRuntime {
                    draw_id,
                    ..Default::default()
                }
            });
            let draw = BundleDrawPlan {
                node_uuid: node.uuid,
                build_node_uuid: bundle.build_node_uuid,
                boundary_field_node_uuid: boundary_field
                    .as_ref()
                    .map(|(plan, _)| plan.build_node_uuid),
                draw_id: runtime.draw_id,
                label: bundle.label,
                flow: bundle.flow,
                per_group: bundle.per_group,
                color_mode: *color_mode,
                voxel_size_mm: bundle.voxel_size_mm,
                threshold: bundle.threshold,
                smooth_sigma: bundle.smooth_sigma,
                opacity: bundle.opacity,
            };
            let boundary_revision = draw.boundary_field_node_uuid.and_then(|uuid| {
                execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.fingerprint)
            });
            let display_fingerprint = workflow_bundle_display_fingerprint(&draw, boundary_revision);
            let record = execution_cache.node_runs.entry(node.uuid).or_default();
            prime_expensive_record(record, display_fingerprint);
            sync_node_state_from_run_record(node_state, record);
            let boundary_stale = boundary_field.as_ref().is_some_and(|(_, stale)| *stale);
            node_state.summary = if stale || boundary_stale {
                format!("Displaying stale bundle surface ({})", color_mode.label())
            } else {
                format!("Displaying bundle surface ({})", color_mode.label())
            };
            scene_plan.bundle_draws.push(draw);
            Ok(None)
        }
        WorkflowNodeKind::BoundaryGlyphDisplay {
            enabled,
            scale,
            density_3d_step,
            slice_density_step,
            color_mode,
            min_contacts,
        } => {
            let (plan, stale) = expect_boundary_field_input(
                inputs.first().and_then(|value| value.as_ref()),
                "Boundary Glyph Display",
            )?;
            let draw = BoundaryGlyphDrawPlan {
                node_uuid: node.uuid,
                build_node_uuid: plan.build_node_uuid,
                label: node.label.clone(),
                visible: *enabled,
                scale: *scale,
                density_3d_step: *density_3d_step,
                slice_density_step: *slice_density_step,
                color_mode: *color_mode,
                min_contacts: *min_contacts,
            };
            node_state.execution = None;
            node_state.summary = if !enabled {
                "Boundary field hidden".to_string()
            } else if stale {
                "Displaying stale boundary field".to_string()
            } else {
                "Displaying boundary field".to_string()
            };
            scene_plan.boundary_glyph_draws.push(draw);
            Ok(None)
        }
        WorkflowNodeKind::ParcelSurfaceBuild => {
            let parcel_selection = expect_parcel_selection_input(inputs, "Parcel Surface Build")?;
            scene_plan.parcellation_draws.push(ParcellationDrawPlan {
                source_id: parcel_selection.source_id,
                labels: parcel_selection.labels,
                opacity: 0.9,
            });
            Ok(None)
        }
    }
}

fn expect_streamline_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<StreamlineFlow, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Streamline(flow),
            ..
        }) => Ok(flow),
        _ => Err(format!("{label} needs a streamline input")),
    }
}

fn expect_surface_input(inputs: &[Option<EvaluatedValue>], label: &str) -> Result<FileId, String> {
    inputs
        .iter()
        .flatten()
        .find_map(|value| {
            if let WorkflowValue::Surface(surface_id) = &value.value {
                Some(*surface_id)
            } else {
                None
            }
        })
        .ok_or_else(|| format!("{label} needs a surface input"))
}

fn expect_bundle_surface_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<(BundleSurfacePlan, bool), String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::BundleSurface(bundle),
            stale,
        }) => Ok((bundle, stale)),
        Some(_) => Err(format!("{label} needs a bundle surface input")),
        None => Err(format!("{label} is missing an input")),
    }
}

fn expect_boundary_field_input(
    input: Option<&EvaluatedValue>,
    label: &str,
) -> Result<(BoundaryFieldPlan, bool), String> {
    match input {
        Some(EvaluatedValue {
            value: WorkflowValue::BoundaryField(plan),
            stale,
        }) => Ok((plan.clone(), *stale)),
        Some(_) => Err(format!("{label} needs a boundary field input")),
        None => Err(format!("{label} is missing an input")),
    }
}

fn expect_volume_input(inputs: &[Option<EvaluatedValue>], label: &str) -> Result<FileId, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Volume(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(format!("{label} needs a volume input")),
    }
}

fn expect_parcellation_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<FileId, String> {
    match inputs.first().cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::Parcellation(source_id),
            ..
        }) => Ok(source_id),
        _ => Err(format!("{label} needs a parcellation input")),
    }
}

fn expect_parcel_selection_input(
    inputs: &[Option<EvaluatedValue>],
    label: &str,
) -> Result<ParcelSelection, String> {
    match inputs.get(1).cloned().flatten() {
        Some(EvaluatedValue {
            value: WorkflowValue::ParcelSelection(selection),
            ..
        }) => Ok(selection),
        _ => Err(format!("{label} needs a parcel selection input")),
    }
}

fn parse_csv_set(csv: &str) -> BTreeSet<String> {
    csv.split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_label_ids(csv: &str) -> BTreeSet<u32> {
    csv.split(',')
        .map(str::trim)
        .filter_map(|value| value.parse::<u32>().ok())
        .collect()
}

fn resolve_selected_labels(csv: &str, parcellation: &ParcellationVolume) -> BTreeSet<u32> {
    let labels = parse_label_ids(csv);
    if !labels.is_empty() {
        return labels;
    }

    let mut resolved = BTreeSet::new();
    for &label in &parcellation.labels {
        if label != 0 {
            resolved.insert(label);
        }
    }
    resolved
}

fn streamline_points(data: &TrxGpuData, streamline_index: usize) -> &[[f32; 3]] {
    let start = data.offsets[streamline_index] as usize;
    let end = data.offsets[streamline_index + 1] as usize;
    &data.positions[start..end]
}

fn streamline_key(data: &TrxGpuData, streamline_index: usize) -> Vec<u8> {
    let points = streamline_points(data, streamline_index);
    let forward = bytemuck::cast_slice(points).to_vec();
    let mut reversed_points = points.to_vec();
    reversed_points.reverse();
    let reverse = bytemuck::cast_slice(reversed_points.as_slice()).to_vec();
    if reverse < forward { reverse } else { forward }
}

fn summarize_value(value: &WorkflowValue) -> String {
    match value {
        WorkflowValue::Streamline(flow) => {
            format!("{} streamlines", flow.selected_streamlines.len())
        }
        WorkflowValue::Volume(_) => "Volume ready".to_string(),
        WorkflowValue::Surface(_) => "Surface ready".to_string(),
        WorkflowValue::Parcellation(_) => "Parcellation ready".to_string(),
        WorkflowValue::ParcelSelection(selection) => {
            format!("{} parcel labels", selection.labels.len())
        }
        WorkflowValue::SurfaceStreamlineMap(projection) => {
            format!(
                "Surface streamline map for surface {}",
                projection.surface_id
            )
        }
        WorkflowValue::BundleSurface(bundle) => {
            if bundle.per_group {
                "Bundle surfaces split by group".to_string()
            } else {
                format!(
                    "Bundle surface from {} streamlines",
                    bundle.flow.selected_streamlines.len()
                )
            }
        }
        WorkflowValue::BoundaryField(plan) => {
            format!(
                "Boundary field from {} streamlines",
                plan.flow.selected_streamlines.len()
            )
        }
    }
}

fn robust_range(values: &[f32]) -> (f32, f32) {
    let mut finite: Vec<f32> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    let lo = finite[((n as f32) * 0.02).floor() as usize].min(finite[n - 1]);
    let hi = finite[((n as f32) * 0.98).floor() as usize].max(lo + 1e-6);
    (lo, hi)
}

fn add_groups_from_parcellation(
    node: &WorkflowNode,
    flow: &StreamlineFlow,
    parcellation: &ParcellationVolume,
    parcellation_name: &str,
) -> Result<StreamlineFlow, String> {
    let mut grouped = subset_tractogram_from_flow(flow)?;
    let prefix = parcellation_name
        .split('.')
        .next()
        .unwrap_or(parcellation_name)
        .trim()
        .to_string();
    let mut label_groups = BTreeMap::<u32, Vec<u32>>::new();

    for (new_index, &streamline_index) in flow.selected_streamlines.iter().enumerate() {
        let mut labels_hit = BTreeSet::new();
        for point in streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index as usize) {
            if let Some(label) = parcellation.sample_label_world(Vec3::from(*point)) {
                if label != 0 {
                    labels_hit.insert(label);
                }
            }
        }
        for label in labels_hit {
            label_groups
                .entry(label)
                .or_default()
                .push(new_index as u32);
        }
    }

    for (label, members) in label_groups {
        if members.is_empty() {
            continue;
        }
        let group_name = format!("{}_{}", prefix, parcellation.label_name(label));
        grouped.insert_group(group_name.clone(), members);
        let color = parcellation.label_color(label);
        let rgb = [[
            (color[0].clamp(0.0, 1.0) * 255.0) as u8,
            (color[1].clamp(0.0, 1.0) * 255.0) as u8,
            (color[2].clamp(0.0, 1.0) * 255.0) as u8,
        ]];
        grouped.insert_dpg(
            group_name,
            "color",
            DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
        );
    }

    let gpu_data = Arc::new(TrxGpuData::from_tractogram(&grouped).map_err(|err| err.to_string())?);
    let selected = (0..gpu_data.nb_streamlines as u32).collect();
    Ok(StreamlineFlow {
        dataset: Arc::new(StreamlineDataset {
            name: node.label.clone(),
            gpu_data,
            backing: StreamlineBacking::Derived(Arc::new(grouped)),
        }),
        selected_streamlines: Arc::new(selected),
        color_mode: flow.color_mode.clone(),
        scalar_auto_range: flow.scalar_auto_range,
        scalar_range_min: flow.scalar_range_min,
        scalar_range_max: flow.scalar_range_max,
    })
}

fn crop_flow_to_parcels(
    flow: &StreamlineFlow,
    parcellation: &ParcellationVolume,
    labels: &BTreeSet<u32>,
    keep_inside: bool,
) -> Result<Tractogram, String> {
    let mut tractogram = Tractogram::new();
    for &streamline_index in flow.selected_streamlines.iter() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), streamline_index as usize);
        let segments = if keep_inside {
            parcellation.crop_streamline_inside(points, labels)
        } else {
            parcellation.crop_streamline_outside(points, labels)
        };
        for segment in segments {
            tractogram
                .push_streamline(&segment)
                .map_err(|err| err.to_string())?;
        }
    }
    Ok(tractogram)
}

fn materialize_merged_streamlines(
    left: &StreamlineFlow,
    right: &StreamlineFlow,
) -> Result<Tractogram, String> {
    let left = subset_tractogram_from_flow(left)?;
    let right = subset_tractogram_from_flow(right)?;
    let mut out = Tractogram::with_header(left.header().clone());

    for streamline in left.streamlines() {
        out.push_streamline(streamline)
            .map_err(|err| err.to_string())?;
    }
    for streamline in right.streamlines() {
        out.push_streamline(streamline)
            .map_err(|err| err.to_string())?;
    }

    Ok(out)
}

fn subset_tractogram_from_flow(flow: &StreamlineFlow) -> Result<Tractogram, String> {
    let header = match &flow.dataset.backing {
        StreamlineBacking::Native(any) => any.header().clone(),
        StreamlineBacking::Imported(tractogram) | StreamlineBacking::Derived(tractogram) => {
            tractogram.header().clone()
        }
    };
    let mut tractogram = Tractogram::with_header(header);
    let mut remap = HashMap::with_capacity(flow.selected_streamlines.len());
    for (new_index, &index) in flow.selected_streamlines.iter().enumerate() {
        let points = streamline_points(flow.dataset.gpu_data.as_ref(), index as usize);
        tractogram
            .push_streamline(points)
            .map_err(|err| err.to_string())?;
        remap.insert(index, new_index as u32);
    }
    for (group_idx, (name, members)) in flow.dataset.gpu_data.groups.iter().enumerate() {
        let remapped: Vec<u32> = members
            .iter()
            .filter_map(|member| remap.get(member).copied())
            .collect();
        if remapped.is_empty() {
            continue;
        }
        tractogram.insert_group(name.clone(), remapped);
        if let Some(Some(color)) = flow.dataset.gpu_data.group_colors.get(group_idx) {
            let rgb = [[
                (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                (color[2].clamp(0.0, 1.0) * 255.0) as u8,
            ]];
            tractogram.insert_dpg(
                name.clone(),
                "color",
                DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
            );
        }
    }
    Ok(tractogram)
}

fn flow_selects_entire_dataset(flow: &StreamlineFlow) -> bool {
    flow.selected_streamlines.len() == flow.dataset.gpu_data.nb_streamlines
        && flow
            .selected_streamlines
            .iter()
            .enumerate()
            .all(|(expected, &actual)| expected == actual as usize)
}

pub fn save_streamline_plan(plan: &SaveStreamlinePlan) -> Result<(), String> {
    if plan.output_path.as_os_str().is_empty() {
        return Err("Save path is empty".to_string());
    }

    if flow_selects_entire_dataset(&plan.flow)
        && matches!(&plan.flow.dataset.backing, StreamlineBacking::Native(_))
        && plan
            .output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("trx"))
    {
        if let StreamlineBacking::Native(any) = &plan.flow.dataset.backing {
            return any.save(&plan.output_path).map_err(|err| err.to_string());
        }
    }

    let tractogram = subset_tractogram_from_flow(&plan.flow)?;
    let header = match &plan.flow.dataset.backing {
        StreamlineBacking::Native(any) => Some(any.header().clone()),
        StreamlineBacking::Imported(tractogram) | StreamlineBacking::Derived(tractogram) => {
            Some(tractogram.header().clone())
        }
    };
    let trx_positions_dtype = match &plan.flow.dataset.backing {
        StreamlineBacking::Native(any) => any.dtype(),
        StreamlineBacking::Imported(_) | StreamlineBacking::Derived(_) => DType::Float32,
    };
    write_tractogram(
        &plan.output_path,
        &tractogram,
        &ConversionOptions {
            header,
            trx_positions_dtype,
        },
    )
    .map_err(|err| err.to_string())
}

pub fn save_workflow_project_to_path(
    document: &WorkflowDocument,
    path: &Path,
) -> Result<(), String> {
    let project = WorkflowProject {
        version: 1,
        document: document.clone(),
    };
    let json = serde_json::to_string_pretty(&project).map_err(|err| err.to_string())?;
    std::fs::write(path, json).map_err(|err| err.to_string())
}

pub fn load_workflow_project_from_path(path: &Path) -> Result<WorkflowProject, String> {
    let contents = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    serde_json::from_str::<WorkflowProject>(&contents)
        .or_else(|_| {
            serde_json::from_str::<WorkflowDocument>(&contents).map(|document| WorkflowProject {
                version: 1,
                document,
            })
        })
        .map_err(|err| err.to_string())
}

fn asset_path_mut(asset: &mut WorkflowAssetDocument) -> &mut PathBuf {
    match asset {
        WorkflowAssetDocument::Streamlines { path, .. }
        | WorkflowAssetDocument::Volume { path, .. }
        | WorkflowAssetDocument::Surface { path, .. }
        | WorkflowAssetDocument::Parcellation { path, .. } => path,
    }
}

fn relativized_document(document: &WorkflowDocument, project_path: &Path) -> WorkflowDocument {
    let mut document = document.clone();
    let Some(base_dir) = project_path.parent() else {
        return document;
    };
    for asset in &mut document.assets {
        let path = asset_path_mut(asset);
        if path.is_absolute()
            && let Ok(relative) = path.strip_prefix(base_dir)
        {
            *path = relative.to_path_buf();
        }
    }
    document
}

fn resolve_document_asset_paths(document: &mut WorkflowDocument, project_path: &Path) {
    let Some(base_dir) = project_path.parent() else {
        return;
    };
    for asset in &mut document.assets {
        let path = asset_path_mut(asset);
        if path.is_relative() {
            *path = base_dir.join(&*path);
        }
    }
}

pub(crate) fn workflow_job_kind_title(kind: WorkflowJobKind) -> &'static str {
    match kind {
        WorkflowJobKind::ReactiveStreamline => "derived streamlines",
        WorkflowJobKind::SurfaceQuery => "surface depth query",
        WorkflowJobKind::SurfaceMap => "surface map",
        WorkflowJobKind::TubeGeometry => "tube geometry",
        WorkflowJobKind::BundleSurface => "bundle surface",
        WorkflowJobKind::BoundaryField => "boundary field",
    }
}

impl super::TrxViewerApp {
    pub(in crate::app) fn poll_workflow_job_messages(&mut self) {
        while let Ok(message) = self.workflow_job_rx.try_recv() {
            match message {
                WorkflowJobMessage::Started {
                    node_uuid,
                    fingerprint,
                    ..
                } => {
                    if let Some(record) =
                        self.workflow_execution_cache.node_runs.get_mut(&node_uuid)
                        && record.current_fingerprint == Some(fingerprint)
                    {
                        record.status = WorkflowExecutionStatus::Running;
                    }
                }
                WorkflowJobMessage::Finished {
                    node_uuid,
                    fingerprint,
                    kind: _,
                    result,
                } => {
                    self.workflow_jobs_in_flight.remove(&node_uuid);
                    let Some(record) = self.workflow_execution_cache.node_runs.get_mut(&node_uuid)
                    else {
                        continue;
                    };
                    if record.current_fingerprint != Some(fingerprint) {
                        continue;
                    }
                    match result {
                        Ok(output) => match output {
                            WorkflowJobOutput::ReactiveStreamline(flow) => {
                                let summary =
                                    format!("{} streamlines", flow.selected_streamlines.len());
                                self.workflow_execution_cache
                                    .derived_streamline_cache
                                    .insert(
                                        node_uuid,
                                        CachedDerivedStreamline { fingerprint, flow },
                                    );
                                mark_expensive_success(record, fingerprint, summary);
                            }
                            WorkflowJobOutput::SurfaceQuery(flow) => {
                                let summary =
                                    format!("{} streamlines", flow.selected_streamlines.len());
                                self.workflow_execution_cache
                                    .surface_query_cache
                                    .insert(node_uuid, CachedSurfaceQuery { fingerprint, flow });
                                mark_expensive_success(record, fingerprint, summary);
                            }
                            WorkflowJobOutput::SurfaceMap(map) => {
                                let summary = format!(
                                    "Surface streamline map for surface {}",
                                    map.surface_id
                                );
                                self.workflow_execution_cache
                                    .surface_streamline_map_cache
                                    .insert(
                                        node_uuid,
                                        CachedSurfaceStreamlineMap { fingerprint, map },
                                    );
                                mark_expensive_success(record, fingerprint, summary);
                            }
                            WorkflowJobOutput::TubeGeometry { vertices, indices } => {
                                self.workflow_execution_cache.tube_geometry_cache.insert(
                                    node_uuid,
                                    CachedTubeGeometry {
                                        fingerprint,
                                        vertices,
                                        indices,
                                    },
                                );
                                mark_expensive_success(
                                    record,
                                    fingerprint,
                                    "Tube geometry ready".to_string(),
                                );
                            }
                            WorkflowJobOutput::BundleSurface { meshes } => {
                                let summary = if meshes.is_empty() {
                                    "Bundle surface is empty".to_string()
                                } else {
                                    format!("{} bundle surface mesh(es)", meshes.len())
                                };
                                self.workflow_execution_cache
                                    .bundle_surface_mesh_cache
                                    .insert(
                                        node_uuid,
                                        CachedBundleSurfaceMeshes {
                                            fingerprint,
                                            meshes,
                                        },
                                    );
                                mark_expensive_success(record, fingerprint, summary);
                            }
                            WorkflowJobOutput::BoundaryField { field } => {
                                if let Some(field) = field {
                                    self.workflow_execution_cache.boundary_field_cache.insert(
                                        node_uuid,
                                        CachedBoundaryField { fingerprint, field },
                                    );
                                    mark_expensive_success(
                                        record,
                                        fingerprint,
                                        "Boundary field ready".to_string(),
                                    );
                                } else {
                                    self.workflow_execution_cache
                                        .boundary_field_cache
                                        .remove(&node_uuid);
                                    mark_expensive_success(
                                        record,
                                        fingerprint,
                                        "Boundary field is empty".to_string(),
                                    );
                                }
                            }
                        },
                        Err(error) => {
                            mark_expensive_failure(record, fingerprint, &error);
                        }
                    }
                }
            }
        }
    }

    fn queue_workflow_job(
        &mut self,
        node_uuid: WorkflowNodeUuid,
        fingerprint: u64,
        kind: WorkflowJobKind,
        payload: WorkflowJobPayload,
    ) {
        if self.workflow_jobs_in_flight.contains_key(&node_uuid) {
            return;
        }
        let Some(record) = self.workflow_execution_cache.node_runs.get_mut(&node_uuid) else {
            return;
        };
        record.current_fingerprint = Some(fingerprint);
        record.status = WorkflowExecutionStatus::Queued;
        self.workflow_jobs_in_flight
            .insert(node_uuid, (kind, fingerprint));
        let tx = self.workflow_job_tx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(WorkflowJobMessage::Started {
                node_uuid,
                fingerprint,
                kind,
            });
            let result = run_workflow_job(payload);
            let _ = tx.send(WorkflowJobMessage::Finished {
                node_uuid,
                fingerprint,
                kind,
                result,
            });
        });
    }

    pub(in crate::app) fn queue_workflow_jobs(&mut self) -> bool {
        for plan in self
            .workflow_runtime
            .scene_plan
            .reactive_streamline_plans
            .clone()
        {
            let fingerprint = workflow_reactive_streamline_fingerprint(&plan);
            if should_queue_expensive_job(
                self.workflow_execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::ReactiveStreamline,
                    WorkflowJobPayload::ReactiveStreamline(plan),
                );
            }
        }

        if !self.workflow_run_expensive_requested && !self.workflow_run_session_active {
            return false;
        }

        let mut queued_any = false;
        self.workflow_run_session_active = true;

        for plan in self.workflow_runtime.scene_plan.surface_query_plans.clone() {
            let fingerprint =
                workflow_surface_query_fingerprint(&plan.flow, plan.surface_id, plan.depth_mm);
            if should_queue_expensive_job(
                self.workflow_execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::SurfaceQuery,
                    WorkflowJobPayload::SurfaceQuery(plan),
                );
                queued_any = true;
            }
        }

        for plan in self.workflow_runtime.scene_plan.surface_map_plans.clone() {
            let fingerprint = workflow_surface_projection_fingerprint(
                &plan.flow,
                plan.surface_id,
                plan.depth_mm,
                plan.dps_field.as_deref(),
            );
            if should_queue_expensive_job(
                self.workflow_execution_cache.node_runs.get(&plan.node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                plan.node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.node_uuid,
                    fingerprint,
                    WorkflowJobKind::SurfaceMap,
                    WorkflowJobPayload::SurfaceMap(plan),
                );
                queued_any = true;
            }
        }

        for draw in self.workflow_runtime.scene_plan.streamline_draws.clone() {
            if draw.render_style != RenderStyle::Tubes {
                continue;
            }
            let fingerprint = workflow_streamline_fingerprint(&draw);
            if should_queue_expensive_job(
                self.workflow_execution_cache.node_runs.get(&draw.node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                draw.node_uuid,
            ) {
                self.queue_workflow_job(
                    draw.node_uuid,
                    fingerprint,
                    WorkflowJobKind::TubeGeometry,
                    WorkflowJobPayload::TubeGeometry(draw),
                );
                queued_any = true;
            }
        }

        for plan in self
            .workflow_runtime
            .scene_plan
            .boundary_field_plans
            .clone()
        {
            let fingerprint = workflow_boundary_plan_fingerprint(&plan);
            if should_queue_expensive_job(
                self.workflow_execution_cache
                    .node_runs
                    .get(&plan.build_node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                plan.build_node_uuid,
            ) {
                self.queue_workflow_job(
                    plan.build_node_uuid,
                    fingerprint,
                    WorkflowJobKind::BoundaryField,
                    WorkflowJobPayload::BoundaryField { plan },
                );
                queued_any = true;
            }
        }

        for plan in self
            .workflow_runtime
            .scene_plan
            .bundle_surface_plans
            .clone()
        {
            let fingerprint = workflow_bundle_plan_fingerprint(&plan);
            let record = self
                .workflow_execution_cache
                .node_runs
                .entry(plan.build_node_uuid)
                .or_default();
            if record.last_success_fingerprint != Some(fingerprint) {
                mark_expensive_success(
                    record,
                    fingerprint,
                    format!(
                        "Bundle surface build for {} streamline(s)",
                        plan.flow.selected_streamlines.len()
                    ),
                );
            }
        }

        for draw in self.workflow_runtime.scene_plan.bundle_draws.clone() {
            let boundary_field = draw.boundary_field_node_uuid.and_then(|uuid| {
                self.workflow_execution_cache
                    .boundary_field_cache
                    .get(&uuid)
                    .map(|cache| cache.field.clone())
            });
            if draw.boundary_field_node_uuid.is_some() && boundary_field.is_none() {
                continue;
            }
            let fingerprint = workflow_bundle_display_fingerprint(
                &draw,
                draw.boundary_field_node_uuid.and_then(|uuid| {
                    self.workflow_execution_cache
                        .boundary_field_cache
                        .get(&uuid)
                        .map(|cache| cache.fingerprint)
                }),
            );
            if should_queue_expensive_job(
                self.workflow_execution_cache.node_runs.get(&draw.node_uuid),
                fingerprint,
                &self.workflow_jobs_in_flight,
                draw.node_uuid,
            ) {
                let plan = BundleSurfacePlan {
                    build_node_uuid: draw.build_node_uuid,
                    label: draw.label.clone(),
                    flow: draw.flow.clone(),
                    per_group: draw.per_group,
                    voxel_size_mm: draw.voxel_size_mm,
                    threshold: draw.threshold,
                    smooth_sigma: draw.smooth_sigma,
                    opacity: draw.opacity,
                };
                self.queue_workflow_job(
                    draw.node_uuid,
                    fingerprint,
                    WorkflowJobKind::BundleSurface,
                    WorkflowJobPayload::BundleSurface {
                        plan,
                        color_mode: draw.color_mode,
                        boundary_field,
                    },
                );
                queued_any = true;
            }
        }

        self.workflow_run_expensive_requested = false;
        if !queued_any && self.workflow_jobs_in_flight.is_empty() {
            self.workflow_run_session_active = false;
        }
        queued_any
    }

    pub(in crate::app) fn refresh_workflow_runtime(&mut self) {
        ensure_node_uuids(&mut self.workflow_document);
        self.workflow_runtime = evaluate_scene_plan(
            &self.workflow_document,
            &self.trx_files,
            &self.nifti_files,
            &self.gifti_surfaces,
            &self.parcellations,
            &mut self.workflow_display_runtimes,
            &mut self.next_workflow_draw_id,
            &mut self.workflow_execution_cache,
            false,
        );
    }

    pub(in crate::app) fn sync_workflow_resources(&mut self, frame: &mut eframe::Frame) {
        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };

        let mut renderer = rs.renderer.write();

        if renderer
            .callback_resources
            .get::<AllStreamlineResources>()
            .is_none()
        {
            renderer.callback_resources.insert(AllStreamlineResources {
                entries: Vec::new(),
            });
        }
        if renderer.callback_resources.get::<MeshResources>().is_none() {
            renderer
                .callback_resources
                .insert(MeshResources::new(&rs.device, rs.target_format));
        }
        if renderer
            .callback_resources
            .get::<GlyphResources>()
            .is_none()
        {
            renderer
                .callback_resources
                .insert(GlyphResources::new(&rs.device, rs.target_format));
        }

        let active_streamline_ids: HashSet<FileId> = self
            .workflow_runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| draw.draw_id)
            .collect();
        let active_bundle_ids: HashSet<FileId> = self
            .workflow_runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| draw.draw_id)
            .collect();
        let workflow_ids: HashSet<FileId> = self
            .workflow_display_runtimes
            .values()
            .map(|runtime| runtime.draw_id)
            .collect();

        if let Some(all) = renderer
            .callback_resources
            .get_mut::<AllStreamlineResources>()
        {
            for draw in &self.workflow_runtime.scene_plan.streamline_draws {
                let fingerprint = workflow_streamline_fingerprint(draw);
                let runtime = self
                    .workflow_display_runtimes
                    .get_mut(&draw.node_uuid)
                    .expect("draw runtime allocated during evaluation");
                let resource_exists = all.entries.iter().any(|(id, _)| *id == draw.draw_id);
                if draw.render_style == RenderStyle::Tubes
                    && !self
                        .workflow_execution_cache
                        .node_runs
                        .get(&draw.node_uuid)
                        .is_some_and(|record| record.last_success_fingerprint == Some(fingerprint))
                {
                    continue;
                }
                if runtime.fingerprint == fingerprint && resource_exists {
                    continue;
                }

                let subset = materialize_flow_gpu(draw.flow.clone());
                let mut resource = StreamlineResources::new(&rs.device, rs.target_format, &subset);
                if draw.render_style == RenderStyle::Tubes {
                    let Some(cache) = self
                        .workflow_execution_cache
                        .tube_geometry_cache
                        .get(&draw.node_uuid)
                        .filter(|cache| cache.fingerprint == fingerprint)
                    else {
                        continue;
                    };
                    resource.update_tube_geometry(&rs.device, &cache.vertices, &cache.indices);
                }

                if let Some(entry) = all.entries.iter_mut().find(|(id, _)| *id == draw.draw_id) {
                    *entry = (draw.draw_id, resource);
                } else {
                    all.entries.push((draw.draw_id, resource));
                }

                runtime.fingerprint = fingerprint;
            }

            all.entries
                .retain(|(id, _)| !workflow_ids.contains(id) || active_streamline_ids.contains(id));
        }

        if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
            for draw in &self.workflow_runtime.scene_plan.surface_draws {
                if let Some(scalars) = &draw.projection_scalars {
                    mesh_resources.update_surface_scalars(&rs.queue, draw.gpu_index, scalars);
                }
            }
        }

        let active_boundary_field_ids: HashSet<WorkflowNodeUuid> = self
            .workflow_runtime
            .scene_plan
            .bundle_draws
            .iter()
            .filter_map(|draw| draw.boundary_field_node_uuid)
            .chain(
                self.workflow_runtime
                    .scene_plan
                    .boundary_glyph_draws
                    .iter()
                    .map(|draw| draw.build_node_uuid),
            )
            .collect();

        self.workflow_execution_cache
            .boundary_field_cache
            .retain(|uuid, _| active_boundary_field_ids.contains(uuid));

        if let Some(glyph_resources) = renderer.callback_resources.get_mut::<GlyphResources>() {
            if let Some(draw) = self
                .workflow_runtime
                .scene_plan
                .boundary_glyph_draws
                .iter()
                .find(|draw| draw.visible)
                .or_else(|| {
                    self.workflow_runtime
                        .scene_plan
                        .boundary_glyph_draws
                        .first()
                })
            {
                if let Some(cache) = self
                    .workflow_execution_cache
                    .boundary_field_cache
                    .get(&draw.build_node_uuid)
                {
                    let boundary_field_changed = self.boundary_field_revision != cache.fingerprint;
                    let field = cache.field.clone();
                    glyph_resources.set_field(
                        &rs.device,
                        field.clone(),
                        draw.scale,
                        draw.min_contacts,
                    );
                    self.boundary_field = Some(field.clone());
                    self.boundary_field_revision = cache.fingerprint;
                    if boundary_field_changed && self.nifti_files.is_empty() {
                        self.reset_slice_view_to_boundary_field(field.as_ref());
                    }
                } else {
                    glyph_resources.clear();
                    self.boundary_field = None;
                    self.boundary_field_revision = 0;
                }
            } else {
                glyph_resources.clear();
                self.boundary_field = None;
                self.boundary_field_revision = 0;
            }
        }

        if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
            for draw in &self.workflow_runtime.scene_plan.bundle_draws {
                let display_fingerprint = workflow_bundle_display_fingerprint(
                    draw,
                    draw.boundary_field_node_uuid.and_then(|uuid| {
                        self.workflow_execution_cache
                            .boundary_field_cache
                            .get(&uuid)
                            .map(|cache| cache.fingerprint)
                    }),
                );
                let runtime = self
                    .workflow_display_runtimes
                    .get_mut(&draw.node_uuid)
                    .expect("bundle runtime allocated during evaluation");
                let Some(cache) = self
                    .workflow_execution_cache
                    .bundle_surface_mesh_cache
                    .get(&draw.node_uuid)
                    .filter(|cache| cache.fingerprint == display_fingerprint)
                else {
                    continue;
                };
                if !self
                    .workflow_execution_cache
                    .node_runs
                    .get(&draw.node_uuid)
                    .is_some_and(|record| {
                        record.last_success_fingerprint == Some(display_fingerprint)
                    })
                {
                    continue;
                }
                if runtime.bundle_fingerprint == Some(display_fingerprint) {
                    continue;
                }
                runtime.bundle_meshes_cpu =
                    cache.meshes.iter().map(|(mesh, _)| mesh.clone()).collect();
                runtime.bundle_fingerprint = Some(display_fingerprint);

                if cache.meshes.is_empty() {
                    mesh_resources.clear_bundle_mesh(draw.draw_id);
                } else {
                    mesh_resources.set_bundle_meshes(draw.draw_id, &rs.device, &cache.meshes);
                }
            }

            for draw_id in workflow_ids
                .iter()
                .copied()
                .filter(|id| !active_bundle_ids.contains(id))
            {
                mesh_resources.clear_bundle_mesh(draw_id);
                if let Some(runtime) = self
                    .workflow_display_runtimes
                    .values_mut()
                    .find(|runtime| runtime.draw_id == draw_id)
                {
                    runtime.bundle_fingerprint = None;
                    runtime.bundle_meshes_cpu.clear();
                }
            }
        }
    }

    pub(in crate::app) fn clear_loaded_scene(&mut self, frame: &mut eframe::Frame) {
        if let Some(rs) = frame.wgpu_render_state() {
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer
                .callback_resources
                .get_mut::<AllStreamlineResources>()
            {
                all.entries.clear();
            }
            if let Some(all) = renderer
                .callback_resources
                .get_mut::<crate::renderer::slice_renderer::AllSliceResources>()
            {
                all.entries.clear();
            }
            if let Some(mesh_resources) = renderer.callback_resources.get_mut::<MeshResources>() {
                for runtime in self.workflow_display_runtimes.values() {
                    mesh_resources.clear_bundle_mesh(runtime.draw_id);
                }
            }
            if let Some(glyph_resources) = renderer.callback_resources.get_mut::<GlyphResources>() {
                glyph_resources.clear();
            }
        }

        self.trx_files.clear();
        self.nifti_files.clear();
        self.gifti_surfaces.clear();
        self.parcellations.clear();
        self.pending_file_loads.clear();
        self.boundary_field = None;
        self.boundary_field_revision = 0;
        self.workflow_runtime = WorkflowRuntime::default();
        self.workflow_execution_cache = WorkflowExecutionCache::default();
        self.workflow_display_runtimes.clear();
        self.workflow_selection = None;
        self.workflow_node_feedback.clear();
        self.workflow_document = default_document();
        self.next_file_id = 0;
        self.next_workflow_draw_id = 1_000_000;
        self.workflow_run_expensive_requested = false;
        self.workflow_run_session_active = false;
        self.workflow_jobs_in_flight.clear();
    }

    pub(in crate::app) fn new_workflow_project(&mut self, frame: &mut eframe::Frame) {
        self.clear_loaded_scene(frame);
        self.workflow_project_path = None;
        self.status_msg = Some("Started a new workflow project.".to_string());
        self.error_msg = None;
    }

    pub(in crate::app) fn save_workflow_project(&mut self, save_as: bool) {
        let target_path = if !save_as {
            self.workflow_project_path.clone()
        } else {
            None
        }
        .or_else(|| {
            rfd::FileDialog::new()
                .add_filter("Workflow Project", &["json"])
                .set_file_name("workflow.json")
                .save_file()
        });

        let Some(target_path) = target_path else {
            return;
        };

        let document = relativized_document(&self.workflow_document, &target_path);
        match save_workflow_project_to_path(&document, &target_path) {
            Ok(()) => {
                self.workflow_project_path = Some(target_path.clone());
                self.status_msg = Some(format!(
                    "Saved workflow project to {}",
                    target_path.display()
                ));
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to save workflow project: {err}"));
            }
        }
    }

    pub(in crate::app) fn open_workflow_project(
        &mut self,
        path: PathBuf,
        frame: &mut eframe::Frame,
    ) {
        if frame.wgpu_render_state().is_none() {
            self.error_msg =
                Some("Cannot open a workflow project before the renderer is ready.".to_string());
            return;
        }

        let mut project = match load_workflow_project_from_path(&path) {
            Ok(project) => project,
            Err(err) => {
                self.error_msg = Some(format!("Failed to read workflow project: {err}"));
                return;
            }
        };
        resolve_document_asset_paths(&mut project.document, &path);

        self.clear_loaded_scene(frame);
        let Some(rs) = frame.wgpu_render_state() else {
            self.error_msg =
                Some("Renderer state disappeared while opening the workflow project.".to_string());
            return;
        };

        for asset in project.document.assets.clone() {
            let load_result: Result<(), String> = match asset {
                WorkflowAssetDocument::Streamlines {
                    id,
                    path: asset_path,
                    imported,
                } => {
                    if imported {
                        trx_rs::read_tractogram(&asset_path, &ConversionOptions::default())
                            .map_err(|err| err.to_string())
                            .and_then(|tractogram| {
                                TrxGpuData::from_tractogram(&tractogram)
                                    .map_err(|err| err.to_string())
                                    .map(|data| crate::app::state::LoadedStreamlineSource {
                                        data,
                                        backing: StreamlineBacking::Imported(Arc::new(tractogram)),
                                    })
                            })
                            .map(|source| {
                                self.apply_loaded_trx_with_options(
                                    asset_path,
                                    source,
                                    rs,
                                    Some(id),
                                    false,
                                );
                            })
                    } else {
                        AnyTrxFile::load(&asset_path)
                            .map_err(|err| err.to_string())
                            .and_then(|any| {
                                TrxGpuData::from_any_trx(&any)
                                    .map_err(|err| err.to_string())
                                    .map(|data| crate::app::state::LoadedStreamlineSource {
                                        data,
                                        backing: StreamlineBacking::Native(Arc::new(any)),
                                    })
                            })
                            .map(|source| {
                                self.apply_loaded_trx_with_options(
                                    asset_path,
                                    source,
                                    rs,
                                    Some(id),
                                    false,
                                );
                            })
                    }
                }
                WorkflowAssetDocument::Volume {
                    id,
                    path: asset_path,
                } => crate::data::nifti_data::NiftiVolume::load(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|volume| {
                        self.apply_loaded_nifti_with_options(
                            asset_path,
                            volume,
                            rs,
                            Some(id),
                            false,
                        );
                    }),
                WorkflowAssetDocument::Surface {
                    id,
                    path: asset_path,
                } => crate::data::gifti_data::GiftiSurfaceData::load(&asset_path)
                    .map_err(|err| err.to_string())
                    .map(|surface| {
                        self.apply_loaded_gifti_surface_with_options(
                            asset_path,
                            surface,
                            rs,
                            Some(id),
                            false,
                        );
                    }),
                WorkflowAssetDocument::Parcellation {
                    id,
                    path: asset_path,
                    label_table_path,
                } => crate::data::parcellation_data::ParcellationVolume::load(
                    &asset_path,
                    label_table_path.as_deref(),
                )
                .map_err(|err| err.to_string())
                .map(|data| {
                    self.apply_loaded_parcellation_with_options(
                        asset_path,
                        crate::app::state::LoadedParcellationSource {
                            data,
                            label_table_path,
                        },
                        Some(id),
                        false,
                    );
                }),
            };

            if let Err(err) = load_result {
                self.error_msg = Some(format!("Failed to load workflow project asset: {err}"));
                return;
            }
        }

        self.workflow_document = project.document;
        ensure_node_uuids(&mut self.workflow_document);
        self.workflow_project_path = Some(path.clone());
        self.status_msg = Some(format!("Opened workflow project {}", path.display()));
        self.error_msg = None;
    }

    pub(in crate::app) fn save_streamline_node(&mut self, node_uuid: WorkflowNodeUuid) {
        let Some(plan) = self
            .workflow_runtime
            .save_streamline_targets
            .get(&node_uuid)
            .cloned()
        else {
            self.error_msg =
                Some("This save node does not have a connected streamline input.".to_string());
            return;
        };

        match save_streamline_plan(&plan) {
            Ok(()) => {
                self.workflow_node_feedback
                    .insert(node_uuid, format!("Saved {}", plan.output_path.display()));
                self.status_msg = Some(format!(
                    "Saved streamlines to {}",
                    plan.output_path.display()
                ));
                self.error_msg = None;
            }
            Err(err) => {
                self.error_msg = Some(format!("Failed to save streamlines: {err}"));
            }
        }
    }
}

fn should_queue_expensive_job(
    record: Option<&ExpensiveNodeRunRecord>,
    fingerprint: u64,
    in_flight: &HashMap<WorkflowNodeUuid, (WorkflowJobKind, u64)>,
    node_uuid: WorkflowNodeUuid,
) -> bool {
    if in_flight
        .get(&node_uuid)
        .is_some_and(|(_, queued_fingerprint)| *queued_fingerprint == fingerprint)
    {
        return false;
    }
    let Some(record) = record else {
        return true;
    };
    record.last_success_fingerprint != Some(fingerprint)
}

fn run_workflow_job(payload: WorkflowJobPayload) -> Result<WorkflowJobOutput, String> {
    match payload {
        WorkflowJobPayload::ReactiveStreamline(plan) => {
            let tractogram = match plan.op {
                ReactiveStreamlineOp::Merge => {
                    materialize_merged_streamlines(&plan.left, &plan.right)?
                }
            };
            let gpu_data =
                Arc::new(TrxGpuData::from_tractogram(&tractogram).map_err(|err| err.to_string())?);
            let selected = (0..gpu_data.nb_streamlines as u32).collect();
            Ok(WorkflowJobOutput::ReactiveStreamline(StreamlineFlow {
                dataset: Arc::new(StreamlineDataset {
                    name: plan.label,
                    gpu_data,
                    backing: StreamlineBacking::Derived(Arc::new(tractogram)),
                }),
                selected_streamlines: Arc::new(selected),
                color_mode: plan.left.color_mode.clone(),
                scalar_auto_range: true,
                scalar_range_min: 0.0,
                scalar_range_max: 1.0,
            }))
        }
        WorkflowJobPayload::SurfaceQuery(plan) => {
            let hits = plan
                .flow
                .dataset
                .gpu_data
                .query_near_surface(&plan.surface, plan.depth_mm);
            let selected = plan
                .flow
                .selected_streamlines
                .iter()
                .copied()
                .filter(|index| hits.contains(index))
                .collect();
            Ok(WorkflowJobOutput::SurfaceQuery(StreamlineFlow {
                selected_streamlines: Arc::new(selected),
                ..plan.flow
            }))
        }
        WorkflowJobPayload::SurfaceMap(plan) => {
            let subset = plan
                .flow
                .dataset
                .gpu_data
                .subset_streamlines(plan.flow.selected_streamlines.as_ref());
            let dps_storage;
            let dps_values = if let Some(field) = &plan.dps_field {
                dps_storage = subset
                    .dps_data
                    .iter()
                    .find(|(name, _)| name == field)
                    .map(|(_, values)| values.clone())
                    .ok_or_else(|| format!("DPS field `{field}` is not available"))?;
                Some(dps_storage.as_slice())
            } else {
                None
            };
            let (density, projected) = subset.project_selected_to_surface(
                &plan.surface,
                &(0..subset.nb_streamlines as u32).collect::<Vec<_>>(),
                plan.depth_mm,
                dps_values,
            );
            let scalars = plan
                .dps_field
                .as_ref()
                .map(|_| projected)
                .unwrap_or(density);
            let (range_min, range_max) = robust_range(&scalars);
            Ok(WorkflowJobOutput::SurfaceMap(SurfaceStreamlineMap {
                surface_id: plan.surface_id,
                scalars,
                range_min,
                range_max,
            }))
        }
        WorkflowJobPayload::TubeGeometry(draw) => {
            let subset = materialize_flow_gpu(draw.flow);
            let selected = (0..subset.nb_streamlines as u32).collect::<Vec<_>>();
            let (positions, colors, offsets) = subset.selected_tube_data(&selected);
            let (vertices, indices) = build_tube_vertices_from_data(
                &positions,
                &colors,
                &offsets,
                draw.tube_radius_mm,
                draw.tube_sides,
            );
            Ok(WorkflowJobOutput::TubeGeometry { vertices, indices })
        }
        WorkflowJobPayload::BundleSurface {
            plan,
            color_mode,
            boundary_field,
        } => Ok(WorkflowJobOutput::BundleSurface {
            meshes: build_bundle_surface_meshes_with_color_mode(
                &plan,
                color_mode,
                boundary_field.as_deref(),
            ),
        }),
        WorkflowJobPayload::BoundaryField { plan } => {
            let subset = materialize_flow_gpu(plan.flow);
            let selected = (0..subset.nb_streamlines as u32).collect::<Vec<_>>();
            let (positions, _colors, offsets) = subset.selected_tube_data(&selected);
            if offsets.len() <= 1 {
                return Ok(WorkflowJobOutput::BoundaryField { field: None });
            }
            let params = crate::data::orientation_field::BoundaryGlyphParams {
                voxel_size_mm: plan.voxel_size_mm,
                sphere_lod: plan.sphere_lod,
                normalization: plan.normalization,
                ..crate::data::orientation_field::BoundaryGlyphParams::default()
            };
            Ok(WorkflowJobOutput::BoundaryField {
                field: BoundaryContactField::build_from_streamlines(
                    &[StreamlineSet { positions, offsets }],
                    &params,
                )
                .map(Arc::new),
            })
        }
    }
}

fn sync_node_state_from_run_record(
    node_state: &mut NodeEvalState,
    record: &ExpensiveNodeRunRecord,
) {
    node_state.execution = Some(record.status.clone());
    node_state.fingerprint = record.current_fingerprint;
    node_state.last_result_summary = record.last_result_summary.clone();
}

fn prime_expensive_record(record: &mut ExpensiveNodeRunRecord, fingerprint: u64) {
    record.current_fingerprint = Some(fingerprint);
    if record.last_success_fingerprint == Some(fingerprint) {
        record.status = WorkflowExecutionStatus::Ready;
    } else if record.last_success_fingerprint.is_some() {
        record.status = WorkflowExecutionStatus::Stale;
    } else {
        record.status = WorkflowExecutionStatus::NeverRun;
    }
}

fn mark_expensive_success(
    record: &mut ExpensiveNodeRunRecord,
    fingerprint: u64,
    result_summary: String,
) {
    record.current_fingerprint = Some(fingerprint);
    record.last_success_fingerprint = Some(fingerprint);
    record.status = WorkflowExecutionStatus::Ready;
    record.last_result_summary = Some(result_summary);
}

fn mark_expensive_failure(record: &mut ExpensiveNodeRunRecord, fingerprint: u64, error: &str) {
    record.current_fingerprint = Some(fingerprint);
    record.status = WorkflowExecutionStatus::Failed(error.to_string());
}

fn materialize_flow_gpu(flow: StreamlineFlow) -> TrxGpuData {
    let mut subset = flow
        .dataset
        .gpu_data
        .subset_streamlines(flow.selected_streamlines.as_ref());
    let scalar_range = if flow.scalar_auto_range {
        None
    } else {
        Some((flow.scalar_range_min, flow.scalar_range_max))
    };
    subset.recolor(&flow.color_mode, scalar_range);
    subset
}

fn bundle_surface_component_flows(plan: &BundleSurfacePlan) -> Vec<(String, StreamlineFlow)> {
    if !plan.per_group {
        return vec![(plan.label.clone(), plan.flow.clone())];
    }

    let selected: HashSet<u32> = plan.flow.selected_streamlines.iter().copied().collect();
    let mut components = Vec::new();
    for (group_name, members) in &plan.flow.dataset.gpu_data.groups {
        let group_selected: Vec<u32> = members
            .iter()
            .copied()
            .filter(|member| selected.contains(member))
            .collect();
        if group_selected.is_empty() {
            continue;
        }
        components.push((
            group_name.clone(),
            StreamlineFlow {
                dataset: plan.flow.dataset.clone(),
                selected_streamlines: Arc::new(group_selected),
                color_mode: plan.flow.color_mode.clone(),
                scalar_auto_range: plan.flow.scalar_auto_range,
                scalar_range_min: plan.flow.scalar_range_min,
                scalar_range_max: plan.flow.scalar_range_max,
            },
        ));
    }

    if components.is_empty() {
        vec![(plan.label.clone(), plan.flow.clone())]
    } else {
        components
    }
}

fn build_bundle_surface_meshes_with_color_mode(
    plan: &BundleSurfacePlan,
    color_mode: BundleSurfaceColorMode,
    boundary_field: Option<&BoundaryContactField>,
) -> Vec<(BundleMesh, String)> {
    bundle_surface_component_flows(plan)
        .into_iter()
        .filter_map(|(label, flow)| {
            let subset = materialize_flow_gpu(flow);
            if subset.nb_streamlines == 0 {
                return None;
            }
            let selected = (0..subset.nb_streamlines as u32).collect::<Vec<_>>();
            let (positions, colors) = subset.selected_vertex_data(&selected);
            let solid_color = bundle_surface_solid_color(&plan.flow, &label, plan.per_group);
            let (strategy, boundary_field) = match color_mode {
                BundleSurfaceColorMode::Solid => {
                    (BundleMeshColorStrategy::Constant(solid_color), None)
                }
                BundleSurfaceColorMode::BoundaryField => (
                    if boundary_field.is_some() {
                        BundleMeshColorStrategy::BoundaryField
                    } else {
                        BundleMeshColorStrategy::Constant(solid_color)
                    },
                    boundary_field,
                ),
            };
            build_bundle_mesh(
                &positions,
                &colors,
                plan.voxel_size_mm,
                plan.threshold,
                plan.smooth_sigma,
                strategy,
                boundary_field,
            )
            .map(|mesh| (mesh, label))
        })
        .collect()
}

fn bundle_surface_solid_color(flow: &StreamlineFlow, label: &str, per_group: bool) -> [f32; 4] {
    if per_group
        && let Some(group_idx) = flow
            .dataset
            .gpu_data
            .groups
            .iter()
            .position(|(name, _)| name == label)
    {
        if let Some(Some(color)) = flow.dataset.gpu_data.group_colors.get(group_idx) {
            return *color;
        }
        if let Some(color) = group_name_color(label) {
            return color;
        }
    }
    pleasant_bundle_color(label)
}

fn pleasant_bundle_color(label: &str) -> [f32; 4] {
    const PALETTE: [[f32; 4]; 8] = [
        [0.165, 0.455, 0.702, 1.0],
        [0.922, 0.467, 0.208, 1.0],
        [0.239, 0.698, 0.412, 1.0],
        [0.753, 0.353, 0.431, 1.0],
        [0.639, 0.471, 0.878, 1.0],
        [0.816, 0.686, 0.267, 1.0],
        [0.247, 0.651, 0.710, 1.0],
        [0.855, 0.400, 0.310, 1.0],
    ];
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    PALETTE[(hasher.finish() as usize) % PALETTE.len()]
}

fn workflow_streamline_fingerprint(draw: &StreamlineDrawPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    draw.label.hash(&mut hasher);
    (draw.render_style as u32).hash(&mut hasher);
    draw.tube_radius_mm.to_bits().hash(&mut hasher);
    draw.tube_sides.hash(&mut hasher);
    draw.slab_half_width_mm.to_bits().hash(&mut hasher);
    hash_flow(&draw.flow, &mut hasher);
    hasher.finish()
}

fn workflow_reactive_streamline_fingerprint(plan: &ReactiveStreamlinePlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    match plan.op {
        ReactiveStreamlineOp::Merge => 0u8.hash(&mut hasher),
    }
    hash_flow(&plan.left, &mut hasher);
    hash_flow(&plan.right, &mut hasher);
    hasher.finish()
}

fn workflow_surface_query_fingerprint(
    flow: &StreamlineFlow,
    surface_id: FileId,
    depth_mm: f32,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface_id.hash(&mut hasher);
    depth_mm.to_bits().hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

fn workflow_surface_projection_fingerprint(
    flow: &StreamlineFlow,
    surface_id: FileId,
    depth_mm: f32,
    field: Option<&str>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    surface_id.hash(&mut hasher);
    depth_mm.to_bits().hash(&mut hasher);
    field.hash(&mut hasher);
    hash_flow(flow, &mut hasher);
    hasher.finish()
}

fn workflow_bundle_build_fingerprint(draw: &BundleDrawPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    draw.label.hash(&mut hasher);
    draw.per_group.hash(&mut hasher);
    draw.voxel_size_mm.to_bits().hash(&mut hasher);
    draw.threshold.to_bits().hash(&mut hasher);
    draw.smooth_sigma.to_bits().hash(&mut hasher);
    draw.opacity.to_bits().hash(&mut hasher);
    hash_flow(&draw.flow, &mut hasher);
    hasher.finish()
}

fn workflow_bundle_display_fingerprint(
    draw: &BundleDrawPlan,
    boundary_field_revision: Option<u64>,
) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    workflow_bundle_build_fingerprint(draw).hash(&mut hasher);
    draw.color_mode.hash(&mut hasher);
    boundary_field_revision.hash(&mut hasher);
    hasher.finish()
}

fn workflow_bundle_plan_fingerprint(plan: &BundleSurfacePlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    plan.per_group.hash(&mut hasher);
    plan.voxel_size_mm.to_bits().hash(&mut hasher);
    plan.threshold.to_bits().hash(&mut hasher);
    plan.smooth_sigma.to_bits().hash(&mut hasher);
    plan.opacity.to_bits().hash(&mut hasher);
    hash_flow(&plan.flow, &mut hasher);
    hasher.finish()
}

fn workflow_boundary_plan_fingerprint(plan: &BoundaryFieldPlan) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    plan.label.hash(&mut hasher);
    plan.voxel_size_mm.to_bits().hash(&mut hasher);
    plan.sphere_lod.hash(&mut hasher);
    plan.normalization.hash(&mut hasher);
    hash_flow(&plan.flow, &mut hasher);
    hasher.finish()
}

fn hash_flow(flow: &StreamlineFlow, state: &mut impl Hasher) {
    flow.dataset.name.hash(state);
    flow.selected_streamlines.len().hash(state);
    for index in flow.selected_streamlines.iter().take(128) {
        index.hash(state);
    }
    match &flow.color_mode {
        ColorMode::DirectionRgb => 0u8.hash(state),
        ColorMode::Dpv(name) => {
            1u8.hash(state);
            name.hash(state);
        }
        ColorMode::Dps(name) => {
            2u8.hash(state);
            name.hash(state);
        }
        ColorMode::Group => 3u8.hash(state),
        ColorMode::Uniform(color) => {
            4u8.hash(state);
            for channel in color {
                channel.to_bits().hash(state);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(extension: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("trx_viewer_workflow_{stamp}.{extension}"))
    }

    #[test]
    fn workflow_project_round_trips_json() {
        let mut document = default_document();
        document.assets.push(WorkflowAssetDocument::Streamlines {
            id: 7,
            path: PathBuf::from("/tmp/example.trx"),
            imported: false,
        });
        make_node(
            &mut document,
            WorkflowNodeKind::StreamlineSource { source_id: 7 },
            Pos2::new(10.0, 10.0),
        );

        let path = temp_path("json");
        save_workflow_project_to_path(&document, &path).unwrap();
        let loaded = load_workflow_project_from_path(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.document.assets.len(), 1);
        assert_eq!(loaded.document.graph.nodes().count(), 1);
    }

    #[test]
    fn save_streamline_plan_preserves_subset_groups() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .unwrap();
        tractogram
            .push_streamline(&[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            .unwrap();
        tractogram.insert_group("bundle_a", vec![1]);
        let rgb = [[[255u8, 128u8, 0u8]]];
        tractogram.insert_dpg(
            "bundle_a",
            "color",
            DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
        );

        let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).unwrap());
        let flow = StreamlineFlow {
            dataset: Arc::new(StreamlineDataset {
                name: "subset".to_string(),
                gpu_data,
                backing: StreamlineBacking::Derived(Arc::new(tractogram)),
            }),
            selected_streamlines: Arc::new(vec![1]),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
        };

        let path = temp_path("trx");
        save_streamline_plan(&SaveStreamlinePlan {
            node_uuid: WorkflowNodeUuid(1),
            output_path: path.clone(),
            flow,
        })
        .unwrap();

        let loaded = AnyTrxFile::load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.nb_streamlines(), 1);
        let groups = loaded.groups_owned();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, "bundle_a");
        assert_eq!(groups[0].1, vec![0]);
    }

    #[test]
    fn add_groups_from_parcellation_creates_named_streamline_groups() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
            .unwrap();
        tractogram
            .push_streamline(&[[1.0, 1.0, 0.0], [1.0, 1.0, 0.2]])
            .unwrap();

        let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).unwrap());
        let flow = StreamlineFlow {
            dataset: Arc::new(StreamlineDataset {
                name: "input".to_string(),
                gpu_data,
                backing: StreamlineBacking::Derived(Arc::new(tractogram)),
            }),
            selected_streamlines: Arc::new(vec![0, 1]),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
        };
        let parcellation = ParcellationVolume {
            labels: vec![1, 0, 0, 2],
            dims: [2, 2, 1],
            voxel_to_ras: glam::Mat4::IDENTITY,
            world_to_voxel: glam::Mat4::IDENTITY,
            label_table: BTreeMap::from([
                (
                    1,
                    crate::data::parcellation_data::ParcelLabel {
                        id: 1,
                        name: "Motor".to_string(),
                        color: [1.0, 0.0, 0.0, 1.0],
                    },
                ),
                (
                    2,
                    crate::data::parcellation_data::ParcelLabel {
                        id: 2,
                        name: "Visual".to_string(),
                        color: [0.0, 1.0, 0.0, 1.0],
                    },
                ),
            ]),
        };
        let node = WorkflowNode {
            uuid: WorkflowNodeUuid(1),
            label: "Grouped".to_string(),
            kind: WorkflowNodeKind::AddGroupsFromParcellation,
        };

        let grouped =
            add_groups_from_parcellation(&node, &flow, &parcellation, "Atlas.nii.gz").unwrap();
        let group_map: HashMap<_, _> = grouped.dataset.gpu_data.groups.iter().cloned().collect();

        assert_eq!(group_map.get("Atlas_Motor"), Some(&vec![0]));
        assert_eq!(group_map.get("Atlas_Visual"), Some(&vec![1]));
    }

    #[test]
    fn surface_projection_requires_explicit_run() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            .unwrap();
        let tractogram = Arc::new(tractogram);
        let gpu_data = Arc::new(TrxGpuData::from_tractogram(tractogram.as_ref()).unwrap());
        let make_streamline = || LoadedTrx {
            id: 1,
            name: "streamlines.trx".to_string(),
            path: PathBuf::from("/tmp/streamlines.trx"),
            data: gpu_data.clone(),
            backing: Some(StreamlineBacking::Derived(tractogram.clone())),
        };

        let surface_data = Arc::new(crate::data::gifti_data::GiftiSurfaceData {
            vertices: vec![[0.0, -1.0, 0.5], [1.0, 1.0, 0.5], [-1.0, 1.0, 0.5]],
            normals: vec![[0.0, 0.0, 1.0]; 3],
            indices: vec![0, 1, 2],
            bbox_min: Vec3::new(-1.0, -1.0, 0.5),
            bbox_max: Vec3::new(1.0, 1.0, 0.5),
        });
        let make_surface = || crate::app::state::LoadedGiftiSurface {
            id: 2,
            name: "surface.gii".to_string(),
            path: PathBuf::from("/tmp/surface.gii"),
            data: surface_data.clone(),
            gpu_index: 0,
            visible: true,
            opacity: 0.7,
            color: [0.8, 0.4, 0.4],
            show_projection_map: false,
            map_opacity: 1.0,
            map_threshold: 0.0,
            surface_gloss: 0.45,
            projection_colormap: SurfaceColormap::Inferno,
            auto_range: true,
            range_min: 0.0,
            range_max: 1.0,
        };

        let mut document = default_document();
        let source = make_node(
            &mut document,
            WorkflowNodeKind::StreamlineSource { source_id: 1 },
            Pos2::new(0.0, 0.0),
        );
        let surface_source = make_node(
            &mut document,
            WorkflowNodeKind::SurfaceSource { source_id: 2 },
            Pos2::new(0.0, 120.0),
        );
        let projection = make_node(
            &mut document,
            WorkflowNodeKind::SurfaceProjectionDensity { depth_mm: 2.0 },
            Pos2::new(240.0, 0.0),
        );
        let display = make_node(
            &mut document,
            WorkflowNodeKind::SurfaceDisplay {
                color: DEFAULT_SURFACE_COLOR,
                opacity: DEFAULT_SURFACE_OPACITY,
                show_projection_map: false,
                map_opacity: 1.0,
                map_threshold: 0.0,
                gloss: 0.45,
                projection_colormap: SurfaceColormap::Inferno,
                range_min: 0.0,
                range_max: 1.0,
            },
            Pos2::new(480.0, 60.0),
        );
        document.graph.connect(
            OutPinId {
                node: source,
                output: 0,
            },
            InPinId {
                node: projection,
                input: 0,
            },
        );
        document.graph.connect(
            OutPinId {
                node: surface_source,
                output: 0,
            },
            InPinId {
                node: projection,
                input: 1,
            },
        );
        document.graph.connect(
            OutPinId {
                node: surface_source,
                output: 0,
            },
            InPinId {
                node: display,
                input: 0,
            },
        );
        document.graph.connect(
            OutPinId {
                node: projection,
                output: 0,
            },
            InPinId {
                node: display,
                input: 1,
            },
        );

        let mut display_ids = HashMap::new();
        let mut next_draw_id = 1_000_000;
        let mut execution_cache = WorkflowExecutionCache::default();

        let runtime_without_run = evaluate_scene_plan(
            &document,
            &[make_streamline()],
            &[],
            &[make_surface()],
            &[],
            &mut display_ids,
            &mut next_draw_id,
            &mut execution_cache,
            false,
        );
        let projection_uuid = document.graph[projection].uuid;
        assert_eq!(
            runtime_without_run
                .node_state
                .get(&projection_uuid)
                .and_then(|state| state.execution.as_ref())
                .map(WorkflowExecutionStatus::label),
            Some("Run required")
        );
        assert!(
            runtime_without_run.scene_plan.surface_draws[0]
                .projection_scalars
                .is_none()
        );

        let map_plan = runtime_without_run.scene_plan.surface_map_plans[0].clone();
        let map_fingerprint = workflow_surface_projection_fingerprint(
            &map_plan.flow,
            map_plan.surface_id,
            map_plan.depth_mm,
            map_plan.dps_field.as_deref(),
        );
        let job_output = run_workflow_job(WorkflowJobPayload::SurfaceMap(map_plan)).unwrap();
        let WorkflowJobOutput::SurfaceMap(map) = job_output else {
            panic!("expected surface map output");
        };
        execution_cache.surface_streamline_map_cache.insert(
            projection_uuid,
            CachedSurfaceStreamlineMap {
                fingerprint: map_fingerprint,
                map,
            },
        );
        mark_expensive_success(
            execution_cache
                .node_runs
                .entry(projection_uuid)
                .or_default(),
            map_fingerprint,
            "Surface streamline map".to_string(),
        );

        let runtime_with_cached_result = evaluate_scene_plan(
            &document,
            &[make_streamline()],
            &[],
            &[make_surface()],
            &[],
            &mut display_ids,
            &mut next_draw_id,
            &mut execution_cache,
            false,
        );
        assert!(
            runtime_with_cached_result.scene_plan.surface_draws[0]
                .projection_scalars
                .is_some()
        );
    }

    #[test]
    fn bundle_surface_build_requires_display_node_to_render() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            .unwrap();
        let tractogram = Arc::new(tractogram);
        let gpu_data = Arc::new(TrxGpuData::from_tractogram(tractogram.as_ref()).unwrap());
        let make_streamline = || LoadedTrx {
            id: 1,
            name: "streamlines.trx".to_string(),
            path: PathBuf::from("/tmp/streamlines.trx"),
            data: gpu_data.clone(),
            backing: Some(StreamlineBacking::Derived(tractogram.clone())),
        };

        let mut document = default_document();
        let source = make_node(
            &mut document,
            WorkflowNodeKind::StreamlineSource { source_id: 1 },
            Pos2::new(0.0, 0.0),
        );
        let build = make_node(
            &mut document,
            WorkflowNodeKind::BundleSurfaceBuild {
                per_group: false,
                voxel_size_mm: 2.0,
                threshold: 3.0,
                smooth_sigma: 0.5,
                opacity: 0.5,
            },
            Pos2::new(240.0, 0.0),
        );
        document.graph.connect(
            OutPinId {
                node: source,
                output: 0,
            },
            InPinId {
                node: build,
                input: 0,
            },
        );

        let mut display_ids = HashMap::new();
        let mut next_draw_id = 1_000_000;
        let mut execution_cache = WorkflowExecutionCache::default();

        let runtime_without_display = evaluate_scene_plan(
            &document,
            &[make_streamline()],
            &[],
            &[],
            &[],
            &mut display_ids,
            &mut next_draw_id,
            &mut execution_cache,
            true,
        );
        assert!(runtime_without_display.scene_plan.bundle_draws.is_empty());

        let display = make_node(
            &mut document,
            WorkflowNodeKind::BundleSurfaceDisplay {
                color_mode: BundleSurfaceColorMode::Solid,
            },
            Pos2::new(480.0, 0.0),
        );
        document.graph.connect(
            OutPinId {
                node: build,
                output: 0,
            },
            InPinId {
                node: display,
                input: 0,
            },
        );

        let runtime_with_display = evaluate_scene_plan(
            &document,
            &[make_streamline()],
            &[],
            &[],
            &[],
            &mut display_ids,
            &mut next_draw_id,
            &mut execution_cache,
            true,
        );
        assert_eq!(runtime_with_display.scene_plan.bundle_draws.len(), 1);
    }

    #[test]
    fn per_group_bundle_surface_build_splits_selected_groups() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .unwrap();
        tractogram
            .push_streamline(&[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
            .unwrap();
        tractogram.insert_group("bundle_a", vec![0]);
        tractogram.insert_group("bundle_b", vec![1]);

        let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).unwrap());
        let flow = StreamlineFlow {
            dataset: Arc::new(StreamlineDataset {
                name: "grouped".to_string(),
                gpu_data,
                backing: StreamlineBacking::Derived(Arc::new(tractogram)),
            }),
            selected_streamlines: Arc::new(vec![0, 1]),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
        };
        let plan = BundleSurfacePlan {
            build_node_uuid: WorkflowNodeUuid(7),
            label: "Bundle Surface".to_string(),
            flow,
            per_group: true,
            voxel_size_mm: 2.0,
            threshold: 3.0,
            smooth_sigma: 0.5,
            opacity: 0.5,
        };

        let mut components = bundle_surface_component_flows(&plan);
        components.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].0, "bundle_a");
        assert_eq!(components[0].1.selected_streamlines.as_ref(), &vec![0]);
        assert_eq!(components[1].0, "bundle_b");
        assert_eq!(components[1].1.selected_streamlines.as_ref(), &vec![1]);
    }

    #[test]
    fn bundle_surface_solid_color_prefers_group_dpg_color() {
        let mut tractogram = Tractogram::new();
        tractogram
            .push_streamline(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
            .unwrap();
        tractogram.insert_group("bundle_a", vec![0]);
        let rgb = [[[32u8, 96u8, 192u8]]];
        tractogram.insert_dpg(
            "bundle_a",
            "color",
            DataArray::owned_bytes(bytemuck::cast_slice(&rgb).to_vec(), 3, DType::UInt8),
        );

        let gpu_data = Arc::new(TrxGpuData::from_tractogram(&tractogram).unwrap());
        let flow = StreamlineFlow {
            dataset: Arc::new(StreamlineDataset {
                name: "grouped".to_string(),
                gpu_data,
                backing: StreamlineBacking::Derived(Arc::new(tractogram)),
            }),
            selected_streamlines: Arc::new(vec![0]),
            color_mode: ColorMode::DirectionRgb,
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
        };

        let color = bundle_surface_solid_color(&flow, "bundle_a", true);
        assert_eq!(color, [32.0 / 255.0, 96.0 / 255.0, 192.0 / 255.0, 1.0]);
    }
}
