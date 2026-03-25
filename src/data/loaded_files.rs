use std::collections::HashSet;
use std::path::PathBuf;

use crate::data::bundle_mesh::BundleMesh;
use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};

pub type FileId = usize;

/// Source for the bundle surface mesh.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BundleMeshSource {
    /// All streamlines in the file.
    All,
    /// Currently filtered / visible selection.
    Selection,
    /// One mesh per TRX group.
    PerGroup,
}

/// All per-TRX-file state: data, rendering options, filters, bundle mesh, etc.
pub struct LoadedTrx {
    pub id: FileId,
    pub name: String,
    pub path: PathBuf,
    pub data: TrxGpuData,
    pub visible: bool,
    pub color_mode: ColorMode,
    pub render_style: RenderStyle,
    pub tube_radius: f32,
    pub group_visible: Vec<bool>,
    pub max_streamlines: usize,
    pub use_random_subset: bool,
    pub streamline_order: Vec<u32>,
    pub uniform_color: [f32; 4],
    pub scalar_auto_range: bool,
    pub scalar_range_min: f32,
    pub scalar_range_max: f32,
    pub colors_dirty: bool,
    pub indices_dirty: bool,
    pub slab_half_width: f32,
    pub show_bundle_mesh: bool,
    pub bundle_mesh_source: BundleMeshSource,
    pub bundle_mesh_voxel_size: f32,
    pub bundle_mesh_threshold: f32,
    pub bundle_mesh_smooth: f32,
    pub bundle_mesh_opacity: f32,
    pub bundle_mesh_ambient: f32,
    pub bundle_meshes_cpu: Vec<BundleMesh>,
    pub bundle_mesh_pending: Option<std::sync::mpsc::Receiver<Vec<(BundleMesh, String)>>>,
    pub bundle_mesh_dirty_at: Option<std::time::Instant>,
    pub sphere_query_result: Option<HashSet<u32>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VolumeColormap {
    Grayscale,
    Hot,
    Cool,
    RedYellow,
    BlueLightblue,
}

impl VolumeColormap {
    pub fn as_u32(&self) -> u32 {
        match self {
            VolumeColormap::Grayscale => 0,
            VolumeColormap::Hot => 1,
            VolumeColormap::Cool => 2,
            VolumeColormap::RedYellow => 3,
            VolumeColormap::BlueLightblue => 4,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            VolumeColormap::Grayscale => "Grayscale",
            VolumeColormap::Hot => "Hot",
            VolumeColormap::Cool => "Cool",
            VolumeColormap::RedYellow => "Red-Yellow",
            VolumeColormap::BlueLightblue => "Blue-Lightblue",
        }
    }

    pub const ALL: &[VolumeColormap] = &[
        VolumeColormap::Grayscale,
        VolumeColormap::Hot,
        VolumeColormap::Cool,
        VolumeColormap::RedYellow,
        VolumeColormap::BlueLightblue,
    ];
}

pub struct LoadedNifti {
    pub id: FileId,
    pub name: String,
    pub volume: NiftiVolume,
    pub colormap: VolumeColormap,
    pub opacity: f32,
    pub z_order: i32,
    pub window_center: f32,
    pub window_width: f32,
    pub visible: bool,
}
