use std::path::PathBuf;

use crate::data::gifti_data::GiftiSurfaceData;
use crate::renderer::mesh_renderer::SurfaceColormap;

// Re-export BundleMeshSource from its canonical location in data::loaded_files.
pub use crate::data::loaded_files::BundleMeshSource;

pub struct LoadedGiftiSurface {
    pub name: String,
    pub path: PathBuf,
    pub data: GiftiSurfaceData,
    pub gpu_index: usize,
    pub visible: bool,
    pub opacity: f32,
    pub color: [f32; 3],
    pub show_projection_map: bool,
    pub map_opacity: f32,
    pub map_threshold: f32,
    pub surface_ambient: f32,
    pub surface_gloss: f32,
    pub projection_mode: SurfaceProjectionMode,
    pub projection_dps: Option<String>,
    pub projection_depth_mm: f32,
    pub projection_colormap: SurfaceColormap,
    pub auto_range: bool,
    pub range_min: f32,
    pub range_max: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceProjectionMode {
    Density,
    MeanDps,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SurfaceProjectionCacheKey {
    pub surface_idx: usize,
    pub selection_revision: u64,
    pub depth_bin: i32,
    pub mode: SurfaceProjectionMode,
    pub dps_name: Option<String>,
}

#[derive(Clone)]
pub struct SurfaceProjectionCacheValue {
    pub density: Vec<f32>,
    pub mean_dps: Vec<f32>,
    pub data_min: f32,
    pub data_max: f32,
}
