use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;

use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::StreamlineBacking;
use crate::data::nifti_data::NiftiVolume;
use crate::data::parcellation_data::ParcellationVolume;
use crate::data::trx_data::TrxGpuData;
use crate::renderer::mesh_renderer::SurfaceColormap;
use trx_rs::Format;

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

pub struct LoadedStreamlineSource {
    pub data: TrxGpuData,
    pub backing: StreamlineBacking,
}

pub struct LoadedParcellationSource {
    pub data: ParcellationVolume,
    pub label_table_path: Option<PathBuf>,
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
}
