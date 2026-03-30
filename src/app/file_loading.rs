use std::path::PathBuf;
use std::sync::Arc;

use glam::Vec3;
use trx_rs::ConversionOptions;

use crate::data::gifti_data::GiftiSurfaceData;
use crate::data::loaded_files::{
    BundleMeshColorMode, BundleMeshSource, LoadedNifti, LoadedTrx, VolumeColormap,
};
use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::{ColorMode, RenderStyle, TrxGpuData};
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::glyph_renderer::GlyphResources;
use crate::renderer::mesh_renderer::{MeshResources, SurfaceColormap};
use crate::renderer::slice_renderer::{AllSliceResources, SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::{AllStreamlineResources, StreamlineResources};

use super::state::{ImportDialogState, LoadedGiftiSurface, SurfaceProjectionMode, WorkerMessage};

impl super::TrxViewerApp {
    pub(super) fn begin_load_trx(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading TRX".to_string());
        self.pending_file_loads.push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = TrxGpuData::load(&path).map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::TrxLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_nifti(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading NIfTI".to_string());
        self.pending_file_loads.push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = NiftiVolume::load(&path).map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::NiftiLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_load_gifti_surface(&mut self, path: PathBuf) {
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Loading {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Loading GIFTI".to_string());
        self.pending_file_loads.push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = GiftiSurfaceData::load(&path).map_err(|e| e.to_string());
            let _ = tx.send(WorkerMessage::GiftiLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn begin_import_tractogram(&mut self, state: &ImportDialogState) {
        let Some(path) = state.source_path.clone() else {
            return;
        };
        let job_id = self.next_job_id;
        self.next_job_id += 1;
        let tx = self.worker_tx.clone();
        let label = path
            .file_name()
            .map(|n| format!("Importing {}", n.to_string_lossy()))
            .unwrap_or_else(|| "Importing tractogram".to_string());
        self.pending_file_loads
            .push(super::state::PendingFileLoad { job_id, label });
        std::thread::spawn(move || {
            let result = match trx_rs::read_tractogram(&path, &ConversionOptions::default()) {
                Ok(tractogram) => TrxGpuData::from_tractogram(&tractogram).map_err(|e| e.to_string()),
                Err(err) => Err(err.to_string()),
            };
            let _ = tx.send(WorkerMessage::ImportedTractogramLoaded {
                job_id,
                path,
                result,
            });
        });
    }

    pub(super) fn apply_loaded_trx(
        &mut self,
        path: PathBuf,
        data: TrxGpuData,
        rs: &egui_wgpu::RenderState,
    ) {
        let is_first =
            self.trx_files.is_empty() && self.nifti_files.is_empty() && self.gifti_surfaces.is_empty();
        self.volume_center = data.center();
        self.volume_extent = data.extent();
        if is_first {
            self.camera_3d = OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
            self.reset_slice_cameras();
        }

        let resources = StreamlineResources::new(&rs.device, rs.target_format, &data);
        let data = Arc::new(data);

        let id = self.next_file_id;
        self.next_file_id += 1;

        {
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer
                .callback_resources
                .get_mut::<AllStreamlineResources>()
            {
                all.entries.push((id, resources));
            } else {
                renderer.callback_resources.insert(AllStreamlineResources {
                    entries: vec![(id, resources)],
                });
            }
            if renderer.callback_resources.get::<MeshResources>().is_none() {
                let mr = MeshResources::new(&rs.device, rs.target_format);
                renderer.callback_resources.insert(mr);
            }
            if renderer.callback_resources.get::<GlyphResources>().is_none() {
                let gr = GlyphResources::new(&rs.device, rs.target_format);
                renderer.callback_resources.insert(gr);
            }
        }

        if is_first {
            self.slice_world_offsets = [
                self.volume_center.z,
                self.volume_center.y,
                self.volume_center.x,
            ];
        }

        let max_streamlines = data.nb_streamlines.min(30_000);
        let streamline_order: Vec<u32> = (0..data.nb_streamlines as u32).collect();
        let group_visible = vec![true; data.groups.len()];
        let needs_index_update = data.nb_streamlines > max_streamlines;

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "file.trx".to_string());

        let trx = LoadedTrx {
            id,
            name,
            path,
            data,
            visible: true,
            color_mode: ColorMode::DirectionRgb,
            render_style: RenderStyle::Flat,
            tube_radius: 0.4,
            tube_sides: 8,
            group_visible,
            max_streamlines,
            use_random_subset: false,
            streamline_order,
            uniform_color: [1.0, 1.0, 1.0, 1.0],
            scalar_auto_range: true,
            scalar_range_min: 0.0,
            scalar_range_max: 1.0,
            colors_dirty: false,
            indices_dirty: needs_index_update,
            tube_mesh_pending: None,
            tube_mesh_dirty_at: None,
            tube_build_revision: 0,
            index_build_pending: false,
            slab_half_width: 5.0,
            show_bundle_mesh: false,
            bundle_mesh_source: BundleMeshSource::All,
            bundle_mesh_color_mode: BundleMeshColorMode::StreamlineColor,
            bundle_mesh_voxel_size: 2.0,
            bundle_mesh_threshold: 3.0,
            bundle_mesh_smooth: 0.2,
            bundle_mesh_opacity: 0.5,
            bundle_meshes_cpu: Vec::new(),
            bundle_mesh_pending: None,
            bundle_mesh_dirty_at: None,
            sphere_query_result: None,
            include_in_boundary_glyphs: false,
        };

        self.trx_files.push(trx);
        self.sphere_query_active = false;
        self.surface_query_active = false;
        self.surface_query_result = None;
        self.surface_query_dirty = false;
        self.surface_query_pending = false;
        self.selection_revision = self.selection_revision.wrapping_add(1);
        self.surface_projection_dirty = true;
        self.error_msg = None;
    }

    pub(super) fn apply_loaded_nifti(
        &mut self,
        path: PathBuf,
        vol: NiftiVolume,
        rs: &egui_wgpu::RenderState,
    ) {
        let first_nifti = self.nifti_files.is_empty();
        let slice_indices = [vol.dims[2] / 2, vol.dims[1] / 2, vol.dims[0] / 2];
        let is_first =
            self.nifti_files.is_empty() && self.trx_files.is_empty() && self.gifti_surfaces.is_empty();
        if is_first {
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
            self.camera_3d = OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
        }

        let slice_resources = SliceResources::new(&rs.device, &rs.queue, rs.target_format, &vol);
        slice_resources.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], &vol);
        slice_resources.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], &vol);
        slice_resources.update_slice(&rs.queue, SliceAxis::Sagittal, self.slice_indices[2], &vol);

        let id = self.next_file_id;
        self.next_file_id += 1;

        {
            let mut renderer = rs.renderer.write();
            if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                all.entries.push((id, slice_resources));
            } else {
                renderer.callback_resources.insert(AllSliceResources {
                    entries: vec![(id, slice_resources)],
                });
            }
        }

        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "volume.nii".to_string());
        self.nifti_files.push(LoadedNifti {
            id,
            name,
            volume: vol,
            colormap: VolumeColormap::Grayscale,
            opacity: 1.0,
            z_order: self.nifti_files.len() as i32,
            window_center: 0.5,
            window_width: 1.0,
            visible: true,
        });
        if first_nifti {
            self.slice_indices = slice_indices;
            self.reset_slice_view();
        } else {
            self.slices_dirty = false;
        }
        self.error_msg = None;
    }

    pub(super) fn apply_loaded_gifti_surface(
        &mut self,
        path: PathBuf,
        surface: GiftiSurfaceData,
        rs: &egui_wgpu::RenderState,
    ) {
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
        let surface = Arc::new(surface);

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
            path,
            data: surface,
            gpu_index,
            visible: true,
            opacity: 0.7,
            color,
            show_projection_map: false,
            map_opacity: 1.0,
            map_threshold: 0.0,
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

    pub(super) fn reset_slice_cameras(&mut self) {
        let half_extents = self
            .nifti_files
            .first()
            .map(|n| n.volume.slice_half_extents())
            .unwrap_or([self.volume_extent * 0.5; 3]);
        self.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, self.volume_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(
                SliceAxis::Coronal,
                self.volume_center,
                half_extents[1] * 2.0,
            ),
            OrthoSliceCamera::new(
                SliceAxis::Sagittal,
                self.volume_center,
                half_extents[2] * 2.0,
            ),
        ];
    }

    pub(crate) fn reset_slice_view(&mut self) {
        let Some(nf) = self.nifti_files.first() else {
            return;
        };
        let vol = &nf.volume;
        let world_center = vol.voxel_to_world(Vec3::new(
            vol.dims[0] as f32 / 2.0,
            vol.dims[1] as f32 / 2.0,
            vol.dims[2] as f32 / 2.0,
        ));
        let half_extents = vol.slice_half_extents();
        self.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, world_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Coronal, world_center, half_extents[1] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Sagittal, world_center, half_extents[2] * 2.0),
        ];
        self.slice_world_offsets = [world_center.z, world_center.y, world_center.x];
        self.slices_dirty = true;
    }
}
