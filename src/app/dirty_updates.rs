use std::collections::HashSet;

use super::helpers::robust_range;
use super::state::{SurfaceProjectionMode, WorkerMessage};
use crate::renderer::mesh_renderer::MeshResources;

impl super::TrxViewerApp {
    pub(super) fn recompute_surface_query(&mut self) {
        if !self.surface_query_active
            || self.trx_files.is_empty()
            || self.surface_query_surface >= self.gifti_surfaces.len()
        {
            self.surface_query_dirty = false;
            self.surface_query_pending = false;
            self.surface_query_result = None;
            for trx in &mut self.trx_files {
                trx.indices_dirty = true;
            }
            self.mark_boundary_field_dirty();
            return;
        }

        self.surface_query_dirty = true;
    }

    pub(super) fn kick_surface_query_job(&mut self) {
        if !self.surface_query_dirty || self.surface_query_pending {
            return;
        }
        if !self.surface_query_active
            || self.trx_files.is_empty()
            || self.surface_query_surface >= self.gifti_surfaces.len()
        {
            self.surface_query_dirty = false;
            self.surface_query_result = None;
            return;
        }

        let revision = self.surface_query_revision.wrapping_add(1);
        self.surface_query_revision = revision;
        self.surface_query_pending = true;
        self.surface_query_dirty = false;

        let surf = self.gifti_surfaces[self.surface_query_surface].data.clone();
        let depth = self.gifti_surfaces[self.surface_query_surface]
            .projection_depth_mm
            .max(0.0);
        let trxs: Vec<_> = self.trx_files.iter().map(|trx| trx.data.clone()).collect();
        let tx = self.worker_tx.clone();

        std::thread::spawn(move || {
            let mut combined = HashSet::new();
            for trx in trxs {
                combined.extend(trx.query_near_surface(&surf, depth));
            }
            let _ = tx.send(WorkerMessage::SurfaceQueryDone {
                revision,
                result: Some(combined),
            });
        });
    }

    pub(super) fn apply_surface_query_result(
        &mut self,
        revision: u64,
        result: Option<HashSet<u32>>,
    ) {
        self.surface_query_pending = false;
        if revision != self.surface_query_revision || self.surface_query_dirty {
            return;
        }
        self.surface_query_result = result;
        for trx in &mut self.trx_files {
            trx.indices_dirty = true;
        }
        self.mark_boundary_field_dirty();
    }

    pub(super) fn kick_surface_projection_job(&mut self) {
        if !self.surface_projection_dirty || self.surface_projection_pending {
            return;
        }
        if self.trx_files.is_empty() || self.gifti_surfaces.is_empty() {
            self.surface_projection_dirty = false;
            return;
        }

        let revision = self.surface_projection_revision.wrapping_add(1);
        self.surface_projection_revision = revision;
        self.surface_projection_pending = true;
        self.surface_projection_dirty = false;

        let trx = self.trx_files[0].data.clone();
        let group_visible = self.trx_files[0].group_visible.clone();
        let max_streamlines = self.trx_files[0].max_streamlines;
        let streamline_order = self.trx_files[0].streamline_order.clone();
        let sphere_query_result = self.trx_files[0].sphere_query_result.clone();
        let surface_query_result = self.surface_query_result.clone();
        let surface_specs: Vec<_> = self
            .gifti_surfaces
            .iter()
            .enumerate()
            .map(|(surface_idx, surface)| {
                (
                    surface_idx,
                    surface.data.clone(),
                    surface.projection_mode,
                    surface.projection_dps.clone(),
                    surface.projection_depth_mm.max(0.0),
                )
            })
            .collect();
        let tx = self.worker_tx.clone();

        std::thread::spawn(move || {
            let selected = trx.filtered_streamline_indices(
                &group_visible,
                max_streamlines,
                &streamline_order,
                sphere_query_result.as_ref(),
                surface_query_result.as_ref(),
            );
            let mut outputs = Vec::with_capacity(surface_specs.len());

            for (surface_idx, surface_data, projection_mode, projection_dps, depth_mm) in surface_specs
            {
                let dps_values = projection_dps.as_ref().and_then(|name| {
                    trx.dps_data
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, v)| v.as_slice())
                });
                let (density, mean_dps) =
                    trx.project_selected_to_surface(&surface_data, &selected, depth_mm, dps_values);
                let active_values = match projection_mode {
                    SurfaceProjectionMode::Density => density,
                    SurfaceProjectionMode::MeanDps => mean_dps,
                };
                let finite: Vec<f32> = active_values
                    .iter()
                    .copied()
                    .filter(|v| v.is_finite())
                    .collect();
                let (data_min, data_max) = robust_range(&finite);
                let fill_value = data_min;
                let scalars = active_values
                    .into_iter()
                    .map(|v| if v.is_finite() { v } else { fill_value })
                    .collect();
                outputs.push(super::state::SurfaceProjectionOutput {
                    surface_idx,
                    scalars,
                    data_min,
                    data_max,
                });
            }

            let _ = tx.send(WorkerMessage::SurfaceProjectionDone { revision, outputs });
        });
    }

    pub(super) fn apply_surface_projection_result(
        &mut self,
        frame: &mut eframe::Frame,
        revision: u64,
        outputs: Vec<super::state::SurfaceProjectionOutput>,
    ) {
        self.surface_projection_pending = false;
        if revision != self.surface_projection_revision || self.surface_projection_dirty {
            return;
        }

        if let Some(rs) = frame.wgpu_render_state() {
            let renderer = rs.renderer.read();
            if let Some(mr) = renderer.callback_resources.get::<MeshResources>() {
                for output in outputs {
                    if let Some(surface) = self.gifti_surfaces.get_mut(output.surface_idx) {
                        if surface.auto_range {
                            surface.range_min = output.data_min;
                            surface.range_max = output.data_max;
                        }
                        mr.update_surface_scalars(&rs.queue, surface.gpu_index, &output.scalars);
                    }
                }
            }
        }
    }
}
