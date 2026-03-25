use super::helpers::robust_range;
use super::state::{
    SurfaceProjectionCacheKey, SurfaceProjectionCacheValue, SurfaceProjectionMode,
};
use crate::renderer::mesh_renderer::MeshResources;

impl super::TrxViewerApp {
    pub(super) fn recompute_surface_query(&mut self) {
        if !self.surface_query_active {
            self.surface_query_result = None;
            for trx in &mut self.trx_files {
                trx.indices_dirty = true;
            }
            return;
        }
        if self.trx_files.is_empty() {
            self.surface_query_result = None;
            for trx in &mut self.trx_files {
                trx.indices_dirty = true;
            }
            return;
        }
        if self.surface_query_surface >= self.gifti_surfaces.len() {
            self.surface_query_result = None;
            for trx in &mut self.trx_files {
                trx.indices_dirty = true;
            }
            return;
        }
        let surf = &self.gifti_surfaces[self.surface_query_surface];
        let depth = surf.projection_depth_mm.max(0.0);
        // Combine results from all TRX files
        let mut combined = std::collections::HashSet::new();
        for trx in &self.trx_files {
            let result = trx.data.query_near_surface(&surf.data, depth);
            combined.extend(result);
        }
        self.surface_query_result = Some(combined);
        for trx in &mut self.trx_files {
            trx.indices_dirty = true;
        }
    }

    pub(super) fn refresh_surface_projections(&mut self, frame: &mut eframe::Frame) {
        if self.trx_files.is_empty() {
            return;
        }
        if self.gifti_surfaces.is_empty() {
            return;
        }

        // Use the first TRX file for projection (can be extended later to combine all)
        let trx = &self.trx_files[0];
        let selected = trx.data.filtered_streamline_indices(
            &trx.group_visible,
            trx.max_streamlines,
            &trx.streamline_order,
            trx.sphere_query_result.as_ref(),
            self.surface_query_result.as_ref(),
        );

        let mut upload_scalars: Vec<(usize, Vec<f32>)> = Vec::new();

        for (surface_idx, surface) in self.gifti_surfaces.iter_mut().enumerate() {
            let key = SurfaceProjectionCacheKey {
                surface_idx,
                selection_revision: self.selection_revision,
                depth_bin: (surface.projection_depth_mm * 100.0).round() as i32,
                mode: surface.projection_mode,
                dps_name: surface.projection_dps.clone(),
            };

            if !self.surface_projection_cache.contains_key(&key) {
                let dps_values = surface
                    .projection_dps
                    .as_ref()
                    .and_then(|name| trx.data.dps_data.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_slice()));
                let (density, mean_dps) = trx.data.project_selected_to_surface(
                    &surface.data,
                    &selected,
                    surface.projection_depth_mm.max(0.0),
                    dps_values,
                );
                let active: Vec<f32> = match surface.projection_mode {
                    SurfaceProjectionMode::Density => density.iter().copied().filter(|v| v.is_finite()).collect(),
                    SurfaceProjectionMode::MeanDps => mean_dps.iter().copied().filter(|v| v.is_finite()).collect(),
                };
                let (data_min, data_max) = robust_range(&active);
                self.surface_projection_cache.insert(
                    key.clone(),
                    SurfaceProjectionCacheValue {
                        density,
                        mean_dps,
                        data_min,
                        data_max,
                    },
                );
            }

            if let Some(cached) = self.surface_projection_cache.get(&key) {
                let active_values = match surface.projection_mode {
                    SurfaceProjectionMode::Density => &cached.density,
                    SurfaceProjectionMode::MeanDps => &cached.mean_dps,
                };
                if surface.auto_range {
                    surface.range_min = cached.data_min;
                    surface.range_max = cached.data_max;
                }
                let fill_value = surface.range_min;
                let scalars: Vec<f32> = active_values
                    .iter()
                    .map(|v| if v.is_finite() { *v } else { fill_value })
                    .collect();
                upload_scalars.push((surface.gpu_index, scalars));
            }
        }

        if let Some(rs) = frame.wgpu_render_state() {
            let renderer = rs.renderer.read();
            if let Some(mr) = renderer.callback_resources.get::<MeshResources>() {
                for (gpu_index, scalars) in upload_scalars {
                    mr.update_surface_scalars(&rs.queue, gpu_index, &scalars);
                }
            }
        }
    }
}
