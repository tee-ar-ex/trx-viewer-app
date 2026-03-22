use std::collections::HashSet;
use std::path::PathBuf;

use glam::Vec3;

use crate::data::nifti_data::NiftiVolume;
use crate::data::trx_data::{ColorMode, TrxGpuData};
use crate::renderer::camera::{OrbitCamera, OrthoSliceCamera};
use crate::renderer::slice_renderer::{SliceAxis, SliceResources};
use crate::renderer::streamline_renderer::StreamlineResources;

/// Main application state.
pub struct TrxViewerApp {
    trx_data: Option<TrxGpuData>,
    nifti_volume: Option<NiftiVolume>,
    camera_3d: OrbitCamera,
    slice_cameras: [OrthoSliceCamera; 3],
    /// Current slice indices [axial(k), coronal(j), sagittal(i)].
    slice_indices: [usize; 3],
    slices_dirty: bool,
    has_streamlines: bool,
    has_slices: bool,
    /// Loaded file paths for display.
    trx_path: Option<PathBuf>,
    nifti_path: Option<PathBuf>,
    /// Volume center/extent for camera reset.
    volume_center: Vec3,
    volume_extent: f32,
    /// Error message to display.
    error_msg: Option<String>,
    /// Current coloring mode.
    color_mode: ColorMode,
    /// Whether vertex colors need re-upload.
    colors_dirty: bool,
    /// Group visibility (one bool per group, all visible by default).
    group_visible: Vec<bool>,
    /// Whether the index buffer needs rebuild.
    indices_dirty: bool,
    /// Uniform color picker value.
    uniform_color: [f32; 4],
    /// Volume intensity windowing.
    window_center: f32,
    window_width: f32,
    /// Max number of streamlines to display.
    max_streamlines: usize,
    /// Whether to randomize which streamlines are shown.
    use_random_subset: bool,
    /// Ordering of streamline indices (identity or shuffled).
    streamline_order: Vec<u32>,
    /// Slab half-width for slice view streamline clipping (mm).
    slab_half_width: f32,
    /// Sphere query state.
    sphere_query_active: bool,
    sphere_center: Vec3,
    sphere_radius: f32,
    sphere_query_result: Option<HashSet<u32>>,
}

impl TrxViewerApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        trx_path: Option<String>,
        nifti_path: Option<String>,
    ) -> Self {
        let mut app = Self {
            trx_data: None,
            nifti_volume: None,
            camera_3d: OrbitCamera::new(Vec3::ZERO, 200.0),
            slice_cameras: [
                OrthoSliceCamera::new(SliceAxis::Axial, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Coronal, Vec3::ZERO, 200.0),
                OrthoSliceCamera::new(SliceAxis::Sagittal, Vec3::ZERO, 200.0),
            ],
            slice_indices: [0; 3],
            slices_dirty: false,
            has_streamlines: false,
            has_slices: false,
            trx_path: None,
            nifti_path: None,
            volume_center: Vec3::ZERO,
            volume_extent: 200.0,
            error_msg: None,
            color_mode: ColorMode::DirectionRgb,
            colors_dirty: false,
            group_visible: Vec::new(),
            indices_dirty: false,
            uniform_color: [1.0, 1.0, 1.0, 1.0],
            window_center: 0.5,
            window_width: 1.0,
            max_streamlines: 30_000,
            use_random_subset: false,
            streamline_order: Vec::new(),
            slab_half_width: 5.0,
            sphere_query_active: false,
            sphere_center: Vec3::ZERO,
            sphere_radius: 10.0,
            sphere_query_result: None,
        };

        if let Some(render_state) = &cc.wgpu_render_state {
            if let Some(path) = trx_path {
                app.load_trx(&PathBuf::from(path), render_state);
            }
            if let Some(path) = nifti_path {
                app.load_nifti(&PathBuf::from(path), render_state);
            }
        }

        app
    }

    fn load_trx(&mut self, path: &PathBuf, rs: &egui_wgpu::RenderState) {
        match TrxGpuData::load(path) {
            Ok(data) => {
                self.volume_center = data.center();
                self.volume_extent = data.extent();
                self.camera_3d =
                    OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
                self.reset_slice_cameras();

                let resources =
                    StreamlineResources::new(&rs.device, rs.target_format, &data);
                rs.renderer.write().callback_resources.insert(resources);

                self.has_streamlines = true;
                self.trx_path = Some(path.clone());
                self.group_visible = vec![true; data.groups.len()];
                self.color_mode = ColorMode::DirectionRgb;
                self.max_streamlines = data.nb_streamlines.min(30_000);
                self.streamline_order = (0..data.nb_streamlines as u32).collect();
                self.use_random_subset = false;
                self.sphere_query_active = false;
                self.sphere_query_result = None;
                // Apply max streamline limit on initial load
                if data.nb_streamlines > self.max_streamlines {
                    self.indices_dirty = true;
                }
                self.trx_data = Some(data);
                self.error_msg = None;
            }
            Err(e) => {
                self.error_msg = Some(format!("Failed to load TRX: {e}"));
            }
        }
    }

    fn load_nifti(&mut self, path: &PathBuf, rs: &egui_wgpu::RenderState) {
        match NiftiVolume::load(path) {
            Ok(vol) => {
                self.slice_indices = [
                    vol.dims[2] / 2,
                    vol.dims[1] / 2,
                    vol.dims[0] / 2,
                ];

                if !self.has_streamlines {
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
                    self.camera_3d =
                        OrbitCamera::new(self.volume_center, self.volume_extent * 0.8);
                }
                self.reset_slice_cameras();

                let slice_resources =
                    SliceResources::new(&rs.device, &rs.queue, rs.target_format, &vol);
                slice_resources.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], &vol);
                slice_resources.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], &vol);
                slice_resources.update_slice(
                    &rs.queue,
                    SliceAxis::Sagittal,
                    self.slice_indices[2],
                    &vol,
                );

                rs.renderer.write().callback_resources.insert(slice_resources);

                self.has_slices = true;
                self.nifti_path = Some(path.clone());
                self.nifti_volume = Some(vol);
                self.slices_dirty = false;
                self.error_msg = None;
            }
            Err(e) => {
                self.error_msg = Some(format!("Failed to load NIfTI: {e}"));
            }
        }
    }

    fn reset_slice_cameras(&mut self) {
        let half_extents = self
            .nifti_volume
            .as_ref()
            .map(|v| v.slice_half_extents())
            .unwrap_or([self.volume_extent * 0.5; 3]);
        self.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, self.volume_center, half_extents[0] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Coronal, self.volume_center, half_extents[1] * 2.0),
            OrthoSliceCamera::new(SliceAxis::Sagittal, self.volume_center, half_extents[2] * 2.0),
        ];
    }

    /// Draw crosshair lines on a 2D slice view showing the other two slice positions.
    fn draw_crosshairs(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        // Get the world-space positions of the other two slices
        let (other_a, other_b) = match axis_index {
            // Axial view: show coronal (Y) and sagittal (X) positions
            0 => (self.slice_world_position(2), self.slice_world_position(1)),
            // Coronal view: show sagittal (X) and axial (Z) positions
            1 => (self.slice_world_position(2), self.slice_world_position(0)),
            // Sagittal view: show coronal (Y) and axial (Z) positions
            _ => (self.slice_world_position(1), self.slice_world_position(0)),
        };

        let slice_pos = self.slice_world_position(axis_index);

        // Create world-space points on the crosshair lines and project them
        // For each crosshair line, we create two points at the extremes of the view
        let far = 10000.0;
        let (h_p1, h_p2, v_p1, v_p2) = match axis_index {
            0 => {
                // Axial: horizontal = coronal(Y), vertical = sagittal(X)
                let y = other_b; // coronal Y position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, y, slice_pos),
                    glam::Vec3::new(far, y, slice_pos),
                    glam::Vec3::new(x, -far, slice_pos),
                    glam::Vec3::new(x, far, slice_pos),
                )
            }
            1 => {
                // Coronal: horizontal = axial(Z), vertical = sagittal(X)
                let z = other_b; // axial Z position
                let x = other_a; // sagittal X position
                (
                    glam::Vec3::new(-far, slice_pos, z),
                    glam::Vec3::new(far, slice_pos, z),
                    glam::Vec3::new(x, slice_pos, -far),
                    glam::Vec3::new(x, slice_pos, far),
                )
            }
            _ => {
                // Sagittal: horizontal = axial(Z), vertical = coronal(Y)
                let z = other_b; // axial Z position
                let y = other_a; // coronal Y position
                (
                    glam::Vec3::new(slice_pos, -far, z),
                    glam::Vec3::new(slice_pos, far, z),
                    glam::Vec3::new(slice_pos, y, -far),
                    glam::Vec3::new(slice_pos, y, far),
                )
            }
        };

        let project = |world: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            // NDC [-1,1] → screen rect
            let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
            let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(); // flip Y
            egui::pos2(sx, sy)
        };

        let crosshair_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 128);
        let stroke = egui::Stroke::new(1.0, crosshair_color);
        let painter = ui.painter_at(rect);

        // Horizontal line (clipped to rect)
        painter.line_segment([project(h_p1), project(h_p2)], stroke);
        // Vertical line (clipped to rect)
        painter.line_segment([project(v_p1), project(v_p2)], stroke);
    }

    fn slice_world_position(&self, axis_index: usize) -> f32 {
        if let Some(vol) = &self.nifti_volume {
            let idx = self.slice_indices[axis_index] as f32;
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
            0.0
        }
    }
}

impl eframe::App for TrxViewerApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Update slice positions if dirty
        if self.slices_dirty {
            if let (Some(vol), Some(rs)) = (&self.nifti_volume, frame.wgpu_render_state()) {
                let renderer = rs.renderer.read();
                if let Some(sr) = renderer.callback_resources.get::<SliceResources>() {
                    sr.update_slice(&rs.queue, SliceAxis::Axial, self.slice_indices[0], vol);
                    sr.update_slice(&rs.queue, SliceAxis::Coronal, self.slice_indices[1], vol);
                    sr.update_slice(&rs.queue, SliceAxis::Sagittal, self.slice_indices[2], vol);
                }
            }
            self.slices_dirty = false;
        }

        // Update vertex colors if dirty
        if self.colors_dirty {
            if let (Some(data), Some(rs)) = (&self.trx_data, frame.wgpu_render_state()) {
                let renderer = rs.renderer.read();
                if let Some(sr) = renderer.callback_resources.get::<StreamlineResources>() {
                    sr.update_colors(&rs.queue, &data.colors);
                }
            }
            self.colors_dirty = false;
        }

        // Update index buffer if any filter changed
        if self.indices_dirty {
            if let (Some(data), Some(rs)) = (&self.trx_data, frame.wgpu_render_state()) {
                let indices = data.build_index_buffer(
                    &self.group_visible,
                    self.max_streamlines,
                    &self.streamline_order,
                    self.sphere_query_result.as_ref(),
                );
                let mut renderer = rs.renderer.write();
                if let Some(sr) = renderer.callback_resources.get_mut::<StreamlineResources>() {
                    sr.update_indices(&rs.device, &rs.queue, &indices);
                }
            }
            self.indices_dirty = false;
        }

        // ── Sidebar ──
        egui::SidePanel::left("sidebar")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("TRX Viewer");
                ui.separator();

                // File loading
                ui.label("Files");
                if ui.button("Open TRX...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("TRX files", &["trx"])
                        .pick_file()
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_trx(&path, rs);
                        }
                    }
                }
                if let Some(ref p) = self.trx_path {
                    ui.small(
                        p.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    );
                }

                if ui.button("Open NIfTI...").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("NIfTI files", &["nii", "nii.gz", "gz"])
                        .pick_file()
                    {
                        if let Some(rs) = frame.wgpu_render_state() {
                            self.load_nifti(&path, rs);
                        }
                    }
                }
                if let Some(ref p) = self.nifti_path {
                    ui.small(
                        p.file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_default(),
                    );
                }

                // Error display
                if let Some(ref msg) = self.error_msg {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, msg);
                }

                ui.separator();

                // TRX info
                if let Some(ref data) = self.trx_data {
                    ui.label("TRX Info");
                    ui.small(format!("Streamlines: {}", data.nb_streamlines));
                    ui.small(format!("Vertices: {}", data.nb_vertices));
                    ui.separator();
                }

                // Volume info
                if let Some(ref vol) = self.nifti_volume {
                    ui.label("Volume Info");
                    ui.small(format!(
                        "Dimensions: {}x{}x{}",
                        vol.dims[0], vol.dims[1], vol.dims[2]
                    ));
                    ui.separator();
                }

                // ── Coloring controls ──
                if self.trx_data.is_some() {
                    ui.label("Coloring");

                    // Collect available mode labels
                    let mut mode_labels = vec!["Direction RGB".to_string()];
                    let dpv_names: Vec<String> = self
                        .trx_data
                        .as_ref()
                        .map(|d| d.dpv_names.clone())
                        .unwrap_or_default();
                    let dps_names: Vec<String> = self
                        .trx_data
                        .as_ref()
                        .map(|d| d.dps_names.clone())
                        .unwrap_or_default();
                    let has_groups = self
                        .trx_data
                        .as_ref()
                        .map(|d| !d.groups.is_empty())
                        .unwrap_or(false);

                    for name in &dpv_names {
                        mode_labels.push(format!("DPV: {name}"));
                    }
                    for name in &dps_names {
                        mode_labels.push(format!("DPS: {name}"));
                    }
                    if has_groups {
                        mode_labels.push("Group".to_string());
                    }
                    mode_labels.push("Uniform".to_string());

                    // Find current selection index
                    let current_idx = match &self.color_mode {
                        ColorMode::DirectionRgb => 0,
                        ColorMode::Dpv(name) => {
                            1 + dpv_names.iter().position(|n| n == name).unwrap_or(0)
                        }
                        ColorMode::Dps(name) => {
                            1 + dpv_names.len()
                                + dps_names.iter().position(|n| n == name).unwrap_or(0)
                        }
                        ColorMode::Group => 1 + dpv_names.len() + dps_names.len(),
                        ColorMode::Uniform(_) => mode_labels.len() - 1,
                    };

                    let mut selected = current_idx;
                    egui::ComboBox::from_label("")
                        .selected_text(&mode_labels[current_idx])
                        .show_ui(ui, |ui| {
                            for (i, label) in mode_labels.iter().enumerate() {
                                ui.selectable_value(&mut selected, i, label);
                            }
                        });

                    if selected != current_idx {
                        let new_mode = if selected == 0 {
                            ColorMode::DirectionRgb
                        } else if selected <= dpv_names.len() {
                            ColorMode::Dpv(dpv_names[selected - 1].clone())
                        } else if selected <= dpv_names.len() + dps_names.len() {
                            ColorMode::Dps(
                                dps_names[selected - 1 - dpv_names.len()].clone(),
                            )
                        } else if has_groups
                            && selected == 1 + dpv_names.len() + dps_names.len()
                        {
                            ColorMode::Group
                        } else {
                            ColorMode::Uniform(self.uniform_color)
                        };

                        self.color_mode = new_mode.clone();
                        if let Some(data) = &mut self.trx_data {
                            data.recolor(&new_mode);
                        }
                        self.colors_dirty = true;
                    }

                    // Uniform color picker
                    if matches!(self.color_mode, ColorMode::Uniform(_)) {
                        let mut c = self.uniform_color;
                        if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                            self.uniform_color = c;
                            self.color_mode = ColorMode::Uniform(c);
                            if let Some(data) = &mut self.trx_data {
                                data.recolor(&ColorMode::Uniform(c));
                            }
                            self.colors_dirty = true;
                        }
                    }

                    ui.separator();
                }

                // ── Group visibility ──
                if self.trx_data.as_ref().map_or(false, |d| !d.groups.is_empty()) {
                    ui.label("Groups");

                    let group_info: Vec<(String, usize)> = self
                        .trx_data
                        .as_ref()
                        .map(|d| {
                            d.groups
                                .iter()
                                .map(|(name, members)| (name.clone(), members.len()))
                                .collect()
                        })
                        .unwrap_or_default();

                    for (i, (name, count)) in group_info.iter().enumerate() {
                        if i < self.group_visible.len() {
                            let label = format!("{name} ({count})");
                            if ui.checkbox(&mut self.group_visible[i], label).changed() {
                                self.indices_dirty = true;
                            }
                        }
                    }

                    ui.separator();
                }

                // ── Streamline filtering ──
                if let Some(ref data) = self.trx_data {
                    ui.label("Streamline Filter");
                    let nb = data.nb_streamlines;
                    ui.horizontal(|ui| {
                        ui.label("Max:");
                        if ui
                            .add(egui::Slider::new(&mut self.max_streamlines, 1..=nb))
                            .changed()
                        {
                            self.indices_dirty = true;
                        }
                    });
                    if ui
                        .checkbox(&mut self.use_random_subset, "Randomize")
                        .changed()
                    {
                        if self.use_random_subset {
                            // Simple xorshift shuffle with fixed seed
                            let mut order: Vec<u32> = (0..nb as u32).collect();
                            let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
                            for i in (1..order.len()).rev() {
                                rng ^= rng << 13;
                                rng ^= rng >> 7;
                                rng ^= rng << 17;
                                let j = (rng as usize) % (i + 1);
                                order.swap(i, j);
                            }
                            self.streamline_order = order;
                        } else {
                            self.streamline_order = (0..nb as u32).collect();
                        }
                        self.indices_dirty = true;
                    }

                    ui.horizontal(|ui| {
                        ui.label("Slab (mm):");
                        ui.add(egui::Slider::new(&mut self.slab_half_width, 0.5..=50.0));
                    });

                    ui.separator();

                    // ── Sphere query ──
                    ui.label("Sphere Query");
                    ui.small("Ctrl+click in slice view to place");
                    if self.sphere_query_active {
                        ui.horizontal(|ui| {
                            ui.label("Radius:");
                            if ui
                                .add(egui::Slider::new(&mut self.sphere_radius, 1.0..=50.0).suffix(" mm"))
                                .changed()
                            {
                                if let Some(ref data) = self.trx_data {
                                    self.sphere_query_result =
                                        Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                                }
                                self.indices_dirty = true;
                            }
                        });
                        let half = self.volume_extent * 0.6;
                        let mut sphere_moved = false;
                        ui.horizontal(|ui| {
                            ui.label("X:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.x,
                                    (self.volume_center.x - half)..=(self.volume_center.x + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Y:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.y,
                                    (self.volume_center.y - half)..=(self.volume_center.y + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Z:");
                            sphere_moved |= ui
                                .add(egui::Slider::new(
                                    &mut self.sphere_center.z,
                                    (self.volume_center.z - half)..=(self.volume_center.z + half),
                                ).suffix(" mm"))
                                .changed();
                        });
                        if sphere_moved {
                            if let Some(ref data) = self.trx_data {
                                self.sphere_query_result =
                                    Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                            }
                            self.indices_dirty = true;
                        }
                        if let Some(ref result) = self.sphere_query_result {
                            ui.small(format!("Matched: {} streamlines", result.len()));
                        }
                        if ui.button("Clear query").clicked() {
                            self.sphere_query_active = false;
                            self.sphere_query_result = None;
                            self.indices_dirty = true;
                        }
                    }

                    ui.separator();
                }

                // Slice controls
                if self.nifti_volume.is_some() {
                    ui.label("Slice Position");

                    let max_k;
                    let max_j;
                    let max_i;
                    if let Some(ref vol) = self.nifti_volume {
                        max_k = vol.dims[2].saturating_sub(1);
                        max_j = vol.dims[1].saturating_sub(1);
                        max_i = vol.dims[0].saturating_sub(1);
                    } else {
                        return;
                    }

                    ui.horizontal(|ui| {
                        ui.label("Axial:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[0], 0..=max_k))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Coronal:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[1], 0..=max_j))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Sagittal:");
                        if ui
                            .add(egui::Slider::new(&mut self.slice_indices[2], 0..=max_i))
                            .changed()
                        {
                            self.slices_dirty = true;
                        }
                    });

                    // Intensity windowing
                    ui.separator();
                    ui.label("Intensity Window");
                    ui.horizontal(|ui| {
                        ui.label("Center:");
                        ui.add(egui::Slider::new(&mut self.window_center, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Width:");
                        ui.add(egui::Slider::new(&mut self.window_width, 0.01..=2.0));
                    });
                }
            });

        // ── Main content: 4 viewports ──
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.trx_data.is_none() && self.nifti_volume.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a TRX or NIfTI file from the sidebar to begin.");
                });
                return;
            }

            let available = ui.available_size();
            let top_height = (available.y * 0.6).max(100.0);
            let bottom_height =
                (available.y - top_height - ui.spacing().item_spacing.y).max(50.0);

            // ── 3D Viewport ──
            let (rect_3d, response_3d) = ui.allocate_exact_size(
                egui::vec2(available.x, top_height),
                egui::Sense::click_and_drag(),
            );

            if response_3d.dragged_by(egui::PointerButton::Primary) {
                let delta = response_3d.drag_delta();
                self.camera_3d.handle_drag(delta.x, -delta.y);
            }
            if response_3d.hovered() {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera_3d.handle_scroll(scroll * 0.01);
                }
            }

            let aspect_3d = rect_3d.width() / rect_3d.height().max(1.0);
            let vp_3d = self.camera_3d.view_projection(aspect_3d);

            ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                rect_3d,
                Scene3DCallback {
                    view_proj: vp_3d,
                    has_streamlines: self.has_streamlines,
                    has_slices: self.has_slices,
                    window_center: self.window_center,
                    window_width: self.window_width,
                },
            ));

            // Draw 3D orientation axes
            self.draw_3d_axes(ui, rect_3d, vp_3d);

            // Draw 3D sphere indicator
            if self.sphere_query_active {
                self.draw_sphere_3d(ui, rect_3d, vp_3d);
            }

            // ── Bottom row: 3 slice views ──
            let slice_width = ((available.x - 2.0 * ui.spacing().item_spacing.x) / 3.0).max(10.0);
            let slice_height = (bottom_height - ui.spacing().item_spacing.y - 18.0).max(10.0);

            ui.horizontal(|ui| {
                let labels = ["Axial", "Coronal", "Sagittal"];
                for i in 0..3 {
                    ui.vertical(|ui| {
                        ui.label(labels[i]);
                        let (rect, response) = ui.allocate_exact_size(
                            egui::vec2(slice_width, slice_height),
                            egui::Sense::click_and_drag(),
                        );

                        if response.hovered() {
                            let scroll = ui.input(|inp| inp.smooth_scroll_delta.y);
                            if scroll.abs() > 0.5 {
                                if let Some(ref vol) = self.nifti_volume {
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
                                }
                            }
                        }

                        // Ctrl+click to place sphere query center
                        if response.clicked() && self.has_streamlines {
                            let ctrl = ui.input(|inp| inp.modifiers.ctrl || inp.modifiers.command);
                            if ctrl {
                                if let Some(pos) = response.interact_pointer_pos() {
                                    let aspect_q = rect.width() / rect.height().max(1.0);
                                    let sp = self.slice_world_position(i);
                                    let world = self.slice_cameras[i].screen_to_world(
                                        pos, rect, aspect_q, sp,
                                    );
                                    self.sphere_center = world;
                                    self.sphere_query_active = true;
                                    if let Some(ref data) = self.trx_data {
                                        self.sphere_query_result =
                                            Some(data.query_sphere(self.sphere_center, self.sphere_radius));
                                    }
                                    self.indices_dirty = true;
                                }
                            }
                        }

                        let aspect = rect.width() / rect.height().max(1.0);
                        let slice_pos = self.slice_world_position(i);
                        let vp_slice =
                            self.slice_cameras[i].view_projection(aspect, slice_pos);

                        // Slab axis: axial=Z(2), coronal=Y(1), sagittal=X(0)
                        let slab_axis = match i {
                            0 => 2u32, // axial → Z
                            1 => 1u32, // coronal → Y
                            _ => 0u32, // sagittal → X
                        };

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            SliceViewCallback {
                                view_proj: vp_slice,
                                quad_index: i,
                                bind_group_index: i + 1,
                                has_slices: self.has_slices,
                                has_streamlines: self.has_streamlines,
                                window_center: self.window_center,
                                window_width: self.window_width,
                                slab_axis,
                                slab_min: slice_pos - self.slab_half_width,
                                slab_max: slice_pos + self.slab_half_width,
                            },
                        ));

                        // Draw crosshairs showing other slice positions
                        if self.has_slices {
                            self.draw_crosshairs(ui, rect, i, vp_slice);
                        }

                        // Draw anatomical orientation labels
                        self.draw_orientation_labels(ui, rect, i, vp_slice);

                        // Draw sphere query circle
                        if self.sphere_query_active {
                            self.draw_sphere_circle(ui, rect, i, vp_slice, slice_pos);
                        }
                    });
                }
            });
        });
    }
}

impl TrxViewerApp {
    /// Draw the sphere query as a circle on a slice view.
    fn draw_sphere_circle(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        axis_index: usize,
        view_proj: glam::Mat4,
        slice_pos: f32,
    ) {
        // Get the sphere center coordinate along this slice's normal axis
        let center_on_axis = match axis_index {
            0 => self.sphere_center.z, // axial
            1 => self.sphere_center.y, // coronal
            _ => self.sphere_center.x, // sagittal
        };

        let d = (slice_pos - center_on_axis).abs();
        if d >= self.sphere_radius {
            return;
        }

        // Circle radius on this slice plane
        let circle_r = (self.sphere_radius * self.sphere_radius - d * d).sqrt();

        // Project sphere center to screen
        let clip = view_proj * self.sphere_center.extend(1.0);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = rect.left() + (ndc_x + 1.0) * 0.5 * rect.width();
        let sy = rect.top() + (1.0 - ndc_y) * 0.5 * rect.height();

        // Convert world-space radius to screen pixels
        // Use a point offset by circle_r in the first in-plane axis
        let offset_world = match axis_index {
            0 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            1 => self.sphere_center + Vec3::new(circle_r, 0.0, 0.0),
            _ => self.sphere_center + Vec3::new(0.0, circle_r, 0.0),
        };
        let clip2 = view_proj * offset_world.extend(1.0);
        let ndc_x2 = clip2.x / clip2.w;
        let ndc_y2 = clip2.y / clip2.w;
        let sx2 = rect.left() + (ndc_x2 + 1.0) * 0.5 * rect.width();
        let sy2 = rect.top() + (1.0 - ndc_y2) * 0.5 * rect.height();
        let screen_r = ((sx2 - sx).powi(2) + (sy2 - sy).powi(2)).sqrt();

        let painter = ui.painter_at(rect);
        let circle_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        painter.circle_stroke(
            egui::pos2(sx, sy),
            screen_r,
            egui::Stroke::new(2.0, circle_color),
        );
    }

    /// Draw three axis-aligned circles in the 3D view to indicate the sphere query position.
    fn draw_sphere_3d(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        view_proj: glam::Mat4,
    ) {
        let painter = ui.painter_at(rect);
        let color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 200);
        let stroke = egui::Stroke::new(1.5, color);
        let n = 48usize;
        let c = self.sphere_center;
        let r = self.sphere_radius;

        let project = |p: glam::Vec3| -> egui::Pos2 {
            let clip = view_proj * p.extend(1.0);
            let nx = clip.x / clip.w;
            let ny = clip.y / clip.w;
            egui::pos2(
                rect.left() + (nx + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ny) * 0.5 * rect.height(),
            )
        };

        // Three rings: XY (axial plane), XZ (coronal), YZ (sagittal)
        let ring_points = |axis_a: glam::Vec3, axis_b: glam::Vec3| -> Vec<egui::Pos2> {
            (0..=n)
                .map(|k| {
                    let t = k as f32 / n as f32 * std::f32::consts::TAU;
                    project(c + axis_a * (r * t.cos()) + axis_b * (r * t.sin()))
                })
                .collect()
        };

        for pts in [
            ring_points(glam::Vec3::X, glam::Vec3::Y), // XY plane
            ring_points(glam::Vec3::X, glam::Vec3::Z), // XZ plane
            ring_points(glam::Vec3::Y, glam::Vec3::Z), // YZ plane
        ] {
            for w in pts.windows(2) {
                painter.line_segment([w[0], w[1]], stroke);
            }
        }
        // Small crosshair at center
        let cp = project(c);
        let arm = 6.0;
        painter.line_segment([egui::pos2(cp.x - arm, cp.y), egui::pos2(cp.x + arm, cp.y)], stroke);
        painter.line_segment([egui::pos2(cp.x, cp.y - arm), egui::pos2(cp.x, cp.y + arm)], stroke);
    }

    /// Draw anatomical orientation labels (R/L/A/P/S/I) on a slice view.
    fn draw_orientation_labels(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        _axis_index: usize,
        view_proj: glam::Mat4,
    ) {
        // Project world-space direction endpoints to screen space
        let far = 10000.0;
        let center = self.volume_center;

        // Define the 6 anatomical directions with labels
        let directions: &[(Vec3, &str)] = &[
            (Vec3::new(far, center.y, center.z), "R"),   // +X = Right
            (Vec3::new(-far, center.y, center.z), "L"),   // -X = Left
            (Vec3::new(center.x, far, center.z), "A"),   // +Y = Anterior
            (Vec3::new(center.x, -far, center.z), "P"),   // -Y = Posterior
            (Vec3::new(center.x, center.y, far), "S"),   // +Z = Superior
            (Vec3::new(center.x, center.y, -far), "I"),   // -Z = Inferior
        ];

        let project = |world: Vec3| -> egui::Pos2 {
            let clip = view_proj * world.extend(1.0);
            let ndc_x = clip.x / clip.w;
            let ndc_y = clip.y / clip.w;
            egui::pos2(
                rect.left() + (ndc_x + 1.0) * 0.5 * rect.width(),
                rect.top() + (1.0 - ndc_y) * 0.5 * rect.height(),
            )
        };

        let painter = ui.painter_at(rect);
        let label_color = egui::Color32::from_rgb(220, 220, 220);
        let font = egui::FontId::proportional(14.0);
        let margin = 16.0;

        // For each direction, check if the projected point is on a rect edge
        for &(dir_point, label) in directions {
            let p = project(dir_point);
            // Only draw if the point projects to near an edge of the rect
            let on_left = (p.x - rect.left()).abs() < margin * 2.0;
            let on_right = (p.x - rect.right()).abs() < margin * 2.0;
            let on_top = (p.y - rect.top()).abs() < margin * 2.0;
            let on_bottom = (p.y - rect.bottom()).abs() < margin * 2.0;

            if !(on_left || on_right || on_top || on_bottom) {
                continue; // This direction doesn't project to an edge (it's the look axis)
            }

            // Clamp to rect edges with margin
            let label_pos = egui::pos2(
                p.x.clamp(rect.left() + margin, rect.right() - margin),
                p.y.clamp(rect.top() + margin, rect.bottom() - margin),
            );

            painter.text(
                label_pos,
                egui::Align2::CENTER_CENTER,
                label,
                font.clone(),
                label_color,
            );
        }
    }

    /// Draw 3D orientation axes in the corner of the 3D viewport.
    fn draw_3d_axes(&self, ui: &egui::Ui, rect: egui::Rect, view_proj: glam::Mat4) {
        let painter = ui.painter_at(rect);

        // Place axes in bottom-left corner
        let origin_screen = egui::pos2(rect.left() + 50.0, rect.bottom() - 50.0);
        let axis_length = 30.0;

        let axes = [
            (Vec3::X, "R", egui::Color32::RED),
            (Vec3::Y, "A", egui::Color32::GREEN),
            (Vec3::Z, "S", egui::Color32::from_rgb(80, 120, 255)),
        ];

        for (dir, label, color) in axes {
            // Project the direction vector (just the rotation, no translation)
            let clip0 = view_proj * Vec3::ZERO.extend(1.0);
            let clip1 = view_proj * dir.extend(1.0);
            // Direction in NDC
            let ndc0 = egui::vec2(clip0.x / clip0.w, clip0.y / clip0.w);
            let ndc1 = egui::vec2(clip1.x / clip1.w, clip1.y / clip1.w);
            let dir_ndc = ndc1 - ndc0;
            let dir_screen = egui::vec2(dir_ndc.x, -dir_ndc.y); // flip Y
            let dir_norm = if dir_screen.length() > 0.001 {
                dir_screen / dir_screen.length()
            } else {
                egui::vec2(0.0, 0.0)
            };

            let end = origin_screen + dir_norm * axis_length;
            painter.line_segment(
                [origin_screen, end],
                egui::Stroke::new(2.0, color),
            );
            painter.text(
                end + dir_norm * 10.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::proportional(12.0),
                color,
            );
        }
    }
}

// ── Paint Callbacks ──

struct Scene3DCallback {
    view_proj: glam::Mat4,
    has_streamlines: bool,
    has_slices: bool,
    window_center: f32,
    window_width: f32,
}

impl egui_wgpu::CallbackTrait for Scene3DCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get_mut::<StreamlineResources>() {
            res.update_uniforms(queue, 0, self.view_proj, 3, 0.0, 0.0);
        }
        if let Some(res) = callback_resources.get_mut::<SliceResources>() {
            res.update_uniforms(queue, 0, self.view_proj, self.window_center, self.window_width);
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if self.has_slices {
            if let Some(sr) = callback_resources.get::<SliceResources>() {
                render_pass.set_pipeline(&sr.pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                render_pass.set_index_buffer(
                    sr.quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                for i in 0..3 {
                    render_pass.set_vertex_buffer(0, sr.quad_buffers[i].slice(..));
                    render_pass.draw_indexed(0..6, 0, 0..1);
                }
            }
        }

        if self.has_streamlines {
            if let Some(sr) = callback_resources.get::<StreamlineResources>() {
                render_pass.set_pipeline(&sr.pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[0], &[]);
                render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                render_pass.set_index_buffer(
                    sr.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
            }
        }
    }
}

struct SliceViewCallback {
    view_proj: glam::Mat4,
    quad_index: usize,
    bind_group_index: usize,
    has_slices: bool,
    has_streamlines: bool,
    window_center: f32,
    window_width: f32,
    /// Slab clipping axis for streamlines: 0=X, 1=Y, 2=Z.
    slab_axis: u32,
    slab_min: f32,
    slab_max: f32,
}

impl egui_wgpu::CallbackTrait for SliceViewCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(res) = callback_resources.get_mut::<SliceResources>() {
            res.update_uniforms(queue, self.bind_group_index, self.view_proj, self.window_center, self.window_width);
        }
        if let Some(res) = callback_resources.get_mut::<StreamlineResources>() {
            res.update_uniforms(
                queue,
                self.bind_group_index,
                self.view_proj,
                self.slab_axis,
                self.slab_min,
                self.slab_max,
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let vp = info.viewport_in_pixels();
        if vp.width_px == 0 || vp.height_px == 0 {
            return;
        }
        render_pass.set_viewport(
            vp.left_px as f32,
            vp.top_px as f32,
            vp.width_px as f32,
            vp.height_px as f32,
            0.0,
            1.0,
        );

        if self.has_slices {
            if let Some(sr) = callback_resources.get::<SliceResources>() {
                render_pass.set_pipeline(&sr.pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                render_pass.set_index_buffer(
                    sr.quad_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint16,
                );
                render_pass.set_vertex_buffer(0, sr.quad_buffers[self.quad_index].slice(..));
                render_pass.draw_indexed(0..6, 0, 0..1);
            }
        }

        if self.has_streamlines {
            if let Some(sr) = callback_resources.get::<StreamlineResources>() {
                render_pass.set_pipeline(&sr.slice_pipeline);
                render_pass.set_bind_group(0, &sr.bind_groups[self.bind_group_index], &[]);
                render_pass.set_vertex_buffer(0, sr.position_buffer.slice(..));
                render_pass.set_vertex_buffer(1, sr.color_buffer.slice(..));
                render_pass.set_index_buffer(
                    sr.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..sr.num_indices, 0, 0..1);
            }
        }
    }
}
