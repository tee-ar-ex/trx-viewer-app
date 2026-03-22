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
    /// Whether the index buffer needs rebuild due to group visibility change.
    groups_dirty: bool,
    /// Uniform color picker value.
    uniform_color: [f32; 4],
    /// Volume intensity windowing.
    window_center: f32,
    window_width: f32,
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
            groups_dirty: false,
            uniform_color: [1.0, 1.0, 1.0, 1.0],
            window_center: 0.5,
            window_width: 1.0,
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
        self.slice_cameras = [
            OrthoSliceCamera::new(SliceAxis::Axial, self.volume_center, self.volume_extent),
            OrthoSliceCamera::new(SliceAxis::Coronal, self.volume_center, self.volume_extent),
            OrthoSliceCamera::new(SliceAxis::Sagittal, self.volume_center, self.volume_extent),
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

        // Update index buffer if group visibility changed
        if self.groups_dirty {
            if let (Some(data), Some(rs)) = (&self.trx_data, frame.wgpu_render_state()) {
                let indices = data.indices_for_visible_groups(&self.group_visible);
                let mut renderer = rs.renderer.write();
                if let Some(sr) = renderer.callback_resources.get_mut::<StreamlineResources>() {
                    sr.update_indices(&rs.device, &rs.queue, &indices);
                }
            }
            self.groups_dirty = false;
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
                                self.groups_dirty = true;
                            }
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

                        if response.dragged_by(egui::PointerButton::Primary) {
                            let delta = response.drag_delta();
                            self.slice_cameras[i]
                                .handle_drag(-delta.x, delta.y, rect.width());
                        }

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

                        // Right-click drag to zoom slice views
                        if response.dragged_by(egui::PointerButton::Secondary) {
                            let delta = response.drag_delta();
                            self.slice_cameras[i].handle_zoom(delta.y * 0.01);
                        }

                        let aspect = rect.width() / rect.height().max(1.0);
                        let slice_pos = self.slice_world_position(i);
                        let vp_slice =
                            self.slice_cameras[i].view_projection(aspect, slice_pos);

                        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                            rect,
                            SliceViewCallback {
                                view_proj: vp_slice,
                                quad_index: i,
                                bind_group_index: i + 1,
                                has_slices: self.has_slices,
                                window_center: self.window_center,
                                window_width: self.window_width,
                            },
                        ));

                        // Draw crosshairs showing other slice positions
                        if self.has_slices {
                            self.draw_crosshairs(ui, rect, i, vp_slice);
                        }
                    });
                }
            });
        });
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
            res.update_uniforms(queue, self.view_proj);
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
                render_pass.set_bind_group(0, &sr.bind_group, &[]);
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
    window_center: f32,
    window_width: f32,
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
        Vec::new()
    }

    fn paint(
        &self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if !self.has_slices {
            return;
        }
        let Some(sr) = callback_resources.get::<SliceResources>() else {
            return;
        };

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
