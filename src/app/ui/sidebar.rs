use super::super::helpers::scalar_range_opt;
use super::super::state::{BundleMeshSource, SurfaceProjectionMode};
use crate::data::loaded_files::VolumeColormap;
use crate::data::trx_data::{colormap_bwr, ColorMode, RenderStyle};
use crate::renderer::mesh_renderer::{MeshResources, SurfaceColormap};
use crate::renderer::slice_renderer::AllSliceResources;
use crate::renderer::streamline_renderer::AllStreamlineResources;

impl super::super::TrxViewerApp {
    pub(in crate::app) fn show_sidebar(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::SidePanel::left("sidebar")
            .default_width(280.0)
            .min_width(280.0)
            .max_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                    ui.heading("TRX Viewer");
                    ui.separator();

                    // Error display (always visible, not in a header)
                    if let Some(ref msg) = self.error_msg {
                        ui.colored_label(egui::Color32::RED, msg);
                        ui.separator();
                    }

                    // ── Volumes ──
                    if !self.nifti_files.is_empty() {
                        egui::CollapsingHeader::new("Volumes")
                            .default_open(true)
                            .show(ui, |ui| {
                                self.show_volumes_section(ui, frame);
                            });
                    }

                    // ── Surfaces ──
                    if !self.gifti_surfaces.is_empty() {
                        egui::CollapsingHeader::new("Surfaces")
                            .default_open(true)
                            .show(ui, |ui| {
                                self.show_surfaces_section(ui, frame);
                            });
                    }

                    // ── Streamlines (per TRX file) ──
                    if !self.trx_files.is_empty() {
                        ui.checkbox(&mut self.show_streamlines, "Show streamlines");

                        for trx_idx in 0..self.trx_files.len() {
                            let trx_name = self.trx_files[trx_idx].name.clone();
                            let header_id = ui.make_persistent_id(format!("trx_{}", self.trx_files[trx_idx].id));
                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                ui.ctx(), header_id, true,
                            )
                            .show_header(ui, |ui| {
                                ui.label(&trx_name);
                            })
                            .body(ui, |ui| {
                                self.show_trx_section(ui, frame, trx_idx);
                            });
                        }
                    }

                    // ── Sphere Query ──
                    if self.has_streamlines() {
                        egui::CollapsingHeader::new("Sphere Query")
                            .default_open(true)
                            .show(ui, |ui| {
                                self.show_sphere_query_section(ui);
                            });
                    }

                    // ── Slice Position ──
                    if !self.nifti_files.is_empty() {
                        egui::CollapsingHeader::new("Slice Position")
                            .default_open(true)
                            .show(ui, |ui| {
                                self.show_slice_position_section(ui);
                            });
                    }
                });
            });
    }

    fn show_volumes_section(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let mut remove_id: Option<usize> = None;
        for nf in self.nifti_files.iter_mut() {
            let header_id = ui.make_persistent_id(format!("vol_{}", nf.id));
            egui::collapsing_header::CollapsingState::load_with_default_open(
                ui.ctx(), header_id, true,
            )
            .show_header(ui, |ui| {
                ui.label(&nf.name);
            })
            .body(ui, |ui| {
                    ui.small(format!(
                        "Dims: {}x{}x{}",
                        nf.volume.dims[0], nf.volume.dims[1], nf.volume.dims[2]
                    ));
                    ui.checkbox(&mut nf.visible, "Visible");

                    ui.horizontal(|ui| {
                        ui.label("Colormap:");
                        egui::ComboBox::from_id_salt(format!("cmap_{}", nf.id))
                            .selected_text(nf.colormap.label())
                            .show_ui(ui, |ui| {
                                for cmap in VolumeColormap::ALL {
                                    ui.selectable_value(&mut nf.colormap, *cmap, cmap.label());
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Opacity:");
                        ui.add(egui::Slider::new(&mut nf.opacity, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Z-order:");
                        ui.add(egui::DragValue::new(&mut nf.z_order));
                    });

                    ui.add_space(4.0);
                    ui.label("Intensity Window");
                    ui.horizontal(|ui| {
                        ui.label("Center:");
                        ui.add(egui::Slider::new(&mut nf.window_center, 0.0..=1.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Width:");
                        ui.add(egui::Slider::new(&mut nf.window_width, 0.01..=2.0));
                    });

                    if ui.button("Remove").clicked() {
                        remove_id = Some(nf.id);
                    }
                });
        }
        if let Some(rid) = remove_id {
            self.nifti_files.retain(|n| n.id != rid);
            if let Some(rs) = frame.wgpu_render_state() {
                let mut renderer = rs.renderer.write();
                if let Some(all) = renderer.callback_resources.get_mut::<AllSliceResources>() {
                    all.entries.retain(|(id, _)| *id != rid);
                }
            }
        }
    }

    fn show_surfaces_section(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // Collect DPS names from all TRX files
        let dps_names_all: Vec<String> = self.trx_files.iter()
            .flat_map(|t| t.data.dps_names.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let mut query_changed = false;
        let mut projection_changed = false;

        ui.group(|ui| {
            ui.checkbox(&mut self.surface_query_active, "Use surface depth filter");
            ui.horizontal(|ui| {
                ui.label("Filter surface");
                let current = self
                    .gifti_surfaces
                    .get(self.surface_query_surface)
                    .map(|s| s.name.clone())
                    .unwrap_or_else(|| "none".to_string());
                egui::ComboBox::from_id_salt("surface_query_surface")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        for (i, s) in self.gifti_surfaces.iter().enumerate() {
                            if ui
                                .selectable_value(&mut self.surface_query_surface, i, &s.name)
                                .changed()
                            {
                                query_changed = true;
                            }
                        }
                    });
            });
            if self.surface_query_surface < self.gifti_surfaces.len() {
                let depth = &mut self.gifti_surfaces[self.surface_query_surface].projection_depth_mm;
                if ui
                    .add(egui::Slider::new(depth, 0.1..=20.0).text("Depth mm"))
                    .changed()
                {
                    query_changed = true;
                    projection_changed = true;
                }
            }
        });

        for (surface_idx, surface) in self.gifti_surfaces.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut surface.visible, "");
                });
                ui.label(&surface.name);
                ui.horizontal(|ui| {
                    ui.label("Opacity");
                    ui.add(egui::Slider::new(&mut surface.opacity, 0.0..=1.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_rgb(&mut surface.color);
                });
                ui.horizontal(|ui| {
                    ui.label("Projection");
                    if ui
                        .checkbox(&mut surface.show_projection_map, "Show map")
                        .changed()
                    {
                        projection_changed = true;
                    }
                    egui::ComboBox::from_id_salt(format!("proj_mode_{surface_idx}"))
                        .selected_text(match surface.projection_mode {
                            SurfaceProjectionMode::Density => "Density",
                            SurfaceProjectionMode::MeanDps => "Mean DPS",
                        })
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(
                                    &mut surface.projection_mode,
                                    SurfaceProjectionMode::Density,
                                    "Density",
                                )
                                .changed()
                            {
                                projection_changed = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut surface.projection_mode,
                                    SurfaceProjectionMode::MeanDps,
                                    "Mean DPS",
                                )
                                .changed()
                            {
                                projection_changed = true;
                            }
                        });
                });
                if matches!(surface.projection_mode, SurfaceProjectionMode::MeanDps) {
                    let current = surface
                        .projection_dps
                        .clone()
                        .unwrap_or_else(|| "Select DPS".to_string());
                    ui.horizontal(|ui| {
                        ui.label("DPS");
                        egui::ComboBox::from_id_salt(format!("proj_dps_{surface_idx}"))
                            .selected_text(current)
                            .show_ui(ui, |ui| {
                                for name in &dps_names_all {
                                    if ui
                                        .selectable_label(
                                            surface.projection_dps.as_ref() == Some(name),
                                            name,
                                        )
                                        .clicked()
                                    {
                                        surface.projection_dps = Some(name.clone());
                                        projection_changed = true;
                                    }
                                }
                            });
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Colormap");
                    egui::ComboBox::from_id_salt(format!("proj_cmap_{surface_idx}"))
                        .selected_text(match surface.projection_colormap {
                            SurfaceColormap::BlueWhiteRed => "Blue-White-Red",
                            SurfaceColormap::Viridis => "Viridis",
                            SurfaceColormap::Inferno => "Inferno",
                        })
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_value(
                                    &mut surface.projection_colormap,
                                    SurfaceColormap::BlueWhiteRed,
                                    "Blue-White-Red",
                                )
                                .changed()
                            {
                                projection_changed = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut surface.projection_colormap,
                                    SurfaceColormap::Viridis,
                                    "Viridis",
                                )
                                .changed()
                            {
                                projection_changed = true;
                            }
                            if ui
                                .selectable_value(
                                    &mut surface.projection_colormap,
                                    SurfaceColormap::Inferno,
                                    "Inferno",
                                )
                                .changed()
                            {
                                projection_changed = true;
                            }
                        });
                });
                if surface.show_projection_map {
                    ui.horizontal(|ui| {
                        ui.label("Map opacity");
                        if ui
                            .add(egui::Slider::new(&mut surface.map_opacity, 0.0..=1.0))
                            .changed()
                        {
                            projection_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Map threshold");
                        if ui
                            .add(egui::Slider::new(&mut surface.map_threshold, 0.0..=1.0))
                            .changed()
                        {
                            projection_changed = true;
                        }
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Ambient");
                    if ui
                        .add(egui::Slider::new(&mut surface.surface_ambient, 0.0..=1.0))
                        .changed()
                    {
                        projection_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Gloss");
                    if ui
                        .add(egui::Slider::new(&mut surface.surface_gloss, 0.0..=1.0))
                        .changed()
                    {
                        projection_changed = true;
                    }
                });
                ui.horizontal(|ui| {
                    if ui.checkbox(&mut surface.auto_range, "Auto range").changed() {
                        projection_changed = true;
                    }
                    if ui.button("Recompute").clicked() {
                        projection_changed = true;
                    }
                });
                if !surface.auto_range {
                    ui.horizontal(|ui| {
                        ui.label("Min");
                        if ui
                            .add(egui::DragValue::new(&mut surface.range_min).speed(0.01))
                            .changed()
                        {
                            projection_changed = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Max");
                        if ui
                            .add(egui::DragValue::new(&mut surface.range_max).speed(0.01))
                            .changed()
                        {
                            projection_changed = true;
                        }
                    });
                }
                ui.small(
                    surface
                        .path
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                );
            });
        }
        if query_changed {
            self.recompute_surface_query();
        }
        if projection_changed {
            self.surface_projection_dirty = true;
        }
    }

    fn show_trx_section(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, trx_idx: usize) {
        let trx = &self.trx_files[trx_idx];

        // TRX info
        ui.small(format!("Streamlines: {}", trx.data.nb_streamlines));
        ui.small(format!("Vertices: {}", trx.data.nb_vertices));
        ui.small(
            trx.path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
        );
        // Need mut access
        ui.checkbox(&mut self.trx_files[trx_idx].visible, "Visible");

        ui.add_space(4.0);

        // ── Coloring controls ──
        self.show_coloring_controls(ui, trx_idx);

        ui.add_space(4.0);

        // ── Render style ──
        {
            let trx = &mut self.trx_files[trx_idx];
            ui.label("Render Style");
            let styles = [
                (RenderStyle::Flat,        "Flat lines"),
                (RenderStyle::Illuminated, "Illuminated"),
                (RenderStyle::Tubes,       "Tube impostors"),
                (RenderStyle::DepthCue,    "Depth cue"),
            ];
            let current_label = styles.iter()
                .find(|(s, _)| *s == trx.render_style)
                .map(|(_, l)| *l)
                .unwrap_or("Flat lines");

            egui::ComboBox::from_id_salt(format!("render_style_{}", trx.id))
                .selected_text(current_label)
                .show_ui(ui, |ui| {
                    for (style, label) in &styles {
                        if ui.selectable_value(&mut trx.render_style, *style, *label).changed() {
                            trx.indices_dirty = true;
                        }
                    }
                });

            if trx.render_style == RenderStyle::Tubes {
                ui.horizontal(|ui| {
                    ui.label("Radius (mm):");
                    if ui.add(egui::Slider::new(&mut trx.tube_radius, 0.1..=3.0).step_by(0.05)).changed() {
                        trx.indices_dirty = true;
                    }
                });
            }
        }

        ui.add_space(4.0);

        // ── Filters ──
        self.show_filters_section_for_trx(ui, trx_idx);

        ui.add_space(4.0);

        // ── Bundle surface mesh ──
        self.show_bundle_mesh_section(ui, frame, trx_idx);

        ui.add_space(4.0);

        // ── Group visibility ──
        {
            let trx = &mut self.trx_files[trx_idx];
            if !trx.data.groups.is_empty() {
                ui.label("Groups");

                let group_info: Vec<(String, usize)> = trx.data.groups
                    .iter()
                    .map(|(name, members)| (name.clone(), members.len()))
                    .collect();

                for (gi, (name, count)) in group_info.iter().enumerate() {
                    if gi < trx.group_visible.len() {
                        let label = format!("{name} ({count})");
                        if ui.checkbox(&mut trx.group_visible[gi], label).changed() {
                            trx.indices_dirty = true;
                        }
                    }
                }
            }
        }

        // ── Remove button ──
        if ui.button("Remove TRX").clicked() {
            let id = self.trx_files[trx_idx].id;
            self.trx_files.remove(trx_idx);
            if let Some(rs) = frame.wgpu_render_state() {
                let mut renderer = rs.renderer.write();
                if let Some(all) = renderer.callback_resources.get_mut::<AllStreamlineResources>() {
                    all.entries.retain(|(fid, _)| *fid != id);
                }
            }
        }
    }

    fn show_coloring_controls(&mut self, ui: &mut egui::Ui, trx_idx: usize) {
        let trx = &self.trx_files[trx_idx];

        ui.label("Coloring");

        let mut mode_labels = vec!["Direction RGB".to_string()];
        let dpv_names: Vec<String> = trx.data.dpv_names.clone();
        let dps_names: Vec<String> = trx.data.dps_names.clone();
        let has_groups = !trx.data.groups.is_empty();
        let trx_id = trx.id;

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

        let current_idx = match &self.trx_files[trx_idx].color_mode {
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
        egui::ComboBox::from_id_salt(format!("color_mode_{trx_id}"))
            .selected_text(&mode_labels[current_idx])
            .show_ui(ui, |ui| {
                for (i, label) in mode_labels.iter().enumerate() {
                    ui.selectable_value(&mut selected, i, label);
                }
            });

        if selected != current_idx {
            let uniform_color = self.trx_files[trx_idx].uniform_color;
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
                ColorMode::Uniform(uniform_color)
            };

            let trx = &mut self.trx_files[trx_idx];
            trx.color_mode = new_mode.clone();
            if trx.scalar_auto_range {
                if let Some((lo, hi)) = trx.data.scalar_range_for_mode(&new_mode) {
                    trx.scalar_range_min = lo;
                    trx.scalar_range_max = hi;
                }
            }
            let range = scalar_range_opt(
                &new_mode, trx.scalar_auto_range,
                trx.scalar_range_min, trx.scalar_range_max,
            );
            trx.data.recolor(&new_mode, range);
            trx.colors_dirty = true;
        }

        // Uniform color picker
        if matches!(self.trx_files[trx_idx].color_mode, ColorMode::Uniform(_)) {
            let mut c = self.trx_files[trx_idx].uniform_color;
            if ui.color_edit_button_rgba_unmultiplied(&mut c).changed() {
                let trx = &mut self.trx_files[trx_idx];
                trx.uniform_color = c;
                trx.color_mode = ColorMode::Uniform(c);
                trx.data.recolor(&ColorMode::Uniform(c), None);
                trx.colors_dirty = true;
            }
        }

        // ── Colorbar (DPV / DPS only) ──
        let is_scalar = matches!(self.trx_files[trx_idx].color_mode, ColorMode::Dpv(_) | ColorMode::Dps(_));
        if is_scalar {
            ui.add_space(4.0);

            let bar_w = ui.available_width();
            let bar_h = 14.0;
            let (bar_rect, _) = ui.allocate_exact_size(
                egui::vec2(bar_w, bar_h), egui::Sense::hover(),
            );
            let painter = ui.painter_at(bar_rect);
            let n = 64usize;
            let sw = bar_w / n as f32;
            for ci in 0..n {
                let t = ci as f32 / (n - 1) as f32;
                let [r, g, b, _] = colormap_bwr(t);
                let col = egui::Color32::from_rgb(
                    (r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8,
                );
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(bar_rect.left() + ci as f32 * sw, bar_rect.top()),
                        egui::vec2(sw + 1.0, bar_h),
                    ),
                    0.0, col,
                );
            }

            let mut range_changed = false;
            let trx = &mut self.trx_files[trx_idx];
            ui.horizontal(|ui| {
                if ui.checkbox(&mut trx.scalar_auto_range, "Auto").changed()
                    && trx.scalar_auto_range
                {
                    if let Some((lo, hi)) = trx.data.scalar_range_for_mode(&trx.color_mode) {
                        trx.scalar_range_min = lo;
                        trx.scalar_range_max = hi;
                        range_changed = true;
                    }
                }
            });
            let trx = &mut self.trx_files[trx_idx];
            ui.horizontal(|ui| {
                ui.label("Min");
                let resp = ui.add_enabled(
                    !trx.scalar_auto_range,
                    egui::DragValue::new(&mut trx.scalar_range_min).speed(0.01),
                );
                if resp.changed() { range_changed = true; }
                ui.label("Max");
                let resp = ui.add_enabled(
                    !trx.scalar_auto_range,
                    egui::DragValue::new(&mut trx.scalar_range_max).speed(0.01),
                );
                if resp.changed() { range_changed = true; }
            });

            if range_changed {
                let trx = &mut self.trx_files[trx_idx];
                let range = Some((trx.scalar_range_min, trx.scalar_range_max));
                let mode = trx.color_mode.clone();
                trx.data.recolor(&mode, range);
                trx.colors_dirty = true;
            }
        }
    }

    fn show_filters_section_for_trx(&mut self, ui: &mut egui::Ui, trx_idx: usize) {
        let trx = &mut self.trx_files[trx_idx];
        let nb = trx.data.nb_streamlines;
        ui.horizontal(|ui| {
            ui.label("Max:");
            if ui
                .add(egui::Slider::new(&mut trx.max_streamlines, 1..=nb))
                .changed()
            {
                trx.indices_dirty = true;
            }
        });
        if ui
            .checkbox(&mut trx.use_random_subset, "Randomize")
            .changed()
        {
            if trx.use_random_subset {
                let mut order: Vec<u32> = (0..nb as u32).collect();
                let mut rng: u64 = 0xDEAD_BEEF_CAFE_BABE;
                for i in (1..order.len()).rev() {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    let j = (rng as usize) % (i + 1);
                    order.swap(i, j);
                }
                trx.streamline_order = order;
            } else {
                trx.streamline_order = (0..nb as u32).collect();
            }
            trx.indices_dirty = true;
        }

        ui.horizontal(|ui| {
            ui.label("Slab (mm):");
            ui.add(egui::Slider::new(&mut trx.slab_half_width, 0.5..=50.0));
        });
    }

    fn show_bundle_mesh_section(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame, trx_idx: usize) {
        let trx = &mut self.trx_files[trx_idx];
        ui.label("Bundle Surface Mesh");
        let mut rebuild = false;
        let mesh_toggled = ui.checkbox(&mut trx.show_bundle_mesh, "Show surface").changed();
        if mesh_toggled {
            if trx.show_bundle_mesh {
                rebuild = true;
            } else {
                let file_id = trx.id;
                if let Some(rs) = frame.wgpu_render_state() {
                    if let Some(mr) = rs.renderer.write()
                        .callback_resources.get_mut::<MeshResources>()
                    {
                        mr.clear_bundle_mesh(file_id);
                        trx.bundle_meshes_cpu.clear();
                    }
                }
            }
        }

        if trx.show_bundle_mesh {
            let src_label = match trx.bundle_mesh_source {
                BundleMeshSource::All       => "All streamlines",
                BundleMeshSource::Selection => "Current selection",
                BundleMeshSource::PerGroup  => "Per group",
            };
            egui::ComboBox::from_id_salt(format!("bundle_src_{}", trx.id))
                .selected_text(src_label)
                .show_ui(ui, |ui| {
                    for (variant, label) in [
                        (BundleMeshSource::All,       "All streamlines"),
                        (BundleMeshSource::Selection, "Current selection"),
                        (BundleMeshSource::PerGroup,  "Per group"),
                    ] {
                        if ui.selectable_value(&mut trx.bundle_mesh_source, variant, label).changed() {
                            rebuild = true;
                        }
                    }
                });

            ui.horizontal(|ui| {
                ui.label("Voxel (mm):");
                rebuild |= ui.add(
                    egui::Slider::new(&mut trx.bundle_mesh_voxel_size, 0.5..=10.0).step_by(0.5)
                ).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Threshold:");
                rebuild |= ui.add(
                    egui::Slider::new(&mut trx.bundle_mesh_threshold, 1.0..=50.0).step_by(1.0)
                ).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Smooth:");
                rebuild |= ui.add(
                    egui::Slider::new(&mut trx.bundle_mesh_smooth, 0.0..=4.0).step_by(0.25)
                ).changed();
            });
            ui.horizontal(|ui| {
                ui.label("Opacity:");
                ui.add(egui::Slider::new(&mut trx.bundle_mesh_opacity, 0.0..=1.0));
            });
            ui.horizontal(|ui| {
                ui.label("Ambient:");
                ui.add(egui::Slider::new(&mut trx.bundle_mesh_ambient, 0.0..=1.0));
            });

            if rebuild {
                trx.bundle_mesh_dirty_at = Some(std::time::Instant::now());
            }

            if ui.button("Rebuild now").clicked() {
                trx.bundle_mesh_dirty_at = Some(
                    std::time::Instant::now()
                        - std::time::Duration::from_millis(200),
                );
            }

            let building = trx.bundle_mesh_dirty_at.is_some() || trx.bundle_mesh_pending.is_some();
            ui.add_enabled(false, egui::Label::new(
                egui::RichText::new(if building { "Building..." } else { " " }).small()
            ));
        }
    }

    fn show_sphere_query_section(&mut self, ui: &mut egui::Ui) {
        ui.small("Ctrl+click in slice view to place");
        if self.sphere_query_active {
            ui.horizontal(|ui| {
                ui.label("Radius:");
                if ui
                    .add(egui::Slider::new(&mut self.sphere_radius, 1.0..=50.0).suffix(" mm"))
                    .changed()
                {
                    for trx in &mut self.trx_files {
                        trx.sphere_query_result =
                            Some(trx.data.query_sphere(self.sphere_center, self.sphere_radius));
                        trx.indices_dirty = true;
                    }
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
                for trx in &mut self.trx_files {
                    trx.sphere_query_result =
                        Some(trx.data.query_sphere(self.sphere_center, self.sphere_radius));
                    trx.indices_dirty = true;
                }
            }
            // Show total matched count across all TRX files
            let total_matched: usize = self.trx_files.iter()
                .filter_map(|t| t.sphere_query_result.as_ref().map(|r| r.len()))
                .sum();
            if total_matched > 0 {
                ui.small(format!("Matched: {} streamlines", total_matched));
            }
            if ui.button("Clear query").clicked() {
                self.sphere_query_active = false;
                for trx in &mut self.trx_files {
                    trx.sphere_query_result = None;
                    trx.indices_dirty = true;
                }
            }
        }
    }

    fn show_slice_position_section(&mut self, ui: &mut egui::Ui) {
        let (max_k, max_j, max_i) = if let Some(nf) = self.nifti_files.first() {
            let vol = &nf.volume;
            (
                vol.dims[2].saturating_sub(1),
                vol.dims[1].saturating_sub(1),
                vol.dims[0].saturating_sub(1),
            )
        } else {
            return;
        };

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
    }
}
