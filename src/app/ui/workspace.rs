use egui_tiles::{Behavior, Tree, UiResponse};

use crate::app::callbacks::{self, BundleDrawInfo, StreamlineDrawInfo, VolumeDrawInfo};
use crate::app::workflow::{self, WorkflowGraphViewer, WorkflowSelection, WorkspacePane};
use crate::renderer::mesh_renderer::MeshDrawStyle;

impl super::super::TrxViewerApp {
    pub(in crate::app) fn show_workspace(
        &mut self,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) {
        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(message) = self.status_msg.clone() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::from_rgb(96, 210, 128), &message);
                        if ui.button("Clear").clicked() {
                            self.status_msg = None;
                        }
                    });
                });
                ui.add_space(8.0);
            }
            if let Some(message) = self.error_msg.clone() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.colored_label(egui::Color32::from_rgb(255, 110, 110), &message);
                        if ui.button("Dismiss").clicked() {
                            self.error_msg = None;
                        }
                    });
                });
                ui.add_space(8.0);
            }
            let mut tree = std::mem::replace(
                &mut self.workflow_document.workspace,
                Tree::empty("workflow_workspace"),
            );
            let mut behavior = WorkspaceBehavior { app: self, frame };
            tree.ui(&mut behavior, ui);
            self.workflow_document.workspace = tree;
        });
    }

    fn show_assets_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Assets");
        ui.separator();

        if self.workflow_document.assets.is_empty() {
            ui.small("Open files to populate the graph.");
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for asset in &self.workflow_document.assets {
                match asset {
                    workflow::WorkflowAssetDocument::Streamlines { id, path, imported } => {
                        let selected =
                            self.workflow_selection == Some(WorkflowSelection::Asset(*id));
                        let label = if *imported {
                            format!("Streamlines (imported)\n{}", path.display())
                        } else {
                            format!("Streamlines\n{}", path.display())
                        };
                        if ui.selectable_label(selected, label).clicked() {
                            self.workflow_selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Volume { id, path } => {
                        let selected =
                            self.workflow_selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Volume\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow_selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Surface { id, path } => {
                        let selected =
                            self.workflow_selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Surface\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow_selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Parcellation { id, path, .. } => {
                        let selected =
                            self.workflow_selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Parcellation\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow_selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                }
                ui.add_space(6.0);
            }
        });
    }

    fn show_graph_pane(&mut self, ui: &mut egui::Ui) {
        workflow::ensure_node_uuids(&mut self.workflow_document);
        ui.horizontal(|ui| {
            if ui.button("Run Expensive Nodes").clicked() {
                self.workflow_run_expensive_requested = true;
                ui.ctx().request_repaint();
            }
            if self.workflow_run_expensive_requested {
                ui.small("Will run on the next graph refresh.");
            }
        });
        ui.separator();
        let mut viewer = WorkflowGraphViewer {
            selected: &mut self.workflow_selection,
            focus_bounds: &mut self.workflow_graph_focus_request,
            viewport_rect: ui.max_rect(),
            node_state: &self.workflow_runtime.node_state,
        };
        egui_snarl::ui::SnarlWidget::new()
            .id(egui::Id::new("workflow_graph"))
            .show(&mut self.workflow_document.graph, &mut viewer, ui);

        let selected_nodes =
            egui_snarl::ui::get_selected_nodes(egui::Id::new("workflow_graph"), ui.ctx());
        if let Some(node_id) = selected_nodes.first().copied() {
            self.workflow_selection = Some(WorkflowSelection::Node(
                self.workflow_document.graph[node_id].uuid,
            ));
        }
    }

    fn show_inspector_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        ui.separator();

        match self.workflow_selection {
            Some(WorkflowSelection::Asset(asset_id)) => self.show_asset_inspector(ui, asset_id),
            Some(WorkflowSelection::Node(node_uuid)) => self.show_node_inspector(ui, node_uuid),
            None => {
                ui.small("Select an asset or node.");
                if let Some(error) = &self.workflow_runtime.graph_error {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, error);
                }
            }
        }
    }

    fn show_asset_inspector(&mut self, ui: &mut egui::Ui, asset_id: usize) {
        if let Some(trx) = self.trx_files.iter().find(|asset| asset.id == asset_id) {
            ui.strong(&trx.name);
            ui.label(trx.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "{} streamlines, {} vertices, {} groups",
                trx.data.nb_streamlines,
                trx.data.nb_vertices,
                trx.data.groups.len()
            ));
            return;
        }
        if let Some(volume) = self.nifti_files.iter().find(|asset| asset.id == asset_id) {
            ui.strong(&volume.name);
            ui.label(format!(
                "Dims: {} x {} x {}",
                volume.volume.dims[0], volume.volume.dims[1], volume.volume.dims[2]
            ));
            return;
        }
        if let Some(surface) = self
            .gifti_surfaces
            .iter()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&surface.name);
            ui.label(surface.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "{} vertices, {} triangles",
                surface.data.vertices.len(),
                surface.data.indices.len() / 3
            ));
            return;
        }
        if let Some(parcel) = self
            .parcellations
            .iter()
            .find(|asset| asset.asset.id == asset_id)
        {
            ui.strong(&parcel.asset.name);
            ui.label(parcel.asset.path.display().to_string());
            ui.separator();
            ui.label(format!(
                "Label volume: {} x {} x {}",
                parcel.asset.data.dims[0], parcel.asset.data.dims[1], parcel.asset.data.dims[2]
            ));
            ui.label(format!(
                "{} nonzero parcel labels",
                parcel
                    .asset
                    .data
                    .labels
                    .iter()
                    .copied()
                    .filter(|label| *label != 0)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
            ));
        }
    }

    fn show_node_inspector(&mut self, ui: &mut egui::Ui, node_uuid: workflow::WorkflowNodeUuid) {
        let Some((node_id, _)) = self
            .workflow_document
            .graph
            .node_ids()
            .find(|(_, node)| node.uuid == node_uuid)
        else {
            ui.small("Selected node is no longer present.");
            return;
        };

        let mut save_now = false;
        let node = &mut self.workflow_document.graph[node_id];
        ui.text_edit_singleline(&mut node.label);
        ui.separator();

        match &mut node.kind {
            workflow::WorkflowNodeKind::LimitStreamlines {
                limit,
                randomize,
                seed,
            } => {
                ui.add(egui::Slider::new(limit, 1..=1_000_000).text("Limit"));
                ui.checkbox(randomize, "Randomize before limiting");
                if *randomize {
                    ui.add(egui::DragValue::new(seed).speed(1.0).prefix("Seed "));
                }
            }
            workflow::WorkflowNodeKind::GroupSelect { groups_csv } => {
                ui.label("Comma-separated group names");
                ui.text_edit_multiline(groups_csv);
            }
            workflow::WorkflowNodeKind::RandomSubset { limit, seed } => {
                ui.add(egui::Slider::new(limit, 1..=1_000_000).text("Limit"));
                ui.add(egui::DragValue::new(seed).speed(1.0).prefix("Seed "));
            }
            workflow::WorkflowNodeKind::SphereQuery { center, radius_mm } => {
                ui.label("Center (RAS+ mm)");
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut center[0]).speed(0.5).prefix("X "));
                    ui.add(egui::DragValue::new(&mut center[1]).speed(0.5).prefix("Y "));
                    ui.add(egui::DragValue::new(&mut center[2]).speed(0.5).prefix("Z "));
                });
                ui.add(egui::DragValue::new(radius_mm).speed(0.5).prefix("Radius "));
            }
            workflow::WorkflowNodeKind::SurfaceDepthQuery { depth_mm }
            | workflow::WorkflowNodeKind::SurfaceProjectionDensity { depth_mm } => {
                ui.add(egui::DragValue::new(depth_mm).speed(0.25).prefix("Depth "));
            }
            workflow::WorkflowNodeKind::SurfaceProjectionMeanDps { depth_mm, field } => {
                ui.add(egui::DragValue::new(depth_mm).speed(0.25).prefix("Depth "));
                ui.text_edit_singleline(field);
            }
            workflow::WorkflowNodeKind::ParcelSelect { labels_csv } => {
                ui.label("Comma-separated label IDs");
                ui.small("Leave empty to use every nonzero parcel label.");
                ui.text_edit_multiline(labels_csv);
            }
            workflow::WorkflowNodeKind::ParcelEnd { endpoint_count } => {
                ui.add(egui::Slider::new(endpoint_count, 1..=2).text("Matching endpoints"));
            }
            workflow::WorkflowNodeKind::ColorByDPV { field }
            | workflow::WorkflowNodeKind::ColorByDPS { field } => {
                ui.text_edit_singleline(field);
            }
            workflow::WorkflowNodeKind::UniformColor { color } => {
                ui.color_edit_button_rgba_unmultiplied(color);
            }
            workflow::WorkflowNodeKind::StreamlineDisplay {
                enabled,
                render_style,
                tube_radius_mm,
                tube_sides,
                slab_half_width_mm,
            } => {
                ui.checkbox(enabled, "Visible");
                egui::ComboBox::from_id_salt(format!("render_style_{}", node_uuid.0))
                    .selected_text(format!("{render_style:?}"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            render_style,
                            crate::data::trx_data::RenderStyle::Flat,
                            "Flat",
                        );
                        ui.selectable_value(
                            render_style,
                            crate::data::trx_data::RenderStyle::Illuminated,
                            "Illuminated",
                        );
                        ui.selectable_value(
                            render_style,
                            crate::data::trx_data::RenderStyle::DepthCue,
                            "Depth Cue",
                        );
                        ui.selectable_value(
                            render_style,
                            crate::data::trx_data::RenderStyle::Tubes,
                            "Tubes",
                        );
                    });
                ui.add(
                    egui::DragValue::new(tube_radius_mm)
                        .speed(0.1)
                        .prefix("Tube radius "),
                );
                ui.add(
                    egui::DragValue::new(tube_sides)
                        .speed(1.0)
                        .prefix("Tube sides "),
                );
                ui.add(
                    egui::DragValue::new(slab_half_width_mm)
                        .speed(0.5)
                        .prefix("Slice slab "),
                );
            }
            workflow::WorkflowNodeKind::VolumeDisplay {
                colormap,
                opacity,
                window_center,
                window_width,
            } => {
                egui::ComboBox::from_id_salt(format!("volume_colormap_{}", node_uuid.0))
                    .selected_text(colormap.label())
                    .show_ui(ui, |ui| {
                        for value in crate::data::loaded_files::VolumeColormap::ALL {
                            ui.selectable_value(colormap, *value, value.label());
                        }
                    });
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                ui.add(egui::Slider::new(window_center, 0.0..=1.0).text("Window center"));
                ui.add(egui::Slider::new(window_width, 0.01..=2.0).text("Window width"));
            }
            workflow::WorkflowNodeKind::SurfaceDisplay {
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
                ui.color_edit_button_rgb(color);
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                ui.checkbox(show_projection_map, "Show surface map");
                ui.add(egui::Slider::new(map_opacity, 0.0..=1.0).text("Map opacity"));
                ui.add(egui::Slider::new(map_threshold, 0.0..=1.0).text("Map threshold"));
                ui.add(egui::Slider::new(gloss, 0.0..=1.0).text("Gloss"));
                egui::ComboBox::from_id_salt(format!("surface_colormap_{}", node_uuid.0))
                    .selected_text(format!("{projection_colormap:?}"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            projection_colormap,
                            crate::renderer::mesh_renderer::SurfaceColormap::BlueWhiteRed,
                            "Blue-White-Red",
                        );
                        ui.selectable_value(
                            projection_colormap,
                            crate::renderer::mesh_renderer::SurfaceColormap::Viridis,
                            "Viridis",
                        );
                        ui.selectable_value(
                            projection_colormap,
                            crate::renderer::mesh_renderer::SurfaceColormap::Inferno,
                            "Inferno",
                        );
                    });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(range_min).speed(0.1).prefix("Min "));
                    ui.add(egui::DragValue::new(range_max).speed(0.1).prefix("Max "));
                });
            }
            workflow::WorkflowNodeKind::BundleSurfaceBuild {
                per_group,
                voxel_size_mm,
                threshold,
                smooth_sigma,
                opacity,
            } => {
                ui.checkbox(per_group, "Per group");
                ui.add(
                    egui::DragValue::new(voxel_size_mm)
                        .speed(0.1)
                        .prefix("Voxel "),
                );
                ui.add(
                    egui::DragValue::new(threshold)
                        .speed(0.1)
                        .prefix("Threshold "),
                );
                ui.add(
                    egui::DragValue::new(smooth_sigma)
                        .speed(0.05)
                        .prefix("Smooth "),
                );
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            }
            workflow::WorkflowNodeKind::BundleSurfaceDisplay { color_mode } => {
                egui::ComboBox::from_id_salt(format!("bundle_surface_color_mode_{}", node_uuid.0))
                    .selected_text(color_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            color_mode,
                            workflow::BundleSurfaceColorMode::Solid,
                            workflow::BundleSurfaceColorMode::Solid.label(),
                        );
                        ui.selectable_value(
                            color_mode,
                            workflow::BundleSurfaceColorMode::BoundaryField,
                            workflow::BundleSurfaceColorMode::BoundaryField.label(),
                        );
                    });
            }
            workflow::WorkflowNodeKind::BoundaryFieldBuild {
                voxel_size_mm,
                sphere_lod,
                normalization,
            } => {
                ui.add(
                    egui::DragValue::new(voxel_size_mm)
                        .speed(0.1)
                        .range(0.5..=100.0)
                        .prefix("Voxel "),
                );
                ui.add(
                    egui::DragValue::new(sphere_lod)
                        .speed(1.0)
                        .range(4..=64)
                        .prefix("Sphere LOD "),
                );
                egui::ComboBox::from_id_salt(format!(
                    "boundary_field_normalization_{}",
                    node_uuid.0
                ))
                .selected_text(normalization.label())
                .show_ui(ui, |ui| {
                    for value in crate::data::orientation_field::BoundaryGlyphNormalization::ALL {
                        ui.selectable_value(normalization, value, value.label());
                    }
                });
            }
            workflow::WorkflowNodeKind::BoundaryGlyphDisplay {
                enabled,
                scale,
                density_3d_step,
                slice_density_step,
                color_mode,
                min_contacts,
            } => {
                ui.checkbox(enabled, "Visible");
                ui.add(egui::DragValue::new(scale).speed(0.1).prefix("Scale "));
                ui.add(
                    egui::DragValue::new(density_3d_step)
                        .speed(1.0)
                        .range(1..=64)
                        .prefix("3D step "),
                );
                ui.add(
                    egui::DragValue::new(slice_density_step)
                        .speed(1.0)
                        .range(1..=64)
                        .prefix("Slice step "),
                );
                ui.add(
                    egui::DragValue::new(min_contacts)
                        .speed(1.0)
                        .range(1..=1_000_000)
                        .prefix("Min contacts "),
                );
                egui::ComboBox::from_id_salt(format!("boundary_glyph_color_mode_{}", node_uuid.0))
                    .selected_text(color_mode.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            color_mode,
                            crate::data::orientation_field::BoundaryGlyphColorMode::DirectionRgb,
                            crate::data::orientation_field::BoundaryGlyphColorMode::DirectionRgb
                                .label(),
                        );
                        ui.selectable_value(
                            color_mode,
                            crate::data::orientation_field::BoundaryGlyphColorMode::Monochrome,
                            crate::data::orientation_field::BoundaryGlyphColorMode::Monochrome
                                .label(),
                        );
                    });
            }
            workflow::WorkflowNodeKind::ParcellationDisplay {
                labels_csv,
                opacity,
            } => {
                ui.label("Comma-separated label IDs");
                ui.small("Leave empty to use every nonzero parcel label.");
                ui.text_edit_multiline(labels_csv);
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            }
            workflow::WorkflowNodeKind::SaveStreamlines { output_path } => {
                ui.horizontal(|ui| {
                    ui.text_edit_singleline(output_path);
                    if ui.button("Browse...").clicked()
                        && let Some(path) = rfd::FileDialog::new()
                            .set_file_name("streamlines.trx")
                            .save_file()
                    {
                        *output_path = path.display().to_string();
                    }
                });
                let ready = self
                    .workflow_runtime
                    .save_streamline_targets
                    .contains_key(&node_uuid);
                if ui
                    .add_enabled(ready, egui::Button::new("Save Now"))
                    .clicked()
                {
                    save_now = true;
                }
                if !ready {
                    ui.small("Connect a streamline input to enable export.");
                }
            }
            _ => {
                ui.small("This node has no editable parameters yet.");
            }
        }

        if let Some(state) = self.workflow_runtime.node_state.get(&node_uuid) {
            ui.separator();
            ui.small(&state.summary);
            if let Some(execution) = &state.execution {
                ui.label(format!("Status: {}", execution.label()));
            }
            if let Some(fingerprint) = state.fingerprint {
                ui.small(format!("Fingerprint: {fingerprint:016x}"));
            }
            if let Some(result_summary) = &state.last_result_summary {
                ui.small(result_summary);
            }
            if let Some(error) = &state.error {
                ui.colored_label(egui::Color32::RED, error);
            }
        }

        if let Some(message) = self.workflow_node_feedback.get(&node_uuid) {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(96, 210, 128), message);
        }

        if save_now {
            self.save_streamline_node(node_uuid);
        }
    }

    fn show_preview_pane(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.trx_files.is_empty()
            && self.nifti_files.is_empty()
            && self.gifti_surfaces.is_empty()
            && self.parcellations.is_empty()
        {
            ui.centered_and_justified(|ui| {
                ui.label("Open files or load a project to begin.");
            });
            return;
        }

        let available = ui.available_size();
        let any_slice_visible = self.slice_visible.iter().any(|&v| v);
        let top_height = if any_slice_visible {
            (available.y * 0.6).max(100.0)
        } else {
            available.y
        };
        let bottom_height = if any_slice_visible {
            (available.y - top_height - ui.spacing().item_spacing.y).max(50.0)
        } else {
            0.0
        };

        let (rect_3d, response_3d) = ui.allocate_exact_size(
            egui::vec2(available.x, top_height),
            egui::Sense::click_and_drag(),
        );

        if response_3d.dragged_by(egui::PointerButton::Primary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera_3d.handle_drag(delta.x, delta.y);
        }
        if response_3d.dragged_by(egui::PointerButton::Middle) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera_3d.pan_screen(delta.x, delta.y);
        }
        if response_3d.dragged_by(egui::PointerButton::Secondary) {
            let delta = ui.input(|i| i.pointer.delta());
            self.camera_3d.handle_zoom_drag(delta.x, delta.y);
        }
        if response_3d.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.camera_3d.handle_scroll(scroll * 0.01);
            }
        }

        let aspect_3d = rect_3d.width() / rect_3d.height().max(1.0);
        let vp_3d = self.camera_3d.view_projection(aspect_3d);

        let surface_draws: Vec<(usize, MeshDrawStyle)> = self
            .workflow_runtime
            .scene_plan
            .surface_draws
            .iter()
            .map(|draw| {
                (
                    draw.gpu_index,
                    MeshDrawStyle {
                        color: [draw.color[0], draw.color[1], draw.color[2], draw.opacity],
                        scalar_min: draw.range_min,
                        scalar_max: draw.range_max,
                        scalar_enabled: draw.show_projection_map,
                        colormap: draw.projection_colormap,
                        gloss: draw.gloss,
                        map_opacity: draw.map_opacity,
                        map_threshold: draw.map_threshold,
                    },
                )
            })
            .collect();

        let volume_draws: Vec<VolumeDrawInfo> = self
            .workflow_runtime
            .scene_plan
            .volume_draws
            .iter()
            .map(|draw| VolumeDrawInfo {
                file_id: draw.source_id,
                window_center: draw.window_center,
                window_width: draw.window_width,
                colormap: draw.colormap.as_u32(),
                opacity: draw.opacity,
            })
            .collect();

        let streamline_draws: Vec<StreamlineDrawInfo> = self
            .workflow_runtime
            .scene_plan
            .streamline_draws
            .iter()
            .map(|draw| StreamlineDrawInfo {
                file_id: draw.draw_id,
                visible: draw.visible,
                render_style: draw.render_style,
                tube_radius: draw.tube_radius_mm,
                slab_half_width: draw.slab_half_width_mm,
            })
            .collect();

        let bundle_draws: Vec<BundleDrawInfo> = self
            .workflow_runtime
            .scene_plan
            .bundle_draws
            .iter()
            .map(|draw| BundleDrawInfo {
                file_id: draw.draw_id,
                opacity: draw.opacity,
            })
            .collect();

        let any_visible_streamlines = streamline_draws.iter().any(|draw| draw.visible);
        let glyph_draw = self
            .workflow_runtime
            .scene_plan
            .boundary_glyph_draws
            .iter()
            .find(|draw| draw.visible);
        let glyph_visible = glyph_draw.is_some() && self.boundary_field.is_some();
        let glyph_color_mode = glyph_draw
            .map(|draw| draw.color_mode)
            .unwrap_or(crate::data::orientation_field::BoundaryGlyphColorMode::DirectionRgb);
        let glyph_density_3d_step = glyph_draw
            .map(|draw| draw.density_3d_step as u32)
            .unwrap_or(1);
        let glyph_slice_density_step = glyph_draw
            .map(|draw| draw.slice_density_step as u32)
            .unwrap_or(1);

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect_3d,
            callbacks::Scene3DCallback {
                view_proj: vp_3d,
                camera_pos: self.camera_3d.eye(),
                camera_dir: self.camera_3d.view_direction(),
                streamline_draws: streamline_draws.clone(),
                show_streamlines: any_visible_streamlines,
                volume_draws,
                slice_visible: self.slice_visible,
                surface_draws,
                bundle_draws,
                show_boundary_glyphs: glyph_visible,
                boundary_glyph_color_mode: glyph_color_mode,
                boundary_glyph_draw_step: glyph_density_3d_step,
                scene_lighting: self.scene_lighting,
            },
        ));

        self.draw_3d_axes(ui, rect_3d, vp_3d);

        let visible_slice_indices: Vec<usize> = self
            .slice_visible
            .iter()
            .enumerate()
            .filter_map(|(i, visible)| visible.then_some(i))
            .collect();
        if visible_slice_indices.is_empty() {
            return;
        }

        let count = visible_slice_indices.len() as f32;
        let spacing = ui.spacing().item_spacing.x * (count - 1.0).max(0.0);
        let slice_width = ((available.x - spacing) / count).max(10.0);
        let slice_height = (bottom_height - ui.spacing().item_spacing.y - 18.0).max(10.0);

        ui.horizontal(|ui| {
            let axis_names = ["Axial", "Coronal", "Sagittal"];
            let axis_labels = ["Z", "Y", "X"];
            for &i in &visible_slice_indices {
                ui.vertical(|ui| {
                    let pos_mm = self.slice_world_position(i);
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.slice_visible[i], "");
                        ui.label(format!(
                            "{} ({} = {:.1} mm)",
                            axis_names[i], axis_labels[i], pos_mm
                        ));
                    });

                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(slice_width, slice_height),
                        egui::Sense::click_and_drag(),
                    );

                    if response.hovered() {
                        let scroll = ui.input(|inp| inp.smooth_scroll_delta.y);
                        if scroll.abs() > 0.5 {
                            let delta = if scroll > 0.0 { 1isize } else { -1 };
                            self.step_slice(i, delta);
                        }
                    }

                    let aspect = rect.width() / rect.height().max(1.0);
                    let slice_pos = self.slice_world_position(i);
                    let vp_slice = self.slice_cameras[i].view_projection(aspect, slice_pos);
                    let glyph_slab_half_width = self
                        .boundary_field
                        .as_ref()
                        .map(|field| 0.5 * field.grid.voxel_size_mm)
                        .unwrap_or(0.0);

                    let slab_axis = match i {
                        0 => 2u32,
                        1 => 1u32,
                        _ => 0u32,
                    };

                    let slice_volume_draws: Vec<VolumeDrawInfo> = self
                        .workflow_runtime
                        .scene_plan
                        .volume_draws
                        .iter()
                        .map(|draw| VolumeDrawInfo {
                            file_id: draw.source_id,
                            window_center: draw.window_center,
                            window_width: draw.window_width,
                            colormap: draw.colormap.as_u32(),
                            opacity: draw.opacity,
                        })
                        .collect();

                    let slice_streamline_draws: Vec<StreamlineDrawInfo> = self
                        .workflow_runtime
                        .scene_plan
                        .streamline_draws
                        .iter()
                        .map(|draw| StreamlineDrawInfo {
                            file_id: draw.draw_id,
                            visible: draw.visible,
                            render_style: draw.render_style,
                            tube_radius: draw.tube_radius_mm,
                            slab_half_width: draw.slab_half_width_mm,
                        })
                        .collect();
                    let slice_show_streamlines =
                        slice_streamline_draws.iter().any(|draw| draw.visible);

                    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
                        rect,
                        callbacks::SliceViewCallback {
                            view_proj: vp_slice,
                            quad_index: i,
                            bind_group_index: i + 1,
                            volume_draws: slice_volume_draws,
                            streamline_draws: slice_streamline_draws,
                            show_streamlines: slice_show_streamlines,
                            slab_axis,
                            slab_min: slice_pos - glyph_slab_half_width,
                            slab_max: slice_pos + glyph_slab_half_width,
                            show_boundary_glyphs: glyph_visible,
                            boundary_glyph_color_mode: glyph_color_mode,
                            boundary_glyph_draw_step: glyph_slice_density_step,
                            scene_lighting: self.scene_lighting,
                        },
                    ));

                    self.draw_crosshairs(ui, rect, i, vp_slice);
                    self.draw_orientation_labels(ui, rect, i, vp_slice);
                    self.draw_mesh_intersections(ui, rect, i, vp_slice, slice_pos);
                    self.draw_bundle_mesh_intersections(ui, rect, i, vp_slice, slice_pos);
                    self.draw_parcellation_intersections(ui, rect, i, vp_slice, slice_pos);
                });
            }
        });
    }
}

struct WorkspaceBehavior<'a> {
    app: &'a mut super::super::TrxViewerApp,
    frame: &'a mut eframe::Frame,
}

impl Behavior<WorkspacePane> for WorkspaceBehavior<'_> {
    fn tab_title_for_pane(&mut self, pane: &WorkspacePane) -> egui::WidgetText {
        match pane {
            WorkspacePane::Assets => "Assets".into(),
            WorkspacePane::Preview => "Preview".into(),
            WorkspacePane::Graph => "Workflow".into(),
            WorkspacePane::Inspector => "Inspector".into(),
        }
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut WorkspacePane,
    ) -> UiResponse {
        match pane {
            WorkspacePane::Assets => self.app.show_assets_pane(ui),
            WorkspacePane::Preview => self.app.show_preview_pane(ui, self.frame),
            WorkspacePane::Graph => self.app.show_graph_pane(ui),
            WorkspacePane::Inspector => self.app.show_inspector_pane(ui),
        }
        UiResponse::None
    }
}
