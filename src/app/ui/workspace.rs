use egui_tiles::{Behavior, Tree, UiResponse};

use crate::app::workflow::{self, WorkflowGraphViewer, WorkflowSelection, WorkspacePane};

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
                &mut self.workflow.document.workspace,
                Tree::empty("workflow_workspace"),
            );
            let mut behavior = WorkspaceBehavior { app: self, frame };
            tree.ui(&mut behavior, ui);
            self.workflow.document.workspace = tree;
        });
    }

    fn show_assets_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Assets");
        ui.separator();

        if self.workflow.document.assets.is_empty() {
            ui.small("Open files to populate the graph.");
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            for asset in &self.workflow.document.assets {
                match asset {
                    workflow::WorkflowAssetDocument::Streamlines { id, path, imported } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        let label = if *imported {
                            format!("Streamlines (imported)\n{}", path.display())
                        } else {
                            format!("Streamlines\n{}", path.display())
                        };
                        if ui.selectable_label(selected, label).clicked() {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Volume { id, path } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Volume\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Surface { id, path } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Surface\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                    workflow::WorkflowAssetDocument::Parcellation { id, path, .. } => {
                        let selected =
                            self.workflow.selection == Some(WorkflowSelection::Asset(*id));
                        if ui
                            .selectable_label(selected, format!("Parcellation\n{}", path.display()))
                            .clicked()
                        {
                            self.workflow.selection = Some(WorkflowSelection::Asset(*id));
                        }
                    }
                }
                ui.add_space(6.0);
            }
        });
    }

    fn show_graph_pane(&mut self, ui: &mut egui::Ui) {
        workflow::ensure_node_uuids(&mut self.workflow.document);
        ui.horizontal(|ui| {
            if ui.button("Run Expensive Nodes").clicked() {
                self.workflow.run_expensive_requested = true;
                ui.ctx().request_repaint();
            }
            if self.workflow.run_expensive_requested {
                ui.small("Will run on the next graph refresh.");
            }
        });
        ui.separator();
        let mut viewer = WorkflowGraphViewer {
            selected: &mut self.workflow.selection,
            focus_bounds: &mut self.workflow.graph_focus_request,
            viewport_rect: ui.max_rect(),
            node_state: &self.workflow.runtime.node_state,
        };
        egui_snarl::ui::SnarlWidget::new()
            .id(egui::Id::new("workflow_graph"))
            .show(&mut self.workflow.document.graph, &mut viewer, ui);

        let selected_nodes =
            egui_snarl::ui::get_selected_nodes(egui::Id::new("workflow_graph"), ui.ctx());
        if let Some(node_id) = selected_nodes.first().copied() {
            self.workflow.selection = Some(WorkflowSelection::Node(
                self.workflow.document.graph[node_id].uuid,
            ));
        }
    }

    fn show_inspector_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inspector");
        ui.separator();

        match self.workflow.selection {
            Some(WorkflowSelection::Asset(asset_id)) => self.show_asset_inspector(ui, asset_id),
            Some(WorkflowSelection::Node(node_uuid)) => self.show_node_inspector(ui, node_uuid),
            None => {
                ui.small("Select an asset or node.");
                if let Some(error) = &self.workflow.runtime.graph_error {
                    ui.separator();
                    ui.colored_label(egui::Color32::RED, error);
                }
            }
        }
    }

    fn show_asset_inspector(&mut self, ui: &mut egui::Ui, asset_id: usize) {
        if let Some(trx) = self
            .scene
            .trx_files
            .iter()
            .find(|asset| asset.id == asset_id)
        {
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
        if let Some(volume) = self
            .scene
            .nifti_files
            .iter()
            .find(|asset| asset.id == asset_id)
        {
            ui.strong(&volume.name);
            ui.label(format!(
                "Dims: {} x {} x {}",
                volume.volume.dims[0], volume.volume.dims[1], volume.volume.dims[2]
            ));
            return;
        }
        if let Some(surface) = self
            .scene
            .gifti_surfaces
            .iter_mut()
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
            .scene
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
            .workflow
            .document
            .graph
            .node_ids()
            .find(|(_, node)| node.uuid == node_uuid)
        else {
            ui.small("Selected node is no longer present.");
            return;
        };

        let mut save_now = false;
        let node = &mut self.workflow.document.graph[node_id];
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
                outline_color,
                outline_thickness,
                show_projection_map,
                map_opacity,
                map_threshold,
                gloss,
                projection_colormap,
                range_min,
                range_max,
            } => {
                ui.label("Surface");
                ui.color_edit_button_rgb(color);
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
                ui.separator();
                ui.label("Slice outline");
                ui.color_edit_button_rgb(outline_color);
                ui.add(egui::Slider::new(outline_thickness, 0.25..=8.0).text("Thickness"));
                ui.separator();
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
                min_component_volume_mm3,
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
                ui.add(
                    egui::DragValue::new(min_component_volume_mm3)
                        .speed(1.0)
                        .range(0.0..=1_000_000.0)
                        .prefix("Min component mm^3 "),
                );
                ui.add(egui::Slider::new(opacity, 0.0..=1.0).text("Opacity"));
            }
            workflow::WorkflowNodeKind::BundleSurfaceDisplay {
                color_mode,
                outline_thickness,
            } => {
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
                ui.separator();
                ui.label("Slice outline");
                ui.add(egui::Slider::new(outline_thickness, 0.25..=8.0).text("Thickness"));
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
                    .workflow
                    .runtime
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

        if let Some(state) = self.workflow.runtime.node_state.get(&node_uuid) {
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

        if let Some(message) = self.workflow.node_feedback.get(&node_uuid) {
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(96, 210, 128), message);
        }

        if save_now {
            self.save_streamline_node(node_uuid);
        }
    }

    fn show_preview_pane(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        ui.heading("Viewers Moved");
        ui.separator();
        ui.label("3D and 2D views now live in separate windows.");
        ui.horizontal(|ui| {
            if ui.button("Open 3D Window").clicked() {
                self.viewport.window_3d_open = true;
            }
            if ui.button("Open 2D Window").clicked() {
                self.viewport.view_2d.window_open = true;
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
