pub struct MenuAction {
    pub new_workflow_project: bool,
    pub open_workflow_project: bool,
    pub save_workflow_project: bool,
    pub save_workflow_project_as: bool,
    pub open_trx: bool,
    pub import_streamlines: bool,
    pub open_nifti: bool,
    pub open_gifti: bool,
    pub open_parcellation: bool,
    pub open_3d_window: bool,
    pub open_2d_window: bool,
    pub export_3d_view: bool,
    pub export_2d_view: bool,
}

pub fn show_menu_bar(ctx: &egui::Context) -> MenuAction {
    let mut action = MenuAction {
        new_workflow_project: false,
        open_workflow_project: false,
        save_workflow_project: false,
        save_workflow_project_as: false,
        open_trx: false,
        import_streamlines: false,
        open_nifti: false,
        open_gifti: false,
        open_parcellation: false,
        open_3d_window: false,
        open_2d_window: false,
        export_3d_view: false,
        export_2d_view: false,
    };
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Workflow Project").clicked() {
                    action.new_workflow_project = true;
                    ui.close();
                }
                if ui.button("Open Workflow Project...").clicked() {
                    action.open_workflow_project = true;
                    ui.close();
                }
                if ui.button("Save Workflow Project").clicked() {
                    action.save_workflow_project = true;
                    ui.close();
                }
                if ui.button("Save Workflow Project As...").clicked() {
                    action.save_workflow_project_as = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Open TRX...").clicked() {
                    action.open_trx = true;
                    ui.close();
                }
                if ui.button("Import Streamlines...").clicked() {
                    action.import_streamlines = true;
                    ui.close();
                }
                if ui.button("Open NIfTI...").clicked() {
                    action.open_nifti = true;
                    ui.close();
                }
                if ui.button("Open GIFTI Surface...").clicked() {
                    action.open_gifti = true;
                    ui.close();
                }
                if ui.button("Open Parcellation...").clicked() {
                    action.open_parcellation = true;
                    ui.close();
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Open 3D Window").clicked() {
                    action.open_3d_window = true;
                    ui.close();
                }
                if ui.button("Open 2D Window").clicked() {
                    action.open_2d_window = true;
                    ui.close();
                }
                ui.separator();
                if ui.button("Export 3D View...").clicked() {
                    action.export_3d_view = true;
                    ui.close();
                }
                if ui.button("Export 2D View...").clicked() {
                    action.export_2d_view = true;
                    ui.close();
                }
            });
        });
    });
    action
}
