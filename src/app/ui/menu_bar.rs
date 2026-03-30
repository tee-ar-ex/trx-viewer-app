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
        });
    });
    action
}
