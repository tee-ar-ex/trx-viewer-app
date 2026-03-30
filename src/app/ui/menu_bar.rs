pub struct MenuAction {
    pub open_trx: bool,
    pub import_tractogram: bool,
    pub open_nifti: bool,
    pub open_gifti: bool,
}

pub fn show_menu_bar(ctx: &egui::Context) -> MenuAction {
    let mut action = MenuAction {
        open_trx: false,
        import_tractogram: false,
        open_nifti: false,
        open_gifti: false,
    };
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Open TRX...").clicked() {
                    action.open_trx = true;
                    ui.close();
                }
                if ui.button("Import Tractogram...").clicked() {
                    action.import_tractogram = true;
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
            });
        });
    });
    action
}
