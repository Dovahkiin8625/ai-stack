mod menu;
pub mod db;
pub mod scanner;
pub mod reader;
pub mod commands;
pub mod readers;

use tauri::{Emitter, Manager};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .setup(|app| {
            let menu = menu::build_menu(app.handle())?;
            app.set_menu(menu)?;
            app.on_menu_event(|app, event| {
                let _ = app.emit("menu", event.id().0.as_str());
            });
            // AppState 只持有不可变路径；不可在 commands::build_state 之后再修改
            app.manage(commands::build_state(app.handle()));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::scan_library,
            commands::list_categories,
            commands::list_resources,
            commands::read_resource,
            commands::list_notes,
            commands::create_note,
            commands::update_note,
            commands::delete_note,
            commands::translate_text,
            commands::start_ai_annotate,
        ])
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}