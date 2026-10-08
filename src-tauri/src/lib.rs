pub mod sync;
pub mod db;
pub mod scanner;
pub mod reader;
pub mod commands;
pub mod readers;
pub mod platform;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .setup(|app| {
            // AppState 只持有不可变路径；不可在 commands::build_state 之后再修改
            let state = commands::build_state(app.handle());
            let root = state.knowledge_root.clone();

            // 桌面端老版本升级迁移：bundled.resources 里的 knowledge/ 搬到 app_data。
            // 解析失败（资源不存在 / 新装用户没有 bundled）直接跳过 —— migrate_legacy_bundled
            // 内部已用 src.is_dir() 兜底，这里多一层防止 resolve 出错时 panic 阻断启动。
            // 失败也只记日志：哪怕首次迁移中断，用户下次启动会继续搬（迁移本身可恢复）。
            if let Ok(bundled) = app
                .path()
                .resolve("knowledge", tauri::path::BaseDirectory::Resource)
            {
                if let Err(e) = platform::migrate_legacy_bundled(&bundled, &root) {
                    eprintln!("[migrate] legacy bundled -> app_data failed: {e:#}");
                }
            }

            // 种子库只在 Android 有意义；失败不阻断启动（远端同步仍可补齐）
            if let Err(e) = platform::materialize_seed(app.handle(), &root) {
                eprintln!("[seed] materialize failed: {e:#}");
            }
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::scan_library,
            commands::list_categories,
            commands::list_resources,
            commands::read_resource,
            commands::read_resource_bytes,
            commands::write_resource,
            commands::read_subcategory_index,
            commands::set_sync_base_url,
            commands::sync_manifest,
            commands::sync_status,
            commands::download_resource,
            commands::download_all,
            commands::list_notes,
            commands::create_note,
            commands::update_note,
            commands::delete_note,
            commands::translate_text,
            commands::start_ai_annotate,
            commands::start_ai_qa,
        ])
        .run(tauri::generate_context!())
        .expect("error while running ai-stack application");
}