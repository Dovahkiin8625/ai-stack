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
    let builder = tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_sql::Builder::default().build());
    // window-state 桌面独占：Android/iOS 上单窗口模型没有窗口位置可保存
    #[cfg(desktop)]
    let builder = builder.plugin(tauri_plugin_window_state::Builder::default().build());
    builder
        .setup(|app| {
            // AppState 只持有不可变路径；不可在 commands::build_state 之后再修改
            let state = commands::build_state(app.handle());
            let root = state.knowledge_root.clone();

            // 桌面端老版本升级迁移：bundled.resources 里的 knowledge/ 搬到 app_data。
            // 标记写在 app_data/.legacy-migrated（**knowledge/ 外**，否则 scanner 会把
            // 它当成文章/分类）。标记存在 → migrate 内部短路，绝不重走 src。
            // 这是为了不静默覆盖用户在 dest 里的本地编辑（write_resource 改的就是这里）。
            if let Ok(bundled) = app
                .path()
                .resolve("knowledge", tauri::path::BaseDirectory::Resource)
            {
                if let Some(parent) = root.parent() {
                    let marker = parent.join(".legacy-migrated");
                    if let Err(e) = platform::migrate_legacy_bundled(&bundled, &root, &marker) {
                        eprintln!("[migrate] legacy bundled -> app_data failed: {e:#}");
                    }
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