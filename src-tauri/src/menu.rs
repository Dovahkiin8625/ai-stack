use tauri::menu::{Menu, MenuBuilder, MenuItemBuilder, Submenu, SubmenuBuilder};
use tauri::{AppHandle, Runtime};

pub const MENU_ID_TOGGLE_THEME: &str = "toggle_theme";
pub const MENU_ID_OPEN_FOLDER: &str = "open_folder";
pub const MENU_ID_NEW_NOTE: &str = "new_note";
pub const MENU_ID_ABOUT: &str = "about";
pub const MENU_ID_QUIT: &str = "quit";

pub fn build_menu<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    let open_folder = MenuItemBuilder::with_id(MENU_ID_OPEN_FOLDER, "打开文件夹…")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let new_note = MenuItemBuilder::with_id(MENU_ID_NEW_NOTE, "新建笔记")
        .accelerator("CmdOrCtrl+N")
        .build(app)?;
    let quit = MenuItemBuilder::with_id(MENU_ID_QUIT, "退出")
        .accelerator("CmdOrCtrl+Q")
        .build(app)?;

    let file: Submenu<R> = SubmenuBuilder::new(app, "文件")
        .item(&open_folder)
        .item(&new_note)
        .separator()
        .item(&quit)
        .build()?;

    let toggle_theme = MenuItemBuilder::with_id(MENU_ID_TOGGLE_THEME, "切换主题")
        .accelerator("CmdOrCtrl+T")
        .build(app)?;
    let view: Submenu<R> = SubmenuBuilder::new(app, "视图")
        .item(&toggle_theme)
        .build()?;

    let about = MenuItemBuilder::with_id(MENU_ID_ABOUT, "关于 AI Stack").build(app)?;
    let help: Submenu<R> = SubmenuBuilder::new(app, "帮助")
        .item(&about)
        .build()?;

    let menu = MenuBuilder::new(app)
        .item(&file)
        .item(&view)
        .item(&help)
        .build()?;
    Ok(menu)
}

/// 给前端抛事件用：列出菜单项 ID（仅用于文档/校验）
pub fn menu_item_ids() -> Vec<&'static str> {
    vec![
        MENU_ID_OPEN_FOLDER,
        MENU_ID_NEW_NOTE,
        MENU_ID_QUIT,
        MENU_ID_TOGGLE_THEME,
        MENU_ID_ABOUT,
    ]
}
