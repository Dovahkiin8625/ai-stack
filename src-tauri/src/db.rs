//! 数据库：连接、迁移、CRUD
use anyhow::Result;

pub struct Db {
    pub conn: rusqlite::Connection,
}

impl Db {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = rusqlite::Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    pub fn migrate(&mut self) -> Result<()> {
        // 任务 2 实现
        Ok(())
    }
}