//! 数据库：连接、迁移、CRUD
use anyhow::{Context, Result};
use rusqlite::params;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct CategoryRow {
    pub path: String,
    pub parent_path: Option<String>,
    pub title: String,
    pub sort_order: i64,
}

#[derive(Debug, Clone)]
pub struct ResourceRow {
    pub id: i64,
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub hash: String,
    pub indexed_at: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct ResourceInput {
    pub category_path: String,
    pub rel_path: String,
    pub r#type: String,
    pub title: String,
    pub size_bytes: i64,
    pub hash: String,
    pub page_count: Option<i64>,
    pub word_count: Option<i64>,
}

pub struct Db {
    pub conn: rusqlite::Connection,
}

impl Db {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create db dir {}", parent.display()))?;
        }
        let conn = rusqlite::Connection::open(path)
            .with_context(|| format!("open db {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    pub fn migrate(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS categories (
                path TEXT PRIMARY KEY,
                parent_path TEXT,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                FOREIGN KEY (parent_path) REFERENCES categories(path)
             );
             CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_path);
             CREATE TABLE IF NOT EXISTS resources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category_path TEXT NOT NULL,
                rel_path TEXT NOT NULL,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                hash TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                UNIQUE(category_path, rel_path)
             );
             CREATE INDEX IF NOT EXISTS idx_resources_category ON resources(category_path);
             CREATE INDEX IF NOT EXISTS idx_resources_hash ON resources(hash);",
        )?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?1)",
            params![now],
        )?;
        tx.commit()?;
        Ok(())
    }

    pub fn upsert_category(
        &self,
        path: &str,
        parent_path: Option<&str>,
        title: &str,
        sort_order: i64,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO categories (path, parent_path, title, sort_order)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET
                parent_path = excluded.parent_path,
                title = excluded.title,
                sort_order = excluded.sort_order",
            params![path, parent_path, title, sort_order],
        )?;
        Ok(())
    }

    pub fn upsert_resource(&self, r: ResourceInput) -> Result<i64> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO resources (category_path, rel_path, type, title,
                                    size_bytes, hash, indexed_at, page_count, word_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(category_path, rel_path) DO UPDATE SET
                type = excluded.type,
                title = excluded.title,
                size_bytes = excluded.size_bytes,
                hash = excluded.hash,
                indexed_at = excluded.indexed_at,
                page_count = excluded.page_count,
                word_count = excluded.word_count",
            params![
                r.category_path,
                r.rel_path,
                r.r#type,
                r.title,
                r.size_bytes,
                r.hash,
                now,
                r.page_count,
                r.word_count,
            ],
        )?;
        let id: i64 = self.conn.query_row(
            "SELECT id FROM resources WHERE category_path = ?1 AND rel_path = ?2",
            params![r.category_path, r.rel_path],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    pub fn list_categories(&self) -> Result<Vec<CategoryRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT path, parent_path, title, sort_order
             FROM categories ORDER BY sort_order, path",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(CategoryRow {
                    path: row.get(0)?,
                    parent_path: row.get(1)?,
                    title: row.get(2)?,
                    sort_order: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn list_resources(&self, category_path: &str) -> Result<Vec<ResourceRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, category_path, rel_path, type, title,
                    size_bytes, hash, indexed_at, page_count, word_count
             FROM resources WHERE category_path = ?1
             ORDER BY rel_path",
        )?;
        let rows = stmt
            .query_map([category_path], |row| {
                Ok(ResourceRow {
                    id: row.get(0)?,
                    category_path: row.get(1)?,
                    rel_path: row.get(2)?,
                    r#type: row.get(3)?,
                    title: row.get(4)?,
                    size_bytes: row.get(5)?,
                    hash: row.get(6)?,
                    indexed_at: row.get(7)?,
                    page_count: row.get(8)?,
                    word_count: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn delete_missing_resources(
        &self,
        category_path: &str,
        keep: &[String],
    ) -> Result<usize> {
        // 取出该分类下所有 rel_path，差集删除
        let mut stmt = self.conn.prepare(
            "SELECT rel_path FROM resources WHERE category_path = ?1",
        )?;
        let existing: Vec<String> = stmt
            .query_map([category_path], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let to_delete: Vec<String> = existing
            .into_iter()
            .filter(|p| !keep.contains(p))
            .collect();
        let mut removed = 0;
        for p in to_delete {
            removed += self.conn.execute(
                "DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2",
                params![category_path, p],
            )?;
        }
        Ok(removed)
    }
}