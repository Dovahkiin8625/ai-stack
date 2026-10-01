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

#[derive(Debug, Clone)]
pub struct NoteRow {
    pub id: i64,
    pub resource_id: i64,
    pub content: String,
    pub anchor_text: Option<String>,
    pub anchor_occurrence: i64,
    pub source: String,
    pub created_at: String,
    pub updated_at: String,
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
                UNIQUE(category_path, rel_path),
                FOREIGN KEY (category_path) REFERENCES categories(path)
             );
             CREATE INDEX IF NOT EXISTS idx_resources_category ON resources(category_path);
             CREATE INDEX IF NOT EXISTS idx_resources_hash ON resources(hash);
             CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                anchor_text TEXT,
                anchor_occurrence INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE
             );
             CREATE INDEX IF NOT EXISTS idx_notes_resource ON notes(resource_id);",
        )?;
        // Phase 5 升级：旧版 notes 表没有 source 列。用 pragma 探测后按需 ALTER，
        // 避免 SQLite 不支持 ADD COLUMN 的 IF NOT EXISTS 时报错。
        let has_source: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('notes') WHERE name = 'source'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_source == 0 {
            tx.execute(
                "ALTER TABLE notes ADD COLUMN source TEXT NOT NULL DEFAULT 'user'",
                [],
            )?;
        }
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

    pub fn list_notes(&self, resource_id: i64) -> Result<Vec<NoteRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, resource_id, content, anchor_text, anchor_occurrence,
                    source, created_at, updated_at
             FROM notes WHERE resource_id = ?1
             ORDER BY id",
        )?;
        let rows = stmt
            .query_map([resource_id], |row| {
                Ok(NoteRow {
                    id: row.get(0)?,
                    resource_id: row.get(1)?,
                    content: row.get(2)?,
                    anchor_text: row.get(3)?,
                    anchor_occurrence: row.get(4)?,
                    source: row.get(5)?,
                    created_at: row.get(6)?,
                    updated_at: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    pub fn insert_note(
        &self,
        resource_id: i64,
        content: &str,
        anchor_text: Option<&str>,
        anchor_occurrence: i64,
        source: &str,
    ) -> Result<NoteRow> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO notes (resource_id, content, anchor_text, anchor_occurrence,
                                source, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![resource_id, content, anchor_text, anchor_occurrence, source, now],
        )?;
        let id = self.conn.last_insert_rowid();
        self.get_note(id)
    }

    pub fn update_note(
        &self,
        id: i64,
        content: &str,
        anchor_text: Option<&str>,
        anchor_occurrence: i64,
        source: Option<&str>,
    ) -> Result<NoteRow> {
        // source 字段：只有 Some(_) 时才写入，避免每次普通编辑都重写 source；
        // AI 笔记被用户编辑后降级为 'user'，是这里唯一会传 Some 的场景。
        let now = chrono::Utc::now().to_rfc3339();
        let changed = match source {
            Some(src) => self.conn.execute(
                "UPDATE notes
                 SET content = ?1, anchor_text = ?2, anchor_occurrence = ?3,
                     source = ?4, updated_at = ?5
                 WHERE id = ?6",
                params![content, anchor_text, anchor_occurrence, src, now, id],
            )?,
            None => self.conn.execute(
                "UPDATE notes
                 SET content = ?1, anchor_text = ?2, anchor_occurrence = ?3, updated_at = ?4
                 WHERE id = ?5",
                params![content, anchor_text, anchor_occurrence, now, id],
            )?,
        };
        if changed == 0 {
            anyhow::bail!("note {id} not found");
        }
        self.get_note(id)
    }

    pub fn delete_note(&self, id: i64) -> Result<()> {
        let changed = self.conn.execute("DELETE FROM notes WHERE id = ?1", [id])?;
        if changed == 0 {
            anyhow::bail!("note {id} not found");
        }
        Ok(())
    }

    fn get_note(&self, id: i64) -> Result<NoteRow> {
        let row = self
            .conn
            .query_row(
                "SELECT id, resource_id, content, anchor_text, anchor_occurrence,
                        source, created_at, updated_at
                 FROM notes WHERE id = ?1",
                [id],
                |r| {
                    Ok(NoteRow {
                        id: r.get(0)?,
                        resource_id: r.get(1)?,
                        content: r.get(2)?,
                        anchor_text: r.get(3)?,
                        anchor_occurrence: r.get(4)?,
                        source: r.get(5)?,
                        created_at: r.get(6)?,
                        updated_at: r.get(7)?,
                    })
                },
            )
            .with_context(|| format!("note {id} not found"))?;
        Ok(row)
    }
}