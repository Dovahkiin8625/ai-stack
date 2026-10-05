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
    /// Unix epoch seconds of last filesystem mtime at scan time.
    /// scanner 用 (size, mtime) 比对判断是否需要重写 DB 行；存进 DB 供下次启动复用。
    pub mtime: i64,
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
    /// 用户在 AI 问答流前输入的问题。AI 讲解场景永远为 None；笔记卡片只在 prompt 非空时
    /// 渲染"❓ 提问"块，让回看时知道这条答案是回答什么问题的。
    pub prompt: Option<String>,
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
                indexed_at TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                UNIQUE(category_path, rel_path),
                FOREIGN KEY (category_path) REFERENCES categories(path)
             );
             CREATE INDEX IF NOT EXISTS idx_resources_category ON resources(category_path);
             -- mtime 列在老库上不存在；新表直接建带 mtime 的版本，
             -- 老表用下面的 ALTER 升级。两者只走一条。
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
        // mtime 列在 v1 schema 里没有 —— 增量升级加在 resources 上。
        // 同 notes.source 的处理方式：pragma 探测缺失后 ALTER ADD COLUMN。
        let has_mtime: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name = 'mtime'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_mtime == 0 {
            tx.execute("ALTER TABLE resources ADD COLUMN mtime INTEGER NOT NULL DEFAULT 0", [])?;
        }
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
        // Phase 5 增量：notes 表新增 prompt 列（用户问 AI 的问题）。
        // 旧笔记 prompt 全为 NULL（向后兼容）；新增 prompt 列用于新增 prompt。
        let has_prompt: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('notes') WHERE name='prompt'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_prompt == 0 {
            tx.execute("ALTER TABLE notes ADD COLUMN prompt TEXT", [])?;
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
                                    size_bytes, mtime, indexed_at, page_count, word_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
             ON CONFLICT(category_path, rel_path) DO UPDATE SET
                type = excluded.type,
                title = excluded.title,
                size_bytes = excluded.size_bytes,
                mtime = excluded.mtime,
                indexed_at = excluded.indexed_at,
                page_count = excluded.page_count,
                word_count = excluded.word_count",
            params![
                r.category_path,
                r.rel_path,
                r.r#type,
                r.title,
                r.size_bytes,
                r.mtime,
                now,
                r.page_count,
                r.word_count,
            ],
        )?;
        // 之前这里再 query_row 一次 SELECT id —— scanner 端没用返回值，纯浪费一次往返。
        // 改用 last_insert_rowid：INSERT 路径拿到新 id；UPDATE 路径 last_insert_rowid 不变（仍是当前行 id）。
        // SQLite 在同一连接内 ON CONFLICT UPDATE 后 last_insert_rowid 会指向被改动的行，可信。
        let id = self.conn.last_insert_rowid();
        Ok(id)
    }

    /// 返回现有资源 (size_bytes, mtime_secs) 的全量映射。
    /// scanner 在 walk 之前调用，对每个文件做 (size, mtime) 比对：
    /// - 一致 → 跳过，不读不写
    /// - 不一致 / 不存在 → upsert
    pub fn existing_resources_meta(
        &self,
    ) -> Result<std::collections::HashMap<(String, String), (i64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT category_path, rel_path, size_bytes, mtime FROM resources",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    (row.get::<_, String>(0)?, row.get::<_, String>(1)?),
                    (row.get::<_, i64>(2)?, row.get::<_, i64>(3)?),
                ))
            })?
            .collect::<rusqlite::Result<std::collections::HashMap<_, _>>>()?;
        Ok(rows)
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
                    size_bytes, indexed_at, page_count, word_count
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
                    indexed_at: row.get(6)?,
                    page_count: row.get(7)?,
                    word_count: row.get(8)?,
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
                    prompt, source, created_at, updated_at
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
                    prompt: row.get(5)?,
                    source: row.get(6)?,
                    created_at: row.get(7)?,
                    updated_at: row.get(8)?,
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
        // 仅"用户问 AI" 场景使用（AI 讲解场景传 None）。
        // 存进 DB 后由 list_notes / get_note 回读，笔记卡片不再用。
        prompt: Option<&str>,
    ) -> Result<NoteRow> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO notes (resource_id, content, anchor_text, anchor_occurrence,
                                source, prompt, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
            params![
                resource_id,
                content,
                anchor_text,
                anchor_occurrence,
                source,
                prompt,
                now
            ],
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
                        prompt, source, created_at, updated_at
                 FROM notes WHERE id = ?1",
                [id],
                |r| {
                    Ok(NoteRow {
                        id: r.get(0)?,
                        resource_id: r.get(1)?,
                        content: r.get(2)?,
                        anchor_text: r.get(3)?,
                        anchor_occurrence: r.get(4)?,
                        prompt: r.get(5)?,
                        source: r.get(6)?,
                        created_at: r.get(7)?,
                        updated_at: r.get(8)?,
                    })
                },
            )
            .with_context(|| format!("note {id} not found"))?;
        Ok(row)
    }
}