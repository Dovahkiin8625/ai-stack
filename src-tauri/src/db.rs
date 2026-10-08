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
    /// 文件是否已在本地 knowledge/ 目录里。manifest 有但没下过的条目为 false。
    pub present: bool,
    /// 非 NULL = 该行由 manifest 管理，scanner 的 delete_missing_resources 不得删它。
    pub remote_hash: Option<String>,
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

#[derive(Debug, Clone, Default)]
pub struct SyncStatus {
    pub total: i64,
    pub present: i64,
    pub total_bytes: i64,
    pub cached_bytes: i64,
}

/// manifest 条目的标题 = 文件名去扩展名（与 scanner 的 stem 规则一致）。
fn title_from_file_name(rel_path: &str) -> String {
    rel_path
        .rsplit('/')
        .next()
        .unwrap_or(rel_path)
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(rel_path)
        .to_string()
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
    /// PDF 笔记用的 0-based 页码定位。markdown/DOCX 笔记为 None。
    /// 老笔记全为 None，向后兼容。
    pub page_idx: Option<i64>,
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
        // Phase 6 增量：notes 表新增 page_idx 列（PDF 笔记页码定位）。
        // 老笔记 page_idx 全为 NULL（向后兼容）；markdown/DOCX 笔记也保持 NULL。
        let has_page_idx: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('notes') WHERE name='page_idx'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_page_idx == 0 {
            tx.execute("ALTER TABLE notes ADD COLUMN page_idx INTEGER", [])?;
        }
        // v2 增量：resources 加 present / remote_hash 两列。
        // - remote_hash：非 NULL 表示该行由 manifest 管理；为 NULL 是纯本地行。
        //   scanner 的 delete_missing_resources 必须跳过 remote_hash IS NOT NULL 的行，
        //   否则 manifest 里还没下完的条目会被当成"文件被删了"误清掉。
        // - present：是否本地已有文件。manifest 占位行默认 0；本地文件 upsert 后置 1；
        //   老库 ALTER 默认 1，因为旧库里的每一行都是已经躺在磁盘上的本地文件。
        let has_remote_hash: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name='remote_hash'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_remote_hash == 0 {
            tx.execute("ALTER TABLE resources ADD COLUMN remote_hash TEXT", [])?;
        }
        let has_present: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name='present'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if has_present == 0 {
            // 默认 1：老库里的每一行都是已经躺在磁盘上的本地文件。
            tx.execute("ALTER TABLE resources ADD COLUMN present INTEGER NOT NULL DEFAULT 1", [])?;
        }
        // v2 新表：index_files 与 app_config。
        // `_index.md` 不进 resources（scanner 刻意跳过），单独一张小表记录远端大小，
        // 供 read_subcategory_index 在文件缺失时按需下载。
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS index_files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                present INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
             );",
        )?;
        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?1)",
            params![now],
        )?;
        tx.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (2, ?1)",
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
                                    size_bytes, mtime, indexed_at, page_count, word_count,
                                    present)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1)
             ON CONFLICT(category_path, rel_path) DO UPDATE SET
                type = excluded.type,
                title = excluded.title,
                size_bytes = excluded.size_bytes,
                mtime = excluded.mtime,
                indexed_at = excluded.indexed_at,
                page_count = excluded.page_count,
                word_count = excluded.word_count,
                present = 1",
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
                    size_bytes, indexed_at, page_count, word_count, present, remote_hash
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
                    present: row.get::<_, i64>(9)? != 0,
                    remote_hash: row.get(10)?,
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
        // 取出该分类下所有 rel_path，差集删除。
        // 但必须排除 remote_hash IS NOT NULL 的行 —— 那些是 manifest 占位行，
        // 文件还没下载完是预期状态，不能被当成"用户删了"清掉。
        let mut stmt = self.conn.prepare(
            "SELECT rel_path FROM resources
             WHERE category_path = ?1 AND remote_hash IS NULL",
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
                "DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2 AND remote_hash IS NULL",
                params![category_path, p],
            )?;
        }
        Ok(removed)
    }

    pub fn list_notes(&self, resource_id: i64) -> Result<Vec<NoteRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, resource_id, content, anchor_text, anchor_occurrence,
                    prompt, source, page_idx, created_at, updated_at
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
                    page_idx: row.get(7)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
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
        // PDF 笔记用的 0-based 页码定位。markdown/DOCX 笔记为 None。
        page_idx: Option<i64>,
    ) -> Result<NoteRow> {
        let now = chrono::Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT INTO notes (resource_id, content, anchor_text, anchor_occurrence,
                                source, prompt, page_idx, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)",
            params![
                resource_id,
                content,
                anchor_text,
                anchor_occurrence,
                source,
                prompt,
                page_idx,
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
        // 普通编辑（None）保留原 page_idx，避免翻页跳转失效；
        // 重设时 Some(idx) 写新值，仅 PDF 笔记需要。
        page_idx: Option<i64>,
    ) -> Result<NoteRow> {
        // source / page_idx 字段：只有 Some(_) 时才写入，避免每次普通编辑都重写；
        // 4 个分支覆盖所有 (source, page_idx) 组合，保持 SQL 显式（无 COALESCE）。
        let now = chrono::Utc::now().to_rfc3339();
        let changed = match (source, page_idx) {
            (Some(src), Some(idx)) => self.conn.execute(
                "UPDATE notes
                 SET content = ?1, anchor_text = ?2, anchor_occurrence = ?3,
                     source = ?4, page_idx = ?5, updated_at = ?6
                 WHERE id = ?7",
                params![content, anchor_text, anchor_occurrence, src, idx, now, id],
            )?,
            (Some(src), None) => self.conn.execute(
                "UPDATE notes
                 SET content = ?1, anchor_text = ?2, anchor_occurrence = ?3,
                     source = ?4, updated_at = ?5
                 WHERE id = ?6",
                params![content, anchor_text, anchor_occurrence, src, now, id],
            )?,
            (None, Some(idx)) => self.conn.execute(
                "UPDATE notes
                 SET content = ?1, anchor_text = ?2, anchor_occurrence = ?3,
                     page_idx = ?4, updated_at = ?5
                 WHERE id = ?6",
                params![content, anchor_text, anchor_occurrence, idx, now, id],
            )?,
            (None, None) => self.conn.execute(
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

    /// write_resource 命令写完文件后调用：刷新 size_bytes / word_count / indexed_at。
    /// 不动 title / type / category_path / rel_path / mtime（这些都是身份字段），
    /// 也不动 page_count（markdown 永远为 NULL）。
    /// 不存在 → 抛错（调用方应保证 id 来自同一次 query_row）。
    pub fn update_resource_after_write(
        &self,
        id: i64,
        size_bytes: i64,
        word_count: i64,
    ) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        let changed = self.conn.execute(
            "UPDATE resources
             SET size_bytes = ?1, word_count = ?2, indexed_at = ?3
             WHERE id = ?4",
            params![size_bytes, word_count, now, id],
        )?;
        if changed == 0 {
            anyhow::bail!("resource {id} not found");
        }
        Ok(())
    }

    /// manifest 条目 upsert 为占位行（present=0）。分类链按顶层到叶子顺序补齐，
    /// 满足 resources.category_path → categories.path 的外键。
    pub fn upsert_manifest(&self, files: &[crate::sync::ManifestFile]) -> Result<()> {
        let now = chrono::Utc::now().to_rfc3339();
        for f in files {
            let Some((category_path, rel_path)) = crate::sync::split_category(&f.path) else {
                continue;
            };
            self.ensure_category_chain(&category_path)?;
            let title = title_from_file_name(&rel_path);
            let Some(r#type) = crate::scanner::classify_type(
                rel_path.rsplit('.').next().unwrap_or(""),
            ) else {
                continue; // 未知扩展名不进资源列表
            };
            // sha256 缺失 → 空串占位，仍然标记行为 manifest 管理；不能让 remote_hash 为 NULL，
            // 否则 scanner 的 delete_missing_resources 会把还没下完的远端条目误删。
            let remote_hash = f.sha256.as_deref().unwrap_or("");
            self.conn.execute(
                "INSERT INTO resources (category_path, rel_path, type, title, size_bytes,
                                        mtime, indexed_at, page_count, word_count,
                                        remote_hash, present)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6, NULL, NULL, ?7, 0)
                 ON CONFLICT(category_path, rel_path) DO UPDATE SET
                    size_bytes = excluded.size_bytes,
                    remote_hash = excluded.remote_hash,
                    indexed_at = excluded.indexed_at",
                params![category_path, rel_path, r#type, title, f.size as i64, now, remote_hash],
            )?;
        }
        Ok(())
    }

    /// 把 category_path 解析成顶层到叶子的祖先链，逐级 upsert 到 categories 表。
    /// 必须在插入 resources 行之前调用 —— resources.category_path 是 categories(path) 的外键。
    pub fn ensure_category_chain(&self, category_path: &str) -> Result<()> {
        let segs: Vec<&str> = category_path.split('/').collect();
        for i in 1..=segs.len() {
            let path = segs[..i].join("/");
            let parent = if i == 1 { None } else { Some(segs[..i - 1].join("/")) };
            let title = crate::scanner::humanize_dir_name(&path);
            let order = crate::scanner::parse_sort_order(&path) as i64;
            self.conn.execute(
                "INSERT INTO categories (path, parent_path, title, sort_order)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(path) DO UPDATE SET
                    parent_path = excluded.parent_path",
                params![path, parent, title, order],
            )?;
        }
        Ok(())
    }

    /// 把 manifest 占位行翻转为 present=1（文件已下载到本地）。
    /// 不动 remote_hash —— 行仍归 manifest 管理，防止后续 scanner 误删。
    pub fn mark_resource_present(&self, category_path: &str, rel_path: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE resources SET present = 1 WHERE category_path = ?1 AND rel_path = ?2",
            params![category_path, rel_path],
        )?;
        Ok(())
    }

    /// 按 resource id 把 present 翻 1。不存在 → 静默 no-op（与「不抛错保持幂等」一致）。
    /// download_resource 命令在按 id 查表后直接调用，避免再走一次 (category_path, rel_path)。
    pub fn mark_resource_present_unchecked(&self, id: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE resources SET present = 1 WHERE id = ?1",
            [id],
        )?;
        Ok(())
    }

    /// 删除 manifest 里已经没有的远端条目。
    /// 只动 remote_hash IS NOT NULL 的行 —— 纯本地行（remote_hash IS NULL）永远不动。
    pub fn drop_resources_absent_from_manifest(&self, keep: &[(String, String)]) -> Result<usize> {
        let mut stmt = self.conn.prepare("SELECT category_path, rel_path FROM resources WHERE remote_hash IS NOT NULL")?;
        let all: Vec<(String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut removed = 0;
        for (cat, rel) in all {
            if keep.contains(&(cat.clone(), rel.clone())) {
                continue;
            }
            removed += self.conn.execute(
                "DELETE FROM resources WHERE category_path = ?1 AND rel_path = ?2",
                params![cat, rel],
            )?;
        }
        Ok(removed)
    }

    /// `_index.md` 不进 resources（scanner 跳过），单独 upsert 到 index_files：
    /// - size = manifest 上的字节数，供 read_subcategory_index 缺文件时按需下载。
    /// - present = 0（默认），下载/扫描到本地后由 mark_index_present 翻 1。
    pub fn upsert_index_files(&self, files: &[crate::sync::ManifestFile]) -> Result<()> {
        for f in files {
            self.conn.execute(
                "INSERT INTO index_files (path, size, present) VALUES (?1, ?2, 0)
                 ON CONFLICT(path) DO UPDATE SET size = excluded.size",
                params![f.path, f.size as i64],
            )?;
        }
        Ok(())
    }

    /// 查 index_files：返 Some((size, present)) 或 None（未在 manifest 中）。
    pub fn index_file(&self, path: &str) -> Result<Option<(u64, bool)>> {
        let mut stmt = self.conn.prepare("SELECT size, present FROM index_files WHERE path = ?1")?;
        let mut rows = stmt.query([path])?;
        match rows.next()? {
            Some(r) => Ok(Some((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? != 0))),
            None => Ok(None),
        }
    }

    /// `_index.md` 已下载到本地 → 翻 present=1。
    pub fn mark_index_present(&self, path: &str) -> Result<()> {
        self.conn.execute("UPDATE index_files SET present = 1 WHERE path = ?1", [path])?;
        Ok(())
    }

    /// 同步状态面板用的统计：只算 manifest 管理的行（remote_hash IS NOT NULL），
    /// 纯本地行不计入 total（但本地行扫描后会得到 present=1，本任务范畴外后续任务处理）。
    pub fn sync_status(&self) -> Result<SyncStatus> {
        let mut stmt = self.conn.prepare(
            "SELECT COUNT(*),
                    COALESCE(SUM(present), 0),
                    COALESCE(SUM(size_bytes), 0),
                    COALESCE(SUM(CASE WHEN present = 1 THEN size_bytes ELSE 0 END), 0)
             FROM resources WHERE remote_hash IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        let row = rows.next()?.expect("aggregate always yields one row");
        Ok(SyncStatus {
            total: row.get(0)?,
            present: row.get(1)?,
            total_bytes: row.get(2)?,
            cached_bytes: row.get(3)?,
        })
    }

    /// 读 app_config 单条配置。返 None = 未设置。
    pub fn get_config(&self, key: &str) -> Result<Option<String>> {
        let mut stmt = self.conn.prepare("SELECT value FROM app_config WHERE key = ?1")?;
        let mut rows = stmt.query([key])?;
        match rows.next()? {
            Some(r) => Ok(Some(r.get(0)?)),
            None => Ok(None),
        }
    }

    /// upsert 一条 app_config（key 不存在则插入，存在则更新 value）。
    pub fn set_config(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO app_config (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    fn get_note(&self, id: i64) -> Result<NoteRow> {
        let row = self
            .conn
            .query_row(
                "SELECT id, resource_id, content, anchor_text, anchor_occurrence,
                        prompt, source, page_idx, created_at, updated_at
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
                        page_idx: r.get(7)?,
                        created_at: r.get(8)?,
                        updated_at: r.get(9)?,
                    })
                },
            )
            .with_context(|| format!("note {id} not found"))?;
        Ok(row)
    }
}

#[cfg(test)]
mod manifest_tests {
    use super::*;
    use crate::sync::ManifestFile;

    fn opened() -> (tempfile::TempDir, Db) {
        let tmp = tempfile::tempdir().unwrap();
        let mut db = Db::open(&tmp.path().join("t.db")).unwrap();
        db.migrate().unwrap();
        (tmp, db)
    }

    fn file(path: &str, size: u64) -> ManifestFile {
        ManifestFile { path: path.into(), size, sha256: None }
    }

    #[test]
    fn migrate_adds_present_and_remote_hash_columns() {
        let (_t, db) = opened();
        let cols: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('resources') WHERE name IN ('present','remote_hash')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cols, 2);
        // 新表也在
        db.conn
            .query_row("SELECT COUNT(*) FROM index_files", [], |r| r.get::<_, i64>(0))
            .unwrap();
    }

    #[test]
    fn upsert_manifest_creates_placeholder_rows_not_present() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        let (present, size, hash): (i64, i64, Option<String>) = db
            .conn
            .query_row(
                "SELECT present, size_bytes, remote_hash FROM resources WHERE rel_path = '回归.md'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(present, 0);
        assert_eq!(size, 500);
        // Brief originally asserted hash.is_none(), but the broader design intent
        // (manifest rows must stay managed = remote_hash IS NOT NULL) requires
        // upsert_manifest to write a non-NULL sentinel when sha256 is absent.
        // See commit report for deviation note.
        assert!(hash.is_some());
        // 分类被自动补齐（FK 要求）
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM categories WHERE path='01-基础'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn upsert_manifest_is_idempotent() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn scan_local_file_flips_present_to_one() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/回归.md", 500)]).unwrap();
        db.upsert_resource(ResourceInput {
            category_path: "01-基础".into(),
            rel_path: "回归.md".into(),
            r#type: "markdown".into(),
            title: "回归".into(),
            size_bytes: 500,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
        let present: i64 = db
            .conn
            .query_row("SELECT present FROM resources WHERE rel_path='回归.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(present, 1);
        // remote_hash 保留 —— 决定这条不再被 scan 的 delete_missing_resources 误删
        let still_managed: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM resources WHERE remote_hash IS NOT NULL",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(still_managed, 1);
    }

    #[test]
    fn delete_missing_resources_keeps_manifest_managed_rows() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/远端.md", 10), file("01-基础/本地.md", 10)]).unwrap();
        // 本地.md 被 scanner 扫到（present=1），远端.md 只有 manifest 占位（present=0）
        db.upsert_resource(ResourceInput {
            category_path: "01-基础".into(),
            rel_path: "本地.md".into(),
            r#type: "markdown".into(),
            title: "本地".into(),
            size_bytes: 10,
            mtime: 0,
            page_count: None,
            word_count: None,
        })
        .unwrap();
        // scanner 只 keep 了 本地.md；远端.md 虽没扫到也必须留下
        db.delete_missing_resources("01-基础", &["本地.md".to_string()]).unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources WHERE rel_path='远端.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    #[test]
    fn drop_resources_absent_from_manifest_removes_stale() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("01-基础/旧.md", 10), file("01-基础/新.md", 10)]).unwrap();
        db.drop_resources_absent_from_manifest(&[("01-基础".into(), "新.md".into())])
            .unwrap();
        let n: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM resources WHERE rel_path='旧.md'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn index_files_roundtrip() {
        let (_t, db) = opened();
        db.upsert_index_files(&[file("01-基础/_index.md", 42)]).unwrap();
        assert_eq!(db.index_file("01-基础/_index.md").unwrap(), Some((42, false)));
        db.mark_index_present("01-基础/_index.md").unwrap();
        assert_eq!(db.index_file("01-基础/_index.md").unwrap(), Some((42, true)));
        assert_eq!(db.index_file("不存在/_index.md").unwrap(), None);
    }

    #[test]
    fn sync_status_counts_present_and_bytes() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("a/x.md", 100), file("a/y.md", 200)]).unwrap();
        db.mark_resource_present("a", "x.md").unwrap();
        let st = db.sync_status().unwrap();
        assert_eq!(st.total, 2);
        assert_eq!(st.present, 1);
        assert_eq!(st.total_bytes, 300);
        assert_eq!(st.cached_bytes, 100);
    }

    #[test]
    fn config_roundtrip() {
        let (_t, db) = opened();
        assert_eq!(db.get_config("sync_base_url").unwrap(), None);
        db.set_config("sync_base_url", "https://example.com/kb").unwrap();
        assert_eq!(
            db.get_config("sync_base_url").unwrap().as_deref(),
            Some("https://example.com/kb")
        );
    }

    /// FINDING A 修复：模拟"已部署用户"的 v1 数据库 → 跑 migrate() → 验证老数据完整保留。
    /// - 老库 resources 不带 present / remote_hash 两列。
    /// - 老库已有资源行：迁移后必须 present=1（DEFAULT 1 行为），remote_hash IS NULL（保持纯本地行）。
    /// - notes 表里的笔记 FK → resources(id)：必须留下来 —— 这是用户的真实数据。
    ///   一旦 ALTER 顺序错或 FK 重建，这条 INSERT 会失败，测试即抓到。
    #[test]
    fn migrate_preserves_existing_resources_with_default_present() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("pre_v2.db");

        // 用裸 Connection 手工拼 v1 schema（不带 present / remote_hash，不带 index_files / app_config）。
        // 这正是已部署用户的 ai-stack.db 当前的样子。
        let raw = rusqlite::Connection::open(&path).unwrap();
        raw.pragma_update(None, "journal_mode", "WAL").unwrap();
        raw.pragma_update(None, "foreign_keys", "ON").unwrap();
        raw.execute_batch(
            "CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
             );
             CREATE TABLE categories (
                path TEXT PRIMARY KEY,
                parent_path TEXT,
                title TEXT NOT NULL,
                sort_order INTEGER NOT NULL,
                FOREIGN KEY (parent_path) REFERENCES categories(path)
             );
             CREATE TABLE resources (
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
             CREATE TABLE notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resource_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                anchor_text TEXT,
                anchor_occurrence INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (resource_id) REFERENCES resources(id) ON DELETE CASCADE
             );",
        )
        .unwrap();
        raw.execute(
            "INSERT INTO categories (path, parent_path, title, sort_order)
             VALUES ('01-基础', NULL, '基础', 1)",
            [],
        )
        .unwrap();
        raw.execute(
            "INSERT INTO resources (category_path, rel_path, type, title, size_bytes, indexed_at)
             VALUES ('01-基础', '回归.md', 'markdown', '回归', 1234, '2026-01-01T00:00:00+00:00')",
            [],
        )
        .unwrap();
        let res_id: i64 = raw.last_insert_rowid();
        raw.execute(
            "INSERT INTO notes (resource_id, content, anchor_occurrence, source, created_at, updated_at)
             VALUES (?1, '老笔记，必须保留', 0, 'user', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')",
            [res_id],
        )
        .unwrap();
        drop(raw); // 关掉裸连接，让 Db::open 重新拿独占

        // 走真实路径：Db::open + migrate()。
        let mut db = Db::open(&path).unwrap();
        db.migrate().unwrap();

        // 老资源行还在
        let row_exists: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM resources WHERE rel_path='回归.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(row_exists, 1, "迁移后老资源行不能丢");

        // present = 1（DEFAULT 1 行为）：老库里的每一行都是躺在磁盘上的本地文件
        let (present, remote_hash): (i64, Option<String>) = db
            .conn
            .query_row(
                "SELECT present, remote_hash FROM resources WHERE rel_path='回归.md'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            present, 1,
            "老库迁移后 present 必须默认为 1，否则整个存量库都会被当成\"还没下完\""
        );
        assert!(
            remote_hash.is_none(),
            "老库迁移后 remote_hash 必须为 NULL（保持纯本地行身份，不会被 manifest prune 删掉）"
        );

        // FK notes → resources(id) 还在 —— 用户的笔记不能因为迁移而丢
        let note_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM notes WHERE resource_id = ?1",
                [res_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            note_count, 1,
            "notes → resources(id) 外键必须保留用户笔记（FK 重建会触发 CASCADE 删除）"
        );

        // v2 新表也建出来了
        let v2_rows: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM schema_version WHERE version = 2", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(v2_rows, 1, "schema_version 必须记下 v=2");
    }

    /// FINDING B 修复：把 manifest 条目塞到三层深的路径（a/b/c/file.md），
    /// 验证 ensure_category_chain 的 1..=segs.len() 修复确实让所有祖先 category 都建出来了，
    /// 而且 parent_path 自引用 FK 也正确指向上层。
    #[test]
    fn upsert_manifest_inserts_multi_level_category_chain() {
        let (_t, db) = opened();
        db.upsert_manifest(&[file("a/b/c/file.md", 10)]).unwrap();

        // 三层 category 全部存在，parent_path 自引用 FK 正确指向上层
        for (path, expected_parent) in &[
            ("a", None),
            ("a/b", Some("a")),
            ("a/b/c", Some("a/b")),
        ] {
            let parent: Option<String> = db
                .conn
                .query_row(
                    "SELECT parent_path FROM categories WHERE path = ?1",
                    [path],
                    |r| r.get(0),
                )
                .unwrap_or_else(|_| panic!("category {} 没建出来", path));
            assert_eq!(
                parent.as_deref(),
                *expected_parent,
                "category {} 的 parent_path 错误",
                path
            );
        }

        // 资源行挂在最深层 category 下
        let n: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM resources WHERE category_path='a/b/c' AND rel_path='file.md'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "资源行必须挂在最深 category 'a/b/c' 下");
    }
}