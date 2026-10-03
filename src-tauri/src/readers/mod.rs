pub mod markdown;
pub mod pdf;
pub mod docx;
pub mod pptx;

pub use markdown::extract as markdown_extract;
pub use pdf::count_pages as pdf_count_pages;
// DOCX / PPTX 不在后端解析为结构化 blocks / slides —— 前端 mammoth / pptxviewjs
// 各自处理自家格式的高保真渲染。后端只剩"数一数"（word 数 / slide 数），
// 留给 DB 索引和 UI 列表用。
pub use docx::word_count as docx_word_count;
pub use pptx::slide_count as pptx_slide_count;