pub mod markdown;
pub mod pdf;
pub mod docx;
pub mod pptx;

pub use markdown::extract as markdown_extract;
pub use pdf::extract as pdf_extract;