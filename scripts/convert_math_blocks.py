"""Convert ```math ... ``` blocks to $$ ... $$ across all knowledge articles.

Why: ```math is a fenced code block (info-string = "math") that the
frontend's `renderMathBlocks` recognises. `$$ ... $$` is a display-math
delimiter that KaTeX auto-render handles directly. Switching the source
to `$$ ... $$` means there's no per-block KaTeX plumbing in
MarkdownReader.tsx — auto-render does everything.

Regex shape: `` ```math\nCONTENT\n``` `` → `` $$\nCONTENT\n$$ ``
Non-greedy `.+?` lets multiple blocks in the same file be replaced
independently.

The opening fence is `^```math\s*$` (no other info-string accepted), the
closing fence is `^```\s*$` (must be a bare triple-backtick — matches
what the rest of the file uses, e.g. ```` ```python ````, ```` ``` ````).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("C:/project/ai-stack/resources/knowledge")

# 形如：
#   ```math
#   H(P) = -\sum_x P(x) \log P(x)
#   ```
# 匹配要点：
# - 开头必须是 3 个反引号 + math + 行尾，结尾必须是 3 个反引号独占一行
# - 中间是非空的 LaTeX 源码（至少一行，不能空块）
# - 全文用 DOTALL 让 `.` 匹配换行，? 配 MULTILINE 让 `^` 锚行首
PATTERN = re.compile(
    r"^```math\s*\n(?P<body>.+?)^```\s*$",
    flags=re.MULTILINE | re.DOTALL,
)


def convert(text: str) -> str:
    return PATTERN.sub(lambda m: "$$\n" + m.group("body") + "$$", text)


def main() -> None:
    converted_files: list[Path] = []
    for path in sorted(ROOT.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        new = convert(original)
        if new != original:
            path.write_text(new, encoding="utf-8")
            converted_files.append(path)

    print(f"converted {len(converted_files)} files")
    for p in converted_files:
        # 每文件展示转换次数
        orig = p.read_text(encoding="utf-8")
        count = PATTERN.subn(lambda m: "", orig)[1]
        rel = p.relative_to(ROOT.parent)
        print(f"  {rel}: {count} blocks")


if __name__ == "__main__":
    main()