// 生成 cargo test 用的微型目录树
import { mkdir, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..', 'src-tauri', 'tests', 'fixtures', 'knowledge-tree');

async function main() {
  // 清空
  await import('node:fs/promises').then((m) => m.rm(ROOT, { recursive: true, force: true }));

  // 01-foundations/01-mathematics/_index.md + note.md
  await mkdir(join(ROOT, '01-foundations', '01-mathematics'), { recursive: true });
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', '_index.md'),
    '# 数学基础\n\n线性代数、概率统计等。\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'linear-algebra-notes.md'),
    '# Linear Algebra Notes\n\nVectors and matrices.\n',
    'utf8',
  );
  await writeFile(
    join(ROOT, '01-foundations', '01-mathematics', 'calculus-notes.md'),
    '# Calculus Notes\n\nDerivatives and integrals.\n',
    'utf8',
  );

  // 02-deep-learning/_index.md (no resource) + sub empty
  await mkdir(join(ROOT, '02-deep-learning', 'transformers'), { recursive: true });
  await writeFile(
    join(ROOT, '02-deep-learning', '_index.md'),
    '# 深度学习\n',
    'utf8',
  );
  // 故意放一个 _index.md 但不放资源，验证 _index.md 不入 resources 表
  await writeFile(
    join(ROOT, '02-deep-learning', 'transformers', '_index.md'),
    '# Transformer\n',
    'utf8',
  );

  // fixtures for readers
  const readers = join(here, '..', 'src-tauri', 'tests', 'fixtures', 'readers');
  await mkdir(readers, { recursive: true });
  await writeFile(
    join(readers, 'simple.md'),
    '# Heading\n\nHello **world**.\n\n- item 1\n- item 2\n',
    'utf8',
  );
  // 最小化有效 PDF（使用 pdf-lib 构造 1 页文本 PDF）
  const { PDFDocument, StandardFonts } = await import('pdf-lib');
  const pdfDoc = await PDFDocument.create();
  const page = pdfDoc.addPage([612, 792]);
  const font = await pdfDoc.embedFont(StandardFonts.Helvetica);
  page.drawText('Hello PDF', { x: 100, y: 700, size: 24, font });
  const pdfBytes = await pdfDoc.save();
  await writeFile(join(readers, 'sample.pdf'), pdfBytes);
  // 使用 docx npm 包生成 .docx（devDependency）
  const { Document, Packer, Paragraph, HeadingLevel, Table, TableRow, TableCell } = await import('docx');
  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({ text: 'Top Heading', heading: HeadingLevel.HEADING_1 }),
        new Paragraph({ text: 'This is body paragraph one.' }),
        new Paragraph({ text: 'Sub Heading', heading: HeadingLevel.HEADING_2 }),
        new Paragraph({ text: 'More body text here.' }),
        new Table({
          rows: [
            new TableRow({ children: [new TableCell({ children: [new Paragraph('A')] }), new TableCell({ children: [new Paragraph('B')] })] }),
            new TableRow({ children: [new TableCell({ children: [new Paragraph('C')] }), new TableCell({ children: [new Paragraph('D')] })] }),
          ],
        }),
      ],
    }],
  });
  const buf = await Packer.toBuffer(doc);
  await writeFile(join(readers, 'sample.docx'), buf);
  const PptxGenJS = (await import('pptxgenjs')).default;
  const pres = new PptxGenJS();
  let slide = pres.addSlide();
  slide.addText('Slide One Title', { x: 0.5, y: 0.3, fontSize: 24 });
  slide.addText('First bullet', { x: 0.5, y: 1.0 });
  slide.addText('Second bullet', { x: 0.5, y: 1.5 });
  slide.addNotes('Speaker notes for slide one.');
  slide = pres.addSlide();
  slide.addText('Slide Two Title', { x: 0.5, y: 0.3, fontSize: 24 });
  slide.addText('Only one bullet', { x: 0.5, y: 1.0 });
  slide.addNotes('Notes for slide two.');
  const pptBuf = await pres.write({ outputType: 'nodebuffer' });
  await writeFile(join(readers, 'sample.pptx'), pptBuf);

  console.log(`fixtures written under ${ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });