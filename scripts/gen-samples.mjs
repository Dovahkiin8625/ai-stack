// 在 resources/knowledge/02-deep-learning/transformers/ 放置真实样本文件用于手动验证
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const TARGET = 'resources/knowledge/02-deep-learning/transformers';

async function main() {
  await mkdir(TARGET, { recursive: true });
  // Markdown
  await writeFile(
    join(TARGET, 'attention-mechanism.md'),
    `# Attention Mechanism

This note introduces the **scaled dot-product attention** used in Transformer models.

## Formula

$$\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$

## Variants

- Self-attention
- Cross-attention
- Multi-head attention
`,
    'utf8',
  );

  // PDF: 复用 fixtures 中的 sample.pdf
  // (留给用户手动放置一个真实 PDF；脚本只放其它格式)

  // DOCX
  const { Document, Packer, Paragraph, HeadingLevel, Table, TableRow, TableCell } = await import('docx');
  const doc = new Document({
    sections: [{
      children: [
        new Paragraph({ text: 'Transformer Overview', heading: HeadingLevel.HEADING_1 }),
        new Paragraph({ text: 'This document summarises the Transformer architecture.' }),
        new Paragraph({ text: 'Encoder', heading: HeadingLevel.HEADING_2 }),
        new Paragraph({ text: 'Stack of self-attention and feed-forward layers.' }),
        new Table({
          rows: [
            new TableRow({ children: [new TableCell({ children: [new Paragraph('Layer')] }), new TableCell({ children: [new Paragraph('Type')] })] }),
            new TableRow({ children: [new TableCell({ children: [new Paragraph('Self-Attention')] }), new TableCell({ children: [new Paragraph('Multi-Head')] })] }),
          ],
        }),
      ],
    }],
  });
  await writeFile(join(TARGET, 'transformer-overview.docx'), await Packer.toBuffer(doc));

  // PPTX
  const PptxGenJS = (await import('pptxgenjs')).default;
  const pres = new PptxGenJS();
  let s = pres.addSlide();
  s.addText('Why Transformers', { x: 0.5, y: 0.3, fontSize: 28 });
  s.addText('Parallel sequence modelling', { x: 0.5, y: 1.2, fontSize: 18 });
  s.addNotes('Highlight RNN vs Transformer parallelism.');
  s = pres.addSlide();
  s.addText('Self-Attention', { x: 0.5, y: 0.3, fontSize: 28 });
  s.addText('Q, K, V projections', { x: 0.5, y: 1.2, fontSize: 18 });
  s.addText('Scaled dot product', { x: 0.5, y: 1.7, fontSize: 18 });
  s.addNotes('Explain the scaling factor.');
  await writeFile(join(TARGET, 'transformer-slides.pptx'), await pres.write({ outputType: 'nodebuffer' }));

  console.log(`samples written under ${TARGET}`);
}

main().catch((e) => { console.error(e); process.exit(1); });