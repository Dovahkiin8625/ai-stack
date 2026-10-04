#!/usr/bin/env node
/**
 * 把 resources/knowledge/<cat>/<sub>/ 下的 pdf/docx/pptx 移到
 *   <cat>/<sub>/三方资料/
 * 并在 <cat>/<sub>/_index.md 末尾追加 `## 三方资料` 章节（标题 + 介绍）。
 *
 * 用法：node scripts/migrate-third-party.mjs [--dry-run]
 * 默认 dry-run（只打印要做什么），确认无误后加 --apply 真正执行。
 */
import { promises as fs } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(process.cwd(), 'resources', 'knowledge');
const THIRD_PARTY = '三方资料';

// 完整的第三方资料 manifest：path → [title, description]
// title 用于 index 列表显示；description 用于简介。
// 路径使用 POSIX 风格（相对于 resources/knowledge）。
const MANIFEST = {
  '02-deep-learning/cnn': [
    ['efficientnet.pdf', 'EfficientNet 论文', 'Tan & Le 2019 提出的复合缩放方法，统一缩放深度/宽度/分辨率。'],
    ['resnet.pdf', 'ResNet 论文', 'He et al. 2015 提出的深度残差学习，ImageNet 图像分类里程碑。'],
  ],
  '02-deep-learning/frameworks': [
    ['pytorch.pdf', 'PyTorch 论文', 'Paszke et al. 2019 介绍 PyTorch 的设计与实现，张量 + 动态计算图。'],
    ['tensorflow.pdf', 'TensorFlow 论文', 'Abadi et al. 2016 介绍 TensorFlow 大规模机器学习系统的设计。'],
  ],
  '02-deep-learning/neural-network-fundamentals': [
    ['batchnorm.pdf', 'BatchNorm 论文', 'Ioffe & Szegedy 2015 通过 mini-batch 归一化加速深层网络训练。'],
    ['dropout.pdf', 'Dropout 论文', 'Srivastava et al. 2014 通过随机失活神经元防止过拟合。'],
  ],
  '02-deep-learning/rnn-lstm': [
    ['gru.pdf', 'GRU 论文', 'Cho et al. 2014 提出的简化门控循环单元。'],
    ['lstm-1997.pdf', 'LSTM 原始论文', 'Hochreiter & Schmidhuber 1997 提出长短期记忆网络。'],
    ['seq2seq.pdf', 'Seq2Seq 论文', 'Sutskever et al. 2014 端到端序列到序列学习。'],
  ],
  '02-deep-learning/training-techniques': [
    ['adam.pdf', 'Adam 优化器论文', 'Kingma & Ba 2014 自适应矩估计优化器。'],
    ['adamw.pdf', 'AdamW 论文', 'Loshchilov & Hutter 2019 解耦权重衰减的 Adam。'],
    ['mixed-precision.pdf', '混合精度训练论文', 'Micikevicius et al. 2017 半精度训练加速深度学习。'],
  ],
  '02-deep-learning/transformers': [
    ['Attention_Is_All_You_Need.pdf', 'Attention Is All You Need', 'Vaswani et al. 2017 Transformer 原始论文，奠基之作。'],
    ['sd1.pptx', 'Stable Diffusion 讲解稿', 'Latent Diffusion 模型的图文讲解。'],
    ['transformer.docx', 'Transformer 综述讲义', 'Transformer 架构详细讲义（Word 文档）。'],
  ],
  '03-large-language-models/architecture-pretraining': [
    ['bert.pdf', 'BERT 论文', 'Devlin et al. 2018 双向编码表示预训练。'],
    ['gpt3.pdf', 'GPT-3 论文', 'Brown et al. 2020 大语言模型的少样本学习能力。'],
    ['llama.pdf', 'LLaMA 论文', 'Touvron et al. 2023 开源基础大模型。'],
    ['t5.pdf', 'T5 论文', 'Raffel et al. 2020 文本到文本迁移 Transformer。'],
  ],
  '03-large-language-models/fine-tuning': [
    ['dpo.pdf', 'DPO 论文', 'Rafailov et al. 2023 直接偏好优化。'],
    ['lora.pdf', 'LoRA 论文', 'Hu et al. 2021 大模型低秩适配微调。'],
    ['rlhf.pdf', 'RLHF 论文', 'Christiano et al. 2017 / Ouyang et al. 2022 人类反馈强化学习。'],
  ],
  '03-large-language-models/inference-optimization': [
    ['flash-attention.pdf', 'FlashAttention 论文', 'Dao et al. 2022 IO 感知的精确注意力算法。'],
    ['vllm.pdf', 'vLLM 论文', 'Kwon et al. 2023 PagedAttention 高吞吐推理引擎。'],
  ],
  '03-large-language-models/multimodal-llm': [
    ['clip.pdf', 'CLIP 论文', 'Radford et al. 2021 对比图文预训练。'],
    ['llava.pdf', 'LLaVA 论文', 'Liu et al. 2023 视觉指令调优。'],
  ],
  '03-large-language-models/rag': [
    ['rag.pdf', 'RAG 论文', 'Lewis et al. 2020 检索增强生成。'],
  ],
  '04-computer-vision/3d-vision': [
    ['nerf.pdf', 'NeRF 论文', 'Mildenhall et al. 2020 神经辐射场 3D 重建。'],
    ['pointnet.pdf', 'PointNet 论文', 'Qi et al. 2017 深度学习处理点云的开创工作。'],
  ],
  '04-computer-vision/image-classification': [
    ['vit.pdf', 'ViT 论文', 'Dosovitskiy et al. 2021 视觉 Transformer。'],
  ],
  '04-computer-vision/image-generation': [
    ['ddpm.pdf', 'DDPM 论文', 'Ho et al. 2020 去噪扩散概率模型。'],
    ['gan.pdf', 'GAN 论文', 'Goodfellow et al. 2014 生成对抗网络。'],
    ['stable-diffusion.pdf', 'Stable Diffusion 论文', 'Rombach et al. 2022 潜在扩散模型。'],
  ],
  '04-computer-vision/object-detection': [
    ['detr.pdf', 'DETR 论文', 'Carion et al. 2020 端到端目标检测。'],
    ['faster-rcnn.pdf', 'Faster R-CNN 论文', 'Ren et al. 2015 区域提议网络。'],
    ['yolo.pdf', 'YOLO 论文', 'Redmon et al. 2016 单阶段实时目标检测。'],
  ],
  '04-computer-vision/segmentation': [
    ['mask-rcnn.pdf', 'Mask R-CNN 论文', 'He et al. 2017 实例分割。'],
    ['sam.pdf', 'SAM 论文', 'Kirillov et al. 2023 Segment Anything 通用分割模型。'],
    ['unet.pdf', 'U-Net 论文', 'Ronneberger et al. 2015 生物医学图像分割。'],
  ],
  '04-computer-vision/video-understanding': [
    ['timesformer.pdf', 'TimeSformer 论文', 'Bertasius et al. 2021 视频理解的时空注意力。'],
    ['vivit.pdf', 'ViViT 论文', 'Arnab et al. 2021 视频 Vision Transformer。'],
  ],
  '05-nlp/information-extraction': [
    ['uie.pdf', 'UIE 论文', 'Lu et al. 2022 统一信息抽取。'],
  ],
  '05-nlp/machine-translation': [
    ['gnmt.pdf', 'GNMT 论文', 'Wu et al. 2016 Google 神经机器翻译系统。'],
  ],
  '05-nlp/question-answering': [
    ['drqa.pdf', 'DrQA 论文', 'Chen et al. 2017 开放域问答的检索 + 阅读理解。'],
    ['squad.pdf', 'SQuAD 论文', 'Rajpurkar et al. 2016 斯坦福问答数据集。'],
  ],
  '05-nlp/sequence-labeling': [
    ['bilstm-crf.pdf', 'BiLSTM-CRF 论文', 'Huang et al. 2015 序列标注。'],
  ],
  '05-nlp/text-representation': [
    ['glove.pdf', 'GloVe 论文', 'Pennington et al. 2014 词向量全局对数双线性回归。'],
    ['sentence-bert.pdf', 'Sentence-BERT 论文', 'Reimers & Gurevych 2019 句向量表征。'],
    ['word2vec.pdf', 'Word2Vec 论文', 'Mikolov et al. 2013 分布式词表示。'],
  ],
  '06-speech-audio/asr': [
    ['wav2vec2.pdf', 'wav2vec 2.0 论文', 'Baevski et al. 2020 自监督语音表示。'],
    ['whisper.pdf', 'Whisper 论文', 'Radford et al. 2022 鲁棒语音识别大模型。'],
  ],
  '06-speech-audio/audio-generation': [
    ['audioldm.pdf', 'AudioLDM 论文', 'Liu et al. 2023 潜在扩散音频生成。'],
    ['musicgen.pdf', 'MusicGen 论文', 'Copet et al. 2023 文本到音乐生成。'],
  ],
  '06-speech-audio/tts': [
    ['tacotron2.pdf', 'Tacotron 2 论文', 'Shen et al. 2018 端到端语音合成。'],
    ['vits.pdf', 'VITS 论文', 'Kim et al. 2021 端到端 TTS + GAN。'],
  ],
  '06-speech-audio/voice-cloning': [
    ['valle.pdf', 'VALL-E 论文', 'Wang et al. 2023 神经语音语言模型。'],
  ],
  '08-ai-agents/architectures': [
    ['react.pdf', 'ReAct 论文', 'Yao et al. 2022 推理 + 行动协同框架。'],
  ],
  '08-ai-agents/memory-systems': [
    ['memgpt.pdf', 'MemGPT 论文', 'Packer et al. 2023 分层记忆的 LLM 智能体。'],
  ],
  '08-ai-agents/multi-agent': [
    ['autogen.pdf', 'AutoGen 论文', 'Wu et al. 2023 多智能体对话框架。'],
  ],
  '08-ai-agents/planning-reasoning': [
    ['cot.pdf', 'Chain-of-Thought 论文', 'Wei et al. 2022 思维链提示。'],
    ['tree-of-thoughts.pdf', 'Tree of Thoughts 论文', 'Yao et al. 2023 思维树搜索。'],
  ],
  '08-ai-agents/tool-use': [
    ['gorilla.pdf', 'Gorilla 论文', 'Patil et al. 2023 大模型调用 API。'],
    ['toolformer.pdf', 'Toolformer 论文', 'Schick et al. 2023 自监督学习使用工具。'],
  ],
};

const APPLY = process.argv.includes('--apply');

function fmtSize(n) {
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / 1024 / 1024).toFixed(1)}MB`;
}

async function statOrNull(p) {
  try { return await fs.stat(p); } catch { return null; }
}

async function moveOne(srcAbs, dstAbs, fileName) {
  if (APPLY) {
    await fs.rename(srcAbs, dstAbs);
  }
  const s = await statOrNull(srcAbs);
  return s ? fmtSize(s.size) : '(missing)';
}

function buildIndexSection(files) {
  const lines = [
    '',
    '## 三方资料',
    '',
  ];
  for (const [file, title, desc] of files) {
    lines.push(`- [${title}](./${THIRD_PARTY}/${file}) — ${desc}`);
  }
  lines.push('');
  return lines.join('\n');
}

async function processSubcategory(relCat, files) {
  const catAbs = path.join(ROOT, relCat);
  const thirdPartyAbs = path.join(catAbs, THIRD_PARTY);
  const indexAbs = path.join(catAbs, '_index.md');

  const existing = await statOrNull(catAbs);
  if (!existing || !existing.isDirectory()) {
    console.warn(`[SKIP] ${relCat}: 目录不存在`);
    return { cat: relCat, moved: 0, skipped: 0 };
  }

  let moved = 0;
  let skipped = 0;
  const missing = [];

  // 1) 确保 三方资料/ 目录存在
  if (APPLY) {
    await fs.mkdir(thirdPartyAbs, { recursive: true });
  }

  // 2) 移动文件
  for (const [file] of files) {
    const src = path.join(catAbs, file);
    const dst = path.join(thirdPartyAbs, file);
    const srcStat = await statOrNull(src);
    const dstStat = await statOrNull(dst);

    if (dstStat) {
      // 目标已存在（之前可能跑过）
      skipped += 1;
      console.log(`  [exists] ${relCat}/三方资料/${file}`);
      if (srcStat && APPLY) {
        // 源还在 —— 删除（保持幂等）
        await fs.unlink(src);
        moved += 1;
        console.log(`  [cleaned] 移除源 ${relCat}/${file}`);
      }
      continue;
    }
    if (!srcStat) {
      missing.push(file);
      console.warn(`  [MISSING] ${relCat}/${file}`);
      continue;
    }
    await moveOne(src, dst, file);
    moved += 1;
    console.log(`  [moved] ${relCat}/${file} → ${THIRD_PARTY}/${file}`);
  }

  // 3) 更新 _index.md（追加 ## 三方资料 章节）
  const section = buildIndexSection(files);
  if (APPLY) {
    let cur = '';
    try { cur = await fs.readFile(indexAbs, 'utf8'); } catch { /* 不存在就新建 */ }
    if (cur.includes('## 三方资料')) {
      console.log(`  [skip] ${relCat}/_index.md 已含 ## 三方资料，跳过追加`);
    } else {
      const sep = cur.endsWith('\n') ? '\n' : '\n\n';
      await fs.writeFile(indexAbs, cur + sep + section.trimStart(), 'utf8');
      console.log(`  [index] ${relCat}/_index.md 已追加 ## 三方资料`);
    }
  } else {
    console.log(`  [dry-run] 将向 ${relCat}/_index.md 追加 ## 三方资料 (${files.length} 条)`);
  }

  if (missing.length) {
    console.warn(`  [WARN] ${relCat}: ${missing.length} 个源文件缺失：${missing.join(', ')}`);
  }

  return { cat: relCat, moved, skipped };
}

async function main() {
  if (!APPLY) {
    console.log('=== DRY-RUN（无 --apply，不修改任何文件）===');
  } else {
    console.log('=== APPLY（将真正修改文件）===');
  }

  const summary = [];
  for (const [cat, files] of Object.entries(MANIFEST)) {
    console.log(`\n[cat] ${cat}`);
    const r = await processSubcategory(cat, files);
    summary.push(r);
  }

  console.log('\n=== Summary ===');
  const totalMoved = summary.reduce((s, r) => s + r.moved, 0);
  const totalSkipped = summary.reduce((s, r) => s + r.skipped, 0);
  console.log(`subcategories: ${summary.length}, moved: ${totalMoved}, skipped: ${totalSkipped}`);
  if (!APPLY) console.log('加 --apply 真正执行');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
