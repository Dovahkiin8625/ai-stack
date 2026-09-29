// scripts/scaffold-knowledge.mjs
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const ROOT = join(here, '..', 'resources', 'knowledge');

const TREE = {
  '01-foundations': ['01-mathematics', '02-computer-science', '03-ml-basics'],
  '02-deep-learning': [
    'neural-network-fundamentals', 'cnn', 'rnn-lstm',
    'transformers', 'training-techniques', 'frameworks',
  ],
  '03-large-language-models': [
    'architecture-pretraining', 'fine-tuning', 'prompt-engineering',
    'rag', 'llm-applications', 'inference-optimization', 'multimodal-llm',
  ],
  '04-computer-vision': [
    'image-classification', 'object-detection', 'segmentation',
    'image-generation', 'video-understanding', '3d-vision',
  ],
  '05-nlp': [
    'text-representation', 'sequence-labeling', 'text-generation',
    'machine-translation', 'question-answering', 'information-extraction',
  ],
  '06-speech-audio': ['asr', 'tts', 'voice-cloning', 'audio-generation'],
  '07-ai-engineering': [
    'model-deployment', 'model-serving', 'monitoring', 'data-engineering',
    'experiment-tracking', 'vector-databases', 'agent-frameworks',
  ],
  '08-ai-agents': [
    'architectures', 'tool-use', 'planning-reasoning',
    'multi-agent', 'memory-systems',
  ],
  '09-ai-safety': [
    'alignment', 'interpretability', 'red-teaming', 'bias-fairness', 'privacy',
  ],
  '10-applications': [
    'ai-for-science', 'ai-for-code', 'ai-for-education',
    'ai-for-healthcare', 'ai-for-finance', 'robotics',
  ],
  '11-tools-ecosystem': [
    'development-tools', 'cloud-platforms', 'open-source-models',
    'datasets', 'benchmarks',
  ],
  '12-industry-trends': [
    'frontier-papers', 'industry-reports', 'conferences', 'news-updates',
  ],
};

const TOP_TITLES = {
  '01-foundations': '基础理论',
  '02-deep-learning': '深度学习',
  '03-large-language-models': '大语言模型',
  '04-computer-vision': '计算机视觉',
  '05-nlp': '自然语言处理',
  '06-speech-audio': '语音与音频',
  '07-ai-engineering': 'AI 工程 / MLOps',
  '08-ai-agents': '智能体',
  '09-ai-safety': 'AI 安全与对齐',
  '10-applications': 'AI 应用',
  '11-tools-ecosystem': '工具与生态',
  '12-industry-trends': '行业与趋势',
};

const indexContent = (topTitle, subTitle, topPath, subPath) => `# ${subTitle}

> 分类：**${topTitle}** → **${subTitle}**
> 路径：\`resources/knowledge/${topPath}/${subPath}\`

本目录用于存放与「${subTitle}」相关的学习资料。

## 收录范围

- 教材与讲义（Markdown / PDF）
- 论文（PDF）
- 讲稿与笔记（Markdown / Word / PPTX）

## 命名约定

- 每个子主题一个文件夹，文件夹命名用 kebab-case。
- 每个文件夹下放一个 \`_index.md\` 作为目录索引（阶段 2 由索引生成器读取）。
- 文件名建议：\`YYYY-MM-DD-<title>.md\` 或原文件名。
`;

async function main() {
  let count = 0;
  for (const [top, subs] of Object.entries(TREE)) {
    await mkdir(join(ROOT, top), { recursive: true });
    for (const sub of subs) {
      const dir = join(ROOT, top, sub);
      await mkdir(dir, { recursive: true });
      const topTitle = TOP_TITLES[top] ?? top;
      const subTitle = sub
        .split('-')
        .map((s) => s.replace(/^\d+/, ''))
        .filter((s) => s.length > 0)
        .map((s) => s[0].toUpperCase() + s.slice(1))
        .join(' ');
      await writeFile(
        join(dir, '_index.md'),
        indexContent(topTitle, subTitle, top, sub),
        'utf8',
      );
      count++;
    }
  }
  console.log(`Created ${count} subdirectories under ${ROOT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
