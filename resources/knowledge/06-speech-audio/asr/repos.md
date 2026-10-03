# 开源仓库：语音识别（ASR）

## OpenAI Whisper

- 仓库：https://github.com/openai/whisper
- 简介：Whisper 官方实现，680k 小时多语种数据训练的 encoder-decoder Transformer；提供 tiny/base/small/medium/large 五档模型。
- 适用：多语种转写、字幕生成、轻量本地推理。

## faster-whisper

- 仓库：https://github.com/SYSTRAN/faster-whisper
- 简介：用 CTranslate2 重写 Whisper 推理，速度比原版快 4×，显存占用更低。
- 适用：高吞吐 Whisper 服务。

## NVIDIA NeMo

- 仓库：https://github.com/NVIDIA/NeMo
- 简介：模块化语音 + 语音翻译 + 说话人识别框架，支持 Conformer / Citrinet / FastConformer 等模型训练与部署。
- 适用：研究级 ASR、TTS、Speaker Diarization。

## FunASR (阿里达摩院)

- 仓库：https://github.com/modelscope/FunASR
- 简介：达摩院开源的中文 ASR 工具包，包含 Paraformer-large 等工业级模型与端到端工具链。
- 适用：中文语音识别、字幕、工业部署。
