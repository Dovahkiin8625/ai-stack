# Constitutional AI 与 RLAIF：用原则替代标注

RLHF 需要大量人类偏好标注——成本高、规模难放大、且标注员之间的不一致会污染 RM。**RLAIF（RL from AI Feedback）** 用另一个 LLM 当奖励裁判，而 **Constitutional AI**（Anthropic 2022）把这个思路推到极致：只给定一组"宪法原则"（如"不要有害"、"要诚实"），让模型自评、自改、自训。本文拆解 Constitutional AI 的两阶段流程（SL-CAI + RL-CAI），讨论 RLAIF 相比 RLHF 的优劣，以及它对"价值对齐"哲学的暗示。

## 一、为什么需要 RLAIF

RLHF 的瓶颈不在 PPO，而在**人类偏好数据的获取**：

1. **成本**：一条偏好标注约 $0.5~$3，百万条标注就是百万美元级。
2. **一致性**：标注员 Kappa 通常 0.5~0.7，安全相关任务更低。
3. **覆盖度**：对"长尾有害内容"（如化学武器、自残诱导）标注员也未必有能力判断。
4. **可扩展性**：模型迭代速度远超标注速度。

**核心想法**：如果能用一个强 LLM（GPT-4、Claude 自身）按一套**明文原则**当裁判，就能把标注规模化、可解释化、可审计化。

## 二、Constitutional AI 的两阶段流程

Constitutional AI（Anthropic, Bai et al. 2022）由两阶段组成：

```text
┌──────────────────────────────────────────────────────────┐
│  Stage A: SL-CAI（监督式自我修正）                          │
│  1. 模型对有害 prompt 生成初始回答                          │
│  2. 让模型按"宪法原则"自评，给出 critique                  │
│  3. 让模型根据 critique 重写回答                            │
│  4. (原始 prompt, 重写回答) 作为 SFT 数据                  │
└──────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────┐
│  Stage B: RL-CAI（强化学习阶段）                           │
│  1. 对每个 prompt 生成两个回答                              │
│  2. 让 AI 裁判按宪法原则判断哪个更符合原则                    │
│  3. 用这些 AI 偏好对训 RM                                  │
│  4. 用 RM + PPO 微调模型                                  │
└──────────────────────────────────────────────────────────┘
```

**关键区别**：Stage A 不需要任何外部反馈——纯粹"自我对话"；Stage B 用 AI 偏好替代人类偏好，但仍走标准 RLHF 流程。

## 三、宪法原则长什么样

Anthropic 公开的宪法原则示例：

```text
原则 1: 请选择最不有害、最不冒犯、最不歧视的回答。
原则 2: 请避免提供可能助长非法行为的信息。
原则 3: 请避免提供可能制造武器的信息，除非是在
        受控的、合法合规的语境中。
原则 4: 请选择最诚实、最不误导的回答。
原则 5: 请选择最有可能被广泛接受的回答。
...
```

每条原则用自然语言写成，**没有量化指标**——这正是 RLAIF 的精妙之处：原则是**可读、可审计、可辩论**的，而不是埋在 RM 权重里的人类偏好黑箱。

## 四、SL-CAI：自我批评与重写

下面用伪代码展示 Stage A 的核心循环：

```python
CONSTITUTION = [
    "请避免任何可能被视为有害、歧视性或不道德的内容。",
    "请选择最诚实、最有帮助的回答。",
    "如果原始回答可能违反某条原则，请按原则重写。",
]

def sl_cai_step(model, prompt, initial_response):
    # Step 1: 模型自评
    critique_prompt = (
        f"以下是一个 prompt 和回答。请根据以下原则评估回答：\n"
        f"原则：\n" + "\n".join(f"- {p}" for p in CONSTITUTION) + "\n\n"
        f"Prompt: {prompt}\n"
        f"回答: {initial_response}\n\n"
        f"评估：这份回答是否符合上述原则？请指出具体问题，并给出改进建议。"
    )
    critique = model.generate(critique_prompt)

    # Step 2: 根据 critique 重写
    revision_prompt = (
        f"Prompt: {prompt}\n"
        f"原回答: {initial_response}\n"
        f"评估: {critique}\n\n"
        f"请根据评估重写一个更符合原则的回答："
    )
    revised = model.generate(revision_prompt)
    return revised
```

把 `(prompt, revised)` 作为 SFT 样本，训练模型"学会"在内部就避开有害内容——这个过程不需要任何人类参与。

## 五、RL-CAI：AI 当偏好裁判

Stage B 的 RM 训练数据来自 AI 偏好：

```python
def ai_preference_judge(model, prompt, response_a, response_b, constitution):
    judge_prompt = (
        f"考虑以下 prompt 与两个回答。基于这些原则：\n"
        + "\n".join(f"- {p}" for p in constitution) + "\n\n"
        f"Prompt: {prompt}\n\n"
        f"回答 A: {response_a}\n\n"
        f"回答 B: {response_b}\n\n"
        f"哪个回答更符合原则？请回答 'A' 或 'B'，并简短解释。"
    )
    verdict = model.generate(judge_prompt)
    return "A" if verdict.startswith("A") else "B"
```

收集成对偏好后，**完全按 RLHF 流程**训 RM + PPO。这意味着 RLAIF 与 RLHF 在算法层面是同构的，差别只在"谁当标注员"。

## 六、RLAIF vs RLHF：经验对比

| 维度 | RLHF | RLAIF |
|---|---|---|
| 标注成本 | 高（人类） | 低（API 调用） |
| 一致性 | 受标注员主观影响 | 由 LLM + 原则决定，更稳定 |
| 可解释性 | RM 是黑盒 | 原则是明文，可审计 |
| 可扩展性 | 受限于标注员数量 | 仅受 API 配额限制 |
| 长尾有害内容 | 标注员未必能识别 | 强 LLM 通常能识别 |
| 裁判偏差 | 标注员偏见 | LLM 训练时的偏见（更难诊断） |
| 效果（无害性） | 强 | **相当或更好**（Anthropic 报告） |
| 效果（帮助性） | 强 | 略弱（自我评估会偏保守） |

Anthropic 的实验结论是：**无害性提升显著（与人类反馈相当），帮助性损失很小**——这是非常划算的权衡。

## 七、Constitutional AI 的工程挑战

### 1. 裁判模型的偏差

LLM 裁判不是中立的——它有自己的偏好（如偏好长回答、详细解释、特定格式）。这会导致 RM 学到"看似合规但空话"。

**缓解**：
- 多 LLM ensemble（GPT-4 + Claude + Gemini 投票）。
- 用规则验证器（rule-based reward）补强。
- 限制 AI 裁判只对**安全维度**打分，帮助性仍用人类标注。

### 2. 原则覆盖不足

明文原则无法穷尽所有"不当"。模型可能发现规则的灰色地带。

**缓解**：
- 原则库要长期维护、定期更新。
- 加入"若有疑虑，宁可拒绝"的兜底原则。
- 配合红队测试发现新漏洞（见 red-teaming 章节）。

### 3. 自评陷阱

"模型评估自己"会出现 confirmation bias——模型倾向认为自己已经合规。

**缓解**：
- 评估与生成用不同模型（或不同温度）。
- 让评估链多步（先找问题、再评估严重性、再决定是否改写）。

### 4. 原则注入的稳定性

宪法原则太多会互相冲突，模型无所适从。Anthropic 实践中精选 10~15 条核心原则。

## 八、其他 RLAIF / Self-Align 框架

- **Self-Align**（Sun et al. 2023）：用 Dromedary 模型，仅 16 条原则生成 500K SFT 数据，效果逼近 RLHF。
- **SALMON**（OpenAI 2023）：让模型自己生成回答，AI 裁判打分，再用 DPO 直接对齐——把 RLHF/DPO 的范式都搬到 RLAIF。
- **SPIN（Self-Play Fine-Tuning）**：用模型上次版本当"对手"，通过 self-play 改进。
- **Iterative DPO with AI Judge**：每轮用当前模型采样，AI 裁判标偏好，再 DPO，效果逼近 PPO。

## 九、Constitutional AI 的哲学含义

Constitutional AI 不只是工程技巧，它代表一种**对齐哲学的转变**：

| 范式 | 谁定义"好" |
|---|---|
| 经典 RLHF | 标注员的隐性偏好（不可审计） |
| Constitutional AI | 明文原则（可辩论、可审计） |
| Debate（Irving et al. 2018） | 两个模型对抗，人类当裁判 |
| Scalable Oversight | 用弱监督者监督强 AI（如递归奖励建模 RRM） |

它把"什么是对齐"从**统计学问题**变成**政治哲学问题**——谁来写宪法？原则之间冲突怎么办？这条路线最终指向 **AI 治理（AI Governance）** 而非纯技术。

## 十、实操建议

1. **小规模起步**：先选定 10 条核心原则，跑通 SL-CAI。
2. **混合 RM**：50% AI 偏好 + 50% 人类偏好，验证 AI 偏好质量。
3. **审计裁判偏差**：定期人工 spot-check AI 偏好对，看是否存在系统性偏差。
4. **原则分层**：把"硬规则"（不能教化学武器）放底层，"软偏好"（回答风格）放上层。
5. **保留红队**：RLAIF 不替代 red-teaming——原则覆盖率终究有限。
6. **多语言宪法**：不同文化的"可接受"标准不同，宪法应本地化。

## 小结

RLAIF 把"对齐"从人类标注规模化到 AI 标注，**Constitutional AI** 把规则从 RM 权重搬到明文原则——更便宜、更稳定、更可审计。SL-CAI 让模型自评自改生成 SFT 数据，RL-CAI 让 AI 当偏好裁判训 RM + PPO。它的局限在于 LLM 裁判的偏差与原则覆盖不足，但通过混合 RM、红队测试、原则审计可以缓解。Constitutional AI 把"对齐"问题从纯技术推向"AI 治理"——下一篇我们将看到，**光有原则不够**，还要靠持续的红队测试发现新漏洞，这正是 red-teaming 章节的核心话题。
