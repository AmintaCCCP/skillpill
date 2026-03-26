# SkillPill

[English](#english) | [中文](#中文)

---

## English

### What is SkillPill?

**SkillPill** is an end-to-end pipeline for distilling classical agent skills, Python tools, and API-based capabilities into ultra-lightweight **MLX LoRA adapters** called **Pills**.

The long-term idea is simple: instead of paying the latency, privacy, and reliability cost of external tool calls every time an agent needs a capability, SkillPill converts that capability into compact local weights that can be hot-loaded into a base model on Apple Silicon.

In practice, that means:

- **No Network I/O**: expensive external API calls can be transformed into local tensor operations
- **Zero VRAM Bloat**: adapters are loaded on demand instead of permanently inflating the model footprint
- **100% Privacy**: execution can stay fully local, with sensitive inputs never leaving the device
- **Fast Skill Switching**: LoRA adapters can be mounted and evicted in milliseconds using an LRU-style cache

SkillPill is designed for a future where local models do not just “chat”, but dynamically acquire and execute distilled skills at runtime.

---

### Vision

Traditional agent systems depend heavily on:

- Python functions
- tool schemas
- API wrappers
- network-bound execution
- orchestration logic around retries, auth, and error handling

That works, but it is slow, brittle, and often privacy-hostile.

SkillPill takes a stronger position: if a tool’s behavior can be represented as high-quality training trajectories, then part of its usefulness can be compressed into a LoRA adapter and executed locally.

This repository is the first step toward that goal.

---

### Current Focus: Phase 1, Skill-Forge

We are **strictly focusing on Phase 1** right now.

Phase 1 is called **Skill-Forge**. Its job is to convert a normal Python-based tool into a high-quality training dataset suitable for LoRA fine-tuning.

The core workflow is:

1. Read a Python tool file and optional README
2. Extract its callable schema, argument types, defaults, and descriptions
3. Generate realistic tool-use trajectories using a cloud LLM
4. Validate those trajectories against the extracted schema
5. Save them as structured training data for downstream MLX LoRA training

This phase matters because bad synthetic data poisons the rest of the pipeline. If the schema is sloppy or the tool-call payloads are inconsistent, the adapter will learn garbage.

---

### PRD Roadmap

#### Phase 1: Skill-Forge, data generation pipeline
Convert a standard Python tool or skill into ShareGPT or ChatML style JSONL training data.

#### Phase 2: LoRA-Trainer
Automate MLX LoRA fine-tuning against the generated dataset, then export adapter weights.

#### Phase 3: MLX-Router
Load the base model once, dynamically mount and evict adapters through an LRU cache, and execute distilled skills locally.

At the moment, this repository implements the foundation of **Phase 1**.

---

### What is Implemented Today?

#### 1. `SchemaExtractor`

`SchemaExtractor` reads a Python source file and extracts a strict schema for a selected top-level function.

It currently supports:

- function name extraction
- docstring extraction
- parameter names
- type hints
- default values
- required vs optional arguments
- basic JSON-schema-like conversion for common Python typing patterns
- optional README summary ingestion
- AST-first parsing for safety and determinism
- runtime `inspect` fallback when AST metadata is incomplete

Supported typing patterns include:

- `str`
- `int`
- `float`
- `bool`
- `list`
- `dict`
- `Optional[...]`
- `Union[...]`
- `Literal[...]`
- selected `pydantic.BaseModel` structures in runtime inspection mode

#### 2. `TrajectoryGenerator`

`TrajectoryGenerator` uses the extracted schema plus an LLM to generate three classes of supervised trajectories:

- **Standard**: valid request, valid tool call, realistic tool observation, final answer
- **Missing Args**: user intent matches the tool but required arguments are missing, so the assistant must ask a clarification question
- **Negative/Chat**: unrelated requests where the assistant should not call the tool at all

This is important because a tool-using model should learn not only **when to call** a tool, but also **when not to call one**, and **when to ask for missing information**.

#### 3. Strict validation with Pydantic

The repository defines typed models for:

- tool schemas
- JSON-schema-like properties
- tool calls
- chat messages
- trajectory batches

Generated tool-call arguments are validated against the extracted parameter schema before output is accepted.

#### 4. Packaging and examples

The repo also includes:

- Python packaging via `pyproject.toml`
- CLI entry points
- example weather tool
- example README input

---

### Project Structure

```text
skillpill/
├── datasets/
├── examples/
│   ├── README.md
│   └── weather_tool.py
├── src/
│   └── skillpill/
│       ├── __init__.py
│       └── forge/
│           ├── __init__.py
│           ├── extractor.py
│           ├── generator.py
│           ├── models.py
│           └── prompts.py
├── findings.md
├── progress.md
├── task_plan.md
├── pyproject.toml
└── README.md
```

---

### Installation

#### Requirements

- Python **3.11+** recommended
- Apple Silicon target in the long-term design
- `pydantic>=2`
- `openai>=1.30`
- `instructor>=1.3`

> Note: if your system `python3` is old, explicitly use Python 3.11. On some Linux machines, `python3` may still point to Python 3.6, which is too old for this codebase.

#### Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Or install dependencies directly:

```bash
pip install pydantic openai instructor
```

---

### Quick Start

#### Extract a schema from a Python tool

```bash
PYTHONPATH=src python -m skillpill.forge.extractor \
  examples/weather_tool.py \
  --readme examples/README.md \
  --function get_weather
```

Expected output is a strict schema object containing:

- tool name
- description
- parameters
- required fields
- default values
- return schema

#### Generate synthetic trajectories

```bash
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python -m skillpill.forge.generator \
  examples/weather_tool.py \
  --readme examples/README.md \
  --function get_weather \
  --model gpt-4o-mini \
  --count 3 \
  --output datasets/get_weather.jsonl
```

This generates:

- 3 standard trajectories
- 3 missing-args trajectories
- 3 negative-chat trajectories

and saves them into a JSONL dataset.

---

### Design Principles

#### 1. AST first, runtime second
AST parsing is safer and more deterministic than blindly importing arbitrary user code. Runtime inspection is used only as a fallback to recover richer annotations when necessary.

#### 2. Schema fidelity matters
The whole pipeline depends on accurate tool signatures. Synthetic tool calls must match the real interface exactly, or fine-tuning quality will collapse.

#### 3. Negative supervision is not optional
A good tool-using model must learn refusal and clarification behavior, not just eager tool invocation.

#### 4. Local-first is the endgame
Cloud LLMs are used here as data generators, not as the final execution engine. The destination is distilled local capability.

---

### Current Limitations

This is still early-stage infrastructure. Known limitations include:

- no `DatasetFormatter` module yet as a separate component
- limited coverage for complex nested types and advanced custom classes
- no Phase 2 training wrapper yet
- no Phase 3 runtime adapter manager yet
- generated trajectories depend on external LLM quality
- schema conversion is intentionally conservative for safety

In other words: the skeleton is real, but the full exoskeleton is still growing. Very on brand.

---

### Next Steps

Recommended next milestones:

1. Add a dedicated `DatasetFormatter`
2. Normalize output into strict ChatML and/or ShareGPT variants
3. Add retry and repair logic for malformed generations
4. Add dataset quality scoring and deduplication
5. Build `train.py` for MLX LoRA fine-tuning
6. Implement the Phase 3 adapter LRU manager

---

### Who is this for?

SkillPill is for people who care about one or more of the following:

- local agent execution
- Apple Silicon inference workflows
- offline-first tools
- turning API skills into trainable behavior
- reducing latency and privacy exposure in agent systems
- MLX-based LoRA workflows

If that sounds niche, yes. That is because it is niche, and niche is where the interesting engineering lives.

---

### Contributing

Contributions, experiments, and design feedback are welcome.

Good contributions would include:

- richer schema extraction
- better dataset validation
- more robust trajectory generation prompts
- MLX training integration
- adapter routing and caching logic
- benchmark scripts and evaluation harnesses

---

### License

License not specified yet.

---

## 中文

### SkillPill 是什么？

**SkillPill** 是一条端到端的蒸馏流水线，用来把传统的 agent skill、Python 工具函数和 API 能力，压缩成极轻量的 **MLX LoRA Adapter**，也就是我们说的 **Pill**。

它的长期目标很直接：不要每次都把能力建立在高延迟、不稳定、泄露隐私风险高的外部工具调用上，而是把这些能力蒸馏成可以在 Apple Silicon 本地热插拔的小权重。

这意味着：

- **No Network I/O**：原本依赖外部 API 的能力，可以被降维成本地张量计算
- **Zero VRAM Bloat**：适配器按需加载，不需要永久撑大模型占用
- **100% Privacy**：敏感数据可以完全留在本地执行
- **毫秒级技能切换**：通过 LRU 缓存池快速挂载和卸载不同技能 LoRA

SkillPill 的核心判断是：未来的本地模型不该只是“会聊天”，而应该能在运行时动态获得、切换并执行蒸馏后的技能。

---

### 项目愿景

传统 Agent 系统通常依赖：

- Python 函数
- 工具 schema
- API 封装
- 网络请求执行
- 认证、重试、错误处理等编排逻辑

这套方式能用，但问题也很明显：慢、脆、依赖外部环境，而且经常牺牲隐私。

SkillPill 的思路更激进一点：如果一个工具的行为可以被表达成高质量训练轨迹，那么这个工具的部分能力就可以被压缩进 LoRA adapter，在本地执行。

这个仓库就是朝这个目标迈出的第一步。

---

### 当前重点：Phase 1，Skill-Forge

我们**当前严格只做 Phase 1**。

Phase 1 叫 **Skill-Forge**，目标是把普通 Python 工具转成适合 LoRA 微调的数据集。

核心流程如下：

1. 读取 Python 工具文件和可选 README
2. 提取函数 schema、参数类型、默认值和描述信息
3. 用云端 LLM 生成真实的工具使用轨迹
4. 用提取出来的 schema 严格校验这些轨迹
5. 输出为后续 MLX LoRA 训练可用的结构化数据

这一阶段非常关键，因为合成数据质量差，后面的整个蒸馏链条都会被污染。schema 不准、tool call 参数不一致，最后训练出来的适配器就只会一本正经地胡说八道。

---

### PRD 路线图

#### Phase 1：Skill-Forge，数据生成流水线
把标准 Python 工具或 skill 转成 ShareGPT / ChatML 风格的 JSONL 训练数据。

#### Phase 2：LoRA-Trainer
自动化执行 MLX LoRA 微调，并导出 adapter 权重。

#### Phase 3：MLX-Router
基座模型常驻内存，通过 LRU 缓存动态挂载和驱逐 adapter，在本地执行蒸馏后的技能。

当前仓库实现的是 **Phase 1 的基础部分**。

---

### 目前已经实现了什么？

#### 1. `SchemaExtractor`

`SchemaExtractor` 会读取 Python 源文件，并为指定的顶层函数提取严格 schema。

当前支持：

- 函数名提取
- docstring 提取
- 参数名提取
- 类型标注提取
- 默认值提取
- 必填 / 可选参数区分
- 常见 Python typing 到 JSON-schema 风格结构的转换
- 读取 README 作为补充语义上下文
- 优先使用 AST，保证安全性和确定性
- AST 信息不足时，回退到运行时 `inspect`

当前覆盖的类型模式包括：

- `str`
- `int`
- `float`
- `bool`
- `list`
- `dict`
- `Optional[...]`
- `Union[...]`
- `Literal[...]`
- 部分 `pydantic.BaseModel` 结构，运行时模式下支持更好

#### 2. `TrajectoryGenerator`

`TrajectoryGenerator` 基于提取出的 schema 和外部 LLM，生成三类监督训练轨迹：

- **Standard**：用户请求完整，tool call 合法，包含工具返回结果和最终回答
- **Missing Args**：用户意图明确，但缺少必填参数，助手应该先追问澄清
- **Negative/Chat**：用户请求与工具无关，助手不应该调用工具

这点很重要，因为一个真正可用的工具型模型，不只是要学会“什么时候调用工具”，还得学会“什么时候别乱调”，以及“参数不够时先问清楚”。

#### 3. 使用 Pydantic 做严格校验

仓库里已经定义了完整的类型模型，用来约束：

- 工具 schema
- JSON-schema 风格字段
- tool call
- chat message
- trajectory batch

所有生成出的 tool-call 参数，都会先按照提取出的 schema 做校验，只有合法结果才会被接受。

#### 4. 基本打包与示例

目前仓库还包含：

- `pyproject.toml` 打包配置
- CLI 入口
- 一个天气工具示例
- 一个 README 示例输入

---

### 项目结构

```text
skillpill/
├── datasets/
├── examples/
│   ├── README.md
│   └── weather_tool.py
├── src/
│   └── skillpill/
│       ├── __init__.py
│       └── forge/
│           ├── __init__.py
│           ├── extractor.py
│           ├── generator.py
│           ├── models.py
│           └── prompts.py
├── findings.md
├── progress.md
├── task_plan.md
├── pyproject.toml
└── README.md
```

---

### 安装说明

#### 环境要求

- 推荐 Python **3.11+**
- 长期目标运行环境是 Apple Silicon
- `pydantic>=2`
- `openai>=1.30`
- `instructor>=1.3`

> 注意：如果你的系统 `python3` 很老，请显式使用 Python 3.11。有些 Linux 机器的 `python3` 还停在 3.6，这对本项目来说已经是出土文物了。

#### 安装方式

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

或者直接装依赖：

```bash
pip install pydantic openai instructor
```

---

### 快速开始

#### 从 Python 工具中提取 schema

```bash
PYTHONPATH=src python -m skillpill.forge.extractor \
  examples/weather_tool.py \
  --readme examples/README.md \
  --function get_weather
```

输出会包含：

- 工具名
- 描述
- 参数定义
- 必填字段
- 默认值
- 返回值 schema

#### 生成合成轨迹数据

```bash
export OPENAI_API_KEY=your_api_key
PYTHONPATH=src python -m skillpill.forge.generator \
  examples/weather_tool.py \
  --readme examples/README.md \
  --function get_weather \
  --model gpt-4o-mini \
  --count 3 \
  --output datasets/get_weather.jsonl
```

这会生成：

- 3 条 standard 轨迹
- 3 条 missing-args 轨迹
- 3 条 negative-chat 轨迹

并保存为 JSONL 数据集。

---

### 设计原则

#### 1. AST 优先，运行时次之
AST 解析比直接 import 任意用户代码更安全、更稳定。只有在 AST 信息不完整时，才使用运行时反射补充信息。

#### 2. Schema 精度决定上限
整个蒸馏链条都建立在真实工具签名之上。只要合成的 tool call 和真实接口不一致，训练质量就会直接塌掉。

#### 3. 负样本监督不是可选项
好用的工具型模型必须学会拒绝、学会追问，不能只学会一件事：看到像请求就抄起工具猛冲。

#### 4. Local-first 才是终局
这里使用云端 LLM 的角色，是数据生成器，不是最终执行引擎。最终目标仍然是本地蒸馏能力。

---

### 当前限制

项目现在仍然处于早期阶段，已知限制包括：

- 还没有独立的 `DatasetFormatter` 模块
- 对复杂嵌套类型和高级自定义类的覆盖仍有限
- 还没有 Phase 2 的训练脚本
- 还没有 Phase 3 的 adapter 管理器
- 轨迹质量依赖外部 LLM 输出质量
- schema 转换策略目前偏保守，以安全和一致性为优先

简单说就是：骨架已经搭好了，但完整外骨骼还在生长。非常龙虾。🦞

---

### 下一步建议

推荐优先做这几件事：

1. 增加独立的 `DatasetFormatter`
2. 标准化输出为严格 ChatML / ShareGPT 格式
3. 对格式错误结果增加重试与修复逻辑
4. 增加数据质量评分与去重
5. 编写 `train.py`，打通 MLX LoRA 微调
6. 实现 Phase 3 的 adapter LRU 管理器

---

### 这个项目适合谁？

SkillPill 主要适合这些方向的人：

- 想做本地 agent 执行
- 关注 Apple Silicon 推理工作流
- 想把工具能力离线化、本地化
- 想把 API 技能转成可训练行为
- 想减少 agent 系统延迟和隐私暴露
- 想结合 MLX 和 LoRA 做技能热插拔

如果这听起来有点偏门，那是因为它确实偏门。而偏门项目往往才最有意思。

---

### 贡献

欢迎贡献代码、实验结果和设计反馈。

比较有价值的贡献方向包括：

- 更丰富的 schema 提取能力
- 更严格的数据集验证
- 更稳健的轨迹生成 prompt
- MLX 训练集成
- adapter 路由与缓存逻辑
- benchmark 与评测脚本

---

### License

暂未指定许可证。
