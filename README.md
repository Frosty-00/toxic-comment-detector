# 有毒评论分类器 — Toxic Comment Classifier

> 基于 DistilBERT 微调的多标签文本毒性检测系统，配合 FastAPI 提供实时 Web 推理界面。

---

## 项目简介

本项目针对网络评论中的有害内容进行自动检测，能够同时识别以下 **6 种毒性类别**：

| 标签 | 含义 |
|---|---|
| `toxic` | 普通毒性言论 |
| `severe_toxic` | 严重毒性言论 |
| `obscene` | 淫秽内容 |
| `threat` | 威胁性言论 |
| `insult` | 侮辱性言论 |
| `identity_hate` | 身份仇恨言论 |

---

## 技术架构

```
用户浏览器
    │  HTTP
    ▼
FastAPI 后端 (app.py)
    │  Tokenize + Inference
    ▼
DistilBERT 微调模型
(results/final-model-*/）
    │  Per-label Threshold
    ▼
JSON 结果返回前端
```

- **模型**：`distilbert-base-uncased` 微调，多标签分类头（6 输出）
- **训练数据**：Kaggle Toxic Comment Classification Challenge（159,571 条，采样 10,000 条训练）
- **推理后端**：FastAPI + Uvicorn
- **前端**：Jinja2 模板渲染的单页 Web 应用

---

## 模型性能

### 整体指标（测试集 2,000 条）

| 维度 | Precision | Recall | F1 |
|---|---|---|---|
| Micro | 0.643 | 0.813 | 0.718 |
| Macro | 0.482 | 0.709 | 0.534 |

### 各标签详细指标（经过阈值调优）

| 标签 | Precision | Recall | F1 | 调优阈值 |
|---|---|---|---|---|
| toxic | 0.858 | 0.819 | 0.838 | 0.43 |
| severe_toxic | 0.338 | 0.857 | 0.485 | 0.08 |
| obscene | 0.793 | 0.850 | 0.821 | 0.32 |
| threat | 0.036 | 0.333 | 0.065 | 0.07 |
| insult | 0.714 | 0.797 | 0.753 | 0.32 |
| identity_hate | 0.150 | 0.600 | 0.240 | 0.12 |

> 相比统一阈值 0.5，每标签独立阈值调优显著提升了稀有类别（`severe_toxic`、`threat`、`identity_hate`）的召回率。

---

## 项目结构

```
gp/
├── app.py                          # FastAPI 后端 & 模型推理逻辑
├── gp.ipynb                        # 数据工程 + 模型训练 + 评估 Notebook
├── requirements.txt                # Python 依赖
├── templates/
│   └── index.html                  # 前端 Web 界面
├── results/
│   ├── final-model-*/              # 微调后的模型权重目录
│   ├── thresholds.json             # 每标签最优判定阈值
│   ├── overall_metrics.csv         # 整体评估指标
│   ├── per_label_metrics.csv       # 各标签评估指标
│   └── threshold_strategy_compare.csv  # 阈值策略对比
└── data/
    └── train.csv                   # 原始 Kaggle 训练数据（需手动下载）
```

---

## 快速开始

### 环境要求

- Python 3.10+
- Windows / macOS / Linux
- （可选）NVIDIA GPU + CUDA，用于加速推理

### 安装与启动

```powershell
# 1. 克隆仓库
git clone <你的仓库地址>
cd gp

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动服务
uvicorn app:app --reload
```

启动后访问：
- **Web 界面**：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- **健康检查**：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

当健康检查接口返回 `"ready": true` 时，表示模型加载成功，可以开始使用。

### 启用 GPU 推理（可选）

```powershell
$env:USE_CUDA = "1"
uvicorn app:app --reload
```

---

## 模型路径自动发现

`app.py` 在启动时会按以下优先级自动查找可用模型：

1. `./results/final-model`（固定路径，最高优先）
2. `./results/final-model-*`（按最后修改时间，取最新）
3. `./results/checkpoint-*`（按 checkpoint 编号，取最大）

每个目录中必须包含 `model.safetensors` 或 `pytorch_model.bin` 才被认为有效。

---

## API 接口说明

### `GET /api/health`

返回模型加载状态。

```json
{
  "ok": true,
  "ready": true,
  "model_path": "results/final-model-20260303-163752",
  "device": "cpu",
  "thresholds": { "toxic": 0.43, "obscene": 0.32, ... }
}
```

### `POST /api/analyze`

输入文本，返回各维度毒性得分。

**请求体：**
```json
{ "text": "你好，今天天气真不错！" }
```

**响应：**
```json
{
  "ok": true,
  "is_toxic": false,
  "scores": { "toxic": 0.012, "severe_toxic": 0.003, ... },
  "hits": { "toxic": false, "severe_toxic": false, ... },
  "thresholds": { "toxic": 0.43, ... },
  "normalized_text": "你好 今天天气真不错"
}
```

---

## 训练流程（gp.ipynb）

1. **数据工程**：从 Kaggle 原始数据集（159,571 条）中采样 10,000 条，执行文本清洗（小写化、去标点、去 IP）并划分 80/20 训练/测试集。
2. **模型微调**：使用 HuggingFace `Trainer` 对 `distilbert-base-uncased` 进行多标签分类微调，BCEWithLogitsLoss 损失函数。
3. **阈值调优**：在测试集上遍历各标签阈值，以最大化 F1 为目标选出每标签最优阈值，保存至 `thresholds.json`。
4. **评估与对比**：对比固定阈值 0.5 与调优阈值的 Precision / Recall / F1，输出对比报告。

---

## 文本预处理（推理时）

推理时对输入文本进行额外的对抗性归一化，以提升对规避攻击的鲁棒性：

- Leet speak 替换（`@→a`、`$→s`、`0→o` 等）
- 常见缩写展开（`wtf → what the fuck`、`stfu → shut the fuck up` 等）
- 去除混淆符号，保留字母数字与空格

---

## 常见问题

**Q：启动后 `/api/health` 返回 `ready: false`？**  
A：请确认 `results/` 下存在包含权重文件（`model.safetensors` 或 `pytorch_model.bin`）的模型目录。模型文件如使用 Git LFS 上传，需确保已正确拉取（`git lfs pull`）。

**Q：安装 PyTorch 速度很慢？**  
A：可指定国内镜像源：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q：`thresholds.json` 缺失会怎样？**  
A：系统会自动回退至所有标签统一使用 0.5 阈值，功能正常但稀有类别检出率可能下降。

---

## 使用 Git LFS 上传大文件

模型权重文件（`.safetensors`）通常超过 100 MB，需使用 Git LFS：

```powershell
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"
git add .gitattributes
git add .
git commit -m "feat: add fine-tuned DistilBERT model"
git push
```

---

## 课程信息

- **课程**：SEHS4713 AI Application
- **项目类型**：Group Project
