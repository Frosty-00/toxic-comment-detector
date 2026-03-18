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

## 项目结构

```
gp/
├── app.py                          # FastAPI 后端 & 模型推理逻辑
├── gp.ipynb                        # 数据工程 + 模型训练 + 评估 Notebook
├── requirements.txt                # 完整 Python 依赖
├── templates/
│   └── index.html                  # 前端 Web 界面
├── results/
│   ├── final-model-*/              # 微调后的模型权重目录（Git LFS）
│   ├── thresholds.json             # 每标签最优判定阈值
│   ├── overall_metrics.csv         # 整体评估指标
│   ├── per_label_metrics.csv       # 各标签评估指标
│   └── threshold_strategy_compare.csv
└── data/
    └── train.csv                   # Kaggle 原始训练数据（已包含在仓库中）
```

---

## 环境配置（新电脑完整步骤）

> **重要**：请严格按照顺序执行，每一步都不要跳过。

### 第一步：确认 Python 版本

本项目需要 **Python 3.10.x**。在 PowerShell 中运行：

```powershell
python --version
```

若版本不是 3.10.x，请前往 [python.org](https://www.python.org/downloads/release/python-31011/) 下载安装 Python 3.10，安装时勾选 **"Add Python to PATH"**。

---

### 第二步：解除 PowerShell 执行策略限制

> Windows 默认禁止运行脚本，每次打开新的 PowerShell 窗口都需要执行此命令，才能激活虚拟环境。

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

执行后不会有任何输出，直接进入下一步即可。

---

### 第三步：克隆仓库并进入项目目录

```powershell
git clone https://github.com/Frosty-00/toxic-comment-detector.git
cd toxic-comment-detector
```

若已经克隆过，直接 `cd` 进入目录即可。

---

### 第四步：创建虚拟环境

```powershell
python -m venv .venv
```

这会在项目目录下生成一个 `.venv` 文件夹，存放独立的 Python 环境。

---

### 第五步：激活虚拟环境

```powershell
.venv\Scripts\activate
```

激活成功后，命令行提示符前面会出现 **`(.venv)`** 字样，例如：

```
(.venv) PS D:\SEHS4713 AI APPLICATION\gp>
```

> 每次打开新的终端窗口，都需要先执行**第二步**（执行策略），再执行**第五步**（激活环境）。

---

### 第六步：安装全部依赖

在已激活虚拟环境的终端中执行：

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

此命令会一次性安装所有依赖，包括：
- `pandas`、`scikit-learn`（数据处理）
- `torch`、`transformers[torch]`、`accelerate>=1.1.0`、`datasets`（模型训练）
- `fastapi`、`uvicorn`（Web 后端）
- `ipykernel`、`ipywidgets`（Jupyter 支持）

> 安装时间约 5～15 分钟，取决于网络速度。如果速度很慢，可以使用国内镜像：
> ```powershell
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

---

### 第七步：将虚拟环境注册为 Jupyter Kernel

> 这一步非常关键！如果跳过，Notebook 会使用系统 Python 而不是虚拟环境，导致 `import pandas` 等报错。

```powershell
python -m ipykernel install --user --name=gp-venv --display-name "Python (gp-venv)"
```

---

### 第八步：在 VS Code / Cursor 中选择正确的 Kernel

1. 用 VS Code 或 Cursor 打开 `gp.ipynb`
2. 点击右上角的 **Kernel 选择器**（可能显示为当前 Python 版本或 "Select Kernel"）
3. 选择 **"Python (gp-venv)"**（即第七步注册的 kernel）
4. 如果没有看到，点击 **"Select Another Kernel"** → **"Jupyter Kernel"** → 刷新后选择

> **验证方式**：在 Notebook 第一个 Cell 中运行以下代码，确认 Python 路径指向 `.venv`：
> ```python
> import sys
> print(sys.executable)
> # 输出应包含 .venv，例如：D:\...\gp\.venv\Scripts\python.exe
> ```

---

---

## 运行 Notebook（训练模型）

完成以上配置后，按顺序运行 `gp.ipynb` 的所有 Cell：

1. **数据工程**：加载并清洗 Kaggle 数据，划分训练/测试集
2. **Tokenization**：使用 DistilBERT Tokenizer 处理文本
3. **模型训练**：微调 `distilbert-base-uncased`（约需 10～30 分钟，视 CPU/GPU 性能）
4. **阈值调优**：为每个标签单独优化判断阈值
5. **评估**：输出各指标报告，保存至 `results/`

> 训练完成后，模型权重会保存到 `results/final-model-<时间戳>/` 目录下。

---

## 启动 Web 应用

确保虚拟环境已激活（提示符有 `(.venv)`），然后运行：

```powershell
uvicorn app:app --reload
```

启动后访问：
- **Web 界面**：[http://127.0.0.1:8000](http://127.0.0.1:8000)
- **健康检查**：[http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

当健康检查返回 `"ready": true` 时，表示模型加载成功，可以开始使用。

### 启用 GPU 推理（可选）

```powershell
$env:USE_CUDA = "1"
uvicorn app:app --reload
```

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

---

## 常见问题

**Q：激活 `.venv` 时报错"无法加载文件，因为在此系统上禁止运行脚本"？**  
A：先在 PowerShell 中执行（每次新开终端都需要）：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

**Q：运行 Notebook 时报 `ModuleNotFoundError: No module named 'pandas'` 或 `sklearn`？**  
A：说明 Notebook 使用的 Kernel 不是虚拟环境，请检查**第七、八步**，确保已注册并选择 `"Python (gp-venv)"` kernel。验证方法：
```python
import sys; print(sys.executable)  # 应包含 .venv
```

**Q：训练时报 `ImportError` 或 `accelerate` 版本过低？**  
A：在激活虚拟环境后运行：
```powershell
pip install --upgrade "transformers[torch]" "accelerate>=1.1.0"
```

**Q：`/api/health` 返回 `"ready": false`？**  
A：确认 `results/` 下存在包含 `model.safetensors` 的模型目录。若使用 Git LFS 拉取仓库，需执行：
```powershell
git lfs pull
```

**Q：安装依赖时下载很慢？**  
A：使用清华镜像源：
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 技术架构

```
用户浏览器
    │  HTTP
    ▼
FastAPI 后端 (app.py)
    │  文本归一化 + Tokenize
    ▼
DistilBERT 微调模型
(results/final-model-*/)
    │  Per-label Threshold 判定
    ▼
JSON 结果返回前端
```

- **模型**：`distilbert-base-uncased` 微调，多标签分类头（6 输出）
- **训练数据**：Kaggle Toxic Comment Classification（159,571 条，采样 10,000 条训练）
- **推理后端**：FastAPI + Uvicorn
- **前端**：Jinja2 模板渲染单页 Web 应用

---

## 课程信息

- **课程**：SEHS4713 AI Application
- **项目类型**：Group Project
