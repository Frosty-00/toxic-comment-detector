# Toxic Comment Classifier (FastAPI + DistilBERT)

这个项目提供一个网页接口，用于检测文本是否包含以下 6 类毒性：
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

## 1) 项目结构（关键文件）

- `app.py`：后端接口与模型推理逻辑
- `templates/index.html`：前端页面
- `results/final-model/`：推理模型目录（必须存在权重文件）
- `results/thresholds.json`：每个标签的判定阈值
- `gp.ipynb`：训练/实验 notebook（可选）

## 2) 环境要求

- Python 3.10.x（建议与家里电脑一致）
- Windows PowerShell

## 3) 首次运行（本地/学校电脑）

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

启动后访问：
- 网页：`http://127.0.0.1:8000`
- 健康检查：`http://127.0.0.1:8000/api/health`

当健康检查返回 `ready: true` 时，表示模型加载成功。

## 4) 上传到 GitHub（建议）

如果模型文件（如 `model.safetensors`）较大，请使用 Git LFS：

```powershell
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git add .
git commit -m "upload project for school run"
git push
```

## 5) 学校电脑复现步骤

```powershell
git clone <你的仓库地址>
cd <你的仓库目录>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

然后打开：`http://127.0.0.1:8000/api/health`

## 6) 常见问题

- 若出现“模型未就绪”，请确认：
  - `results/final-model/` 下存在 `model.safetensors`（或 `pytorch_model.bin`）
  - `results/thresholds.json` 存在（缺失时会使用默认阈值 0.5）
- 若学校电脑安装 PyTorch 慢，可先尝试更稳定网络后重试 `pip install -r requirements.txt`

